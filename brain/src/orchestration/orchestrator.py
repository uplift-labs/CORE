import asyncio
import logging
from typing import AsyncGenerator
from queue import Queue, Empty
import threading
import warnings

from autogen_core import CancellationToken
from agents.mcp_agent import MCPAgent
from memory.memory_manager import MemoryManager
from autogen_agentchat.messages import TextMessage, ModelClientStreamingChunkEvent
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import UserMessage
from datetime import datetime
from orchestration.system_tool import set_access_token, set_device_id, set_user_id

# Import OpenAI exceptions for better error handling
try:
    from openai import APIConnectionError, APIError
except ImportError:
    APIConnectionError = Exception
    APIError = Exception

# Configure logger
logger = logging.getLogger(__name__)

# Suppress verbose third-party library logs
logging.getLogger("autogen_core").setLevel(logging.WARNING)
logging.getLogger("autogen_core.events").setLevel(logging.WARNING)
logging.getLogger("autogen_agentchat").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("websockets").setLevel(logging.WARNING)

# Suppress "Task exception was never retrieved" warnings
warnings.filterwarnings("ignore", message=".*Task exception was never retrieved.*")
warnings.filterwarnings("ignore", message=".*Event loop is closed.*")


class AgentOrchestrator:
    """Orchestrates multi-agent collaboration using custom RuleBasedGroupChat."""

    def __init__(self, user_id: str = "dhruv", access_token: str = None, max_turns: int = 4, device_id: str = None):
        """Initialize orchestrator with user context and agents."""
        self.user_id = user_id
        self.max_turns = max_turns
        try:
            self.memory = MemoryManager(user_id=user_id)
            self.access_token = access_token
            self.device_id = device_id
            # Create shared context for all agents
            self.model_context = BufferedChatCompletionContext(buffer_size=20)
            logger.debug(f"Orchestrator initialized for user: {user_id}")
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator for user {user_id}: {e}", exc_info=True)
            raise

    async def _load_chat_history(self, message: str) -> str:
        """Load previous messages from memory into model context."""
        context_string, previous_messages = self.memory.get_messages(message)
        for msg in previous_messages:
            await self.model_context.add_message(msg)
        return context_string

    def _save_to_memory(self, messages: list) -> None:
        """Save messages to persistent memory."""
        messages = [m for m in messages if m.get(
            "role", "").strip().lower() != "system"]
        if messages:
            self.memory.add_messages(messages)

    def _create_assistant(self, context: str, enable_streaming: bool = False) -> MCPAgent:
        """Initialize the MCP assistant with tools."""
        # Set access token, device_id, and user_id for system tool operations
        if self.access_token:
            set_access_token(self.access_token)
        if self.device_id:
            set_device_id(self.device_id)
        if self.user_id:
            set_user_id(self.user_id)
        # Lazy import to avoid circular dependency
        from orchestration.system_tool import execute_system_intent
        
        kwargs = {
            "name": "assistant",
            "context": context,
            "tools": [execute_system_intent],
            "model_context": self.model_context,
        }
        
        # Enable streaming if requested
        if enable_streaming:
            kwargs["model_client_stream"] = True
        
        self.assistant = MCPAgent(**kwargs)
        return self.assistant

    def _normalize_messages(self, user_message: TextMessage, assistant_message: TextMessage) -> list:
        """Convert messages to memory format compatible with MemoryManager."""
        normalized = []
        if user_message and getattr(user_message, "content", "").strip():
            normalized.append({
                "role": user_message.source,
                "name": user_message.source,
                "content": user_message.content.replace("TERMINATE", "").strip(),
                "timestamp": getattr(user_message, "created_at", datetime.now()),
            })
        if assistant_message and getattr(assistant_message, "content", "").strip():
            normalized.append({
                "role": assistant_message.source,
                "name": assistant_message.source,
                "content": assistant_message.content.replace("TERMINATE", "").strip(),
                "timestamp": getattr(assistant_message, "created_at", datetime.now()),
            })
        return normalized

    async def start_chat_stream_async(self, task: str) -> AsyncGenerator[str, None]:
        """
        Process user message through rule-based agent orchestration with streaming.
        Yields chunks of the response as they are generated.

        Flow:
        1. Load previous conversation history
        2. Stream assistant response
        3. Save messages to memory
        4. Yield response chunks
        """
        assistant = None
        full_response = ""
        task_message = None
        
        try:
            task_message = TextMessage(
                content=task, source="user", created_at=datetime.now())
            
            # Step 1: Load previous conversation
            context = await self._load_chat_history(task)
            # Step 2: Add user message to context
            await self.model_context.add_message(UserMessage(content=task, source="user"))

            # Step 3: Create assistant with tool access and streaming enabled
            assistant = self._create_assistant(context, enable_streaming=True)

            # Step 4: Stream the assistant response
            cancellation_token = CancellationToken()
            tool_responses = []  # Track tool responses to include in final message
            
            async for event in assistant.on_messages_stream([task_message], cancellation_token):
                # Handle streaming chunk events
                if isinstance(event, ModelClientStreamingChunkEvent):
                    chunk = event.content
                    if chunk:
                        full_response += chunk
                        yield chunk
                
                # Handle tool execution events
                elif hasattr(event, 'chat_message') and event.chat_message:
                    message = event.chat_message
                    
                    # Check for tool-related messages by inspecting attributes
                    # Tool responses might be embedded in TextMessage or have specific attributes
                    if hasattr(message, 'tool_call_id') or hasattr(message, 'tool_calls') or 'tool' in str(type(message)).lower():
                        tool_result = getattr(message, 'content', '') or str(message)
                        if tool_result:
                            tool_responses.append(tool_result)
                            logger.debug(f"Tool response captured for user {self.user_id}")
                            # Yield tool response immediately so user sees it
                            yield f"\n{tool_result}\n"
                    
                    # Handle final text response
                    if isinstance(message, TextMessage):
                        message_content = message.content
                        message_str = str(message_content) if message_content else ""
                        
                        # Always use the complete final message content (includes tool results)
                        if message_str and message_str != full_response:
                            # If we have accumulated chunks, check if final message has more
                            if full_response and len(message_str) > len(full_response):
                                # Final message has more content (likely includes tool results)
                                remaining = message_str[len(full_response):]
                                if remaining:
                                    logger.debug(f"Yielding remaining content: {len(remaining)} chars")
                                    yield remaining
                                    full_response = message_str
                            elif not full_response:
                                # No chunks accumulated, use full message
                                full_response = message_str
                                yield message_str
                            else:
                                # Final message might be the same or shorter, ensure we have the complete one
                                full_response = message_str
                
                # Handle any other message types
                elif hasattr(event, 'chat_message'):
                    message = event.chat_message
                    if hasattr(message, 'content') and message.content:
                        content = str(message.content)
                        if content not in full_response:
                            full_response += content
                            yield content

            # Include tool responses in the final response if they weren't already included
            if tool_responses and full_response:
                # Check if tool responses are already in the full_response
                for tool_response in tool_responses:
                    if tool_response and tool_response not in full_response:
                        full_response += tool_response
            
            # Save to memory after streaming completes
            if task_message and full_response:
                try:
                    assistant_message = TextMessage(
                        content=full_response, 
                        source="assistant", 
                        created_at=datetime.now()
                    )
                    normalized = self._normalize_messages(task_message, assistant_message)
                    self._save_to_memory(normalized)
                    logger.debug(f"Saved conversation to memory for user: {self.user_id}")
                except Exception as e:
                    logger.error(f"Failed to save conversation to memory for user {self.user_id}: {e}", exc_info=True)

        except (APIConnectionError, ConnectionError) as e:
            logger.error(f"Connection error in streaming for user {self.user_id}: {e}", exc_info=True)
            yield "Connection error: Unable to connect to the AI service. Please check your network connection and API configuration."
        except APIError as e:
            logger.error(f"API error in streaming for user {self.user_id}: {e}", exc_info=True)
            yield f"API error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in streaming for user {self.user_id}: {e}", exc_info=True)
            yield f"Error: An unexpected error occurred"

    def start_chat_stream(self, message: str):
        """Sync wrapper that creates event loop for streaming using a queue."""
        queue = Queue()
        exception_holder = [None]
        done = threading.Event()
        
        def run_async_generator():
            """Run the async generator in a separate thread with its own event loop."""
            loop = None
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                async def consume():
                    try:
                        async for chunk in self.start_chat_stream_async(message):
                            queue.put(('chunk', chunk))
                        queue.put(('done', None))
                    except Exception as e:
                        logger.error(f"Error in async generator for user {self.user_id}: {e}", exc_info=True)
                        exception_holder[0] = e
                        queue.put(('error', str(e)))
                    finally:
                        done.set()
                
                loop.run_until_complete(consume())
            except Exception as e:
                exception_holder[0] = e
                queue.put(('error', str(e)))
                done.set()
            finally:
                if loop:
                    try:
                        # Give a small delay for HTTP clients to finish cleanup
                        import time
                        time.sleep(0.1)
                        
                        # Get all pending tasks
                        pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
                        
                        # Cancel all pending tasks
                        for task in pending:
                            task.cancel()
                        
                        # Wait for tasks to complete (with timeout)
                        if pending:
                            try:
                                loop.run_until_complete(
                                    asyncio.wait_for(
                                        asyncio.gather(*pending, return_exceptions=True),
                                        timeout=1.0
                                    )
                                )
                            except (asyncio.TimeoutError, RuntimeError):
                                # Ignore timeout and "Event loop is closed" errors during cleanup
                                pass
                            except Exception:
                                pass
                        
                        # Close the loop
                        try:
                            loop.close()
                        except RuntimeError:
                            # Ignore "Event loop is closed" errors
                            pass
                    except Exception:
                        # Suppress all cleanup errors
                        pass
        
        # Start the async generator in a background thread
        thread = threading.Thread(target=run_async_generator, daemon=True)
        thread.start()
        
        # Yield chunks from the queue
        while True:
            try:
                item_type, item = queue.get(timeout=0.1)
                if item_type == 'chunk':
                    yield item
                elif item_type == 'done':
                    break
                elif item_type == 'error':
                    yield f"Error: {item}"
                    break
            except Empty:
                if done.is_set():
                    # Check if there are any remaining items
                    try:
                        item_type, item = queue.get_nowait()
                        if item_type == 'chunk':
                            yield item
                        elif item_type == 'error':
                            yield f"Error: {item}"
                    except Empty:
                        pass
                    break
                continue
        
        # Check for exceptions
        if exception_holder[0]:
            raise exception_holder[0]