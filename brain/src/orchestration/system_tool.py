"""
System Tool for executing user intents via SystemAgent.

This module provides a tool function that can be used with autogen_agentchat
to execute system-level tasks through the SystemAgent.
"""
import json
import asyncio
import logging
import re
from typing import Optional, List, Dict, Any
from autogen_core import CancellationToken
from autogen_agentchat.messages import TextMessage
from agents.system_agent import SystemAgent
from datetime import datetime
from brain_core.send_intent import send_intent
from brain_core.sup_extractor import supabase_service
import requests

logger = logging.getLogger(__name__)


# Create a singleton SystemAgent instance to reuse across tool calls
_system_agent_instance = None

# Cache for user agents (to avoid repeated DB queries)
_user_agents_cache: Dict[str, List[Dict[str, Any]]] = {}

def _get_user_active_agents(user_id: str) -> List[Dict[str, Any]]:
    """Fetch all active agents for a user from Supabase."""
    if user_id in _user_agents_cache:
        return _user_agents_cache[user_id]
    
    try:
        # Get agents from centralized Supabase service
        # Service role key is used by default (bypasses RLS)
        agents = supabase_service.get_user_active_agents(user_id)
        
        # Cache the results
        _user_agents_cache[user_id] = agents
        logger.debug(f"Loaded {len(agents)} active agents for user: {user_id}")
        return agents
    except Exception as e:
        logger.error(f"Failed to fetch active agents for user {user_id}: {e}", exc_info=True)
        return []

def _update_agent_last_used(installed_agent_id: str) -> None:
    """Update the last_used_at timestamp for an installed agent."""
    try:
        supabase_service.update_agent_last_used(installed_agent_id)
    except Exception as e:
        logger.warning(f"Failed to update last_used_at for agent {installed_agent_id}: {e}")

def _get_tool_by_id(tool_id: str, user_id: Optional[str] = None) -> Optional[dict]:
    """Find a tool/agent by its ID from user's active agents."""
    if not user_id:
        return None
    
    agents = _get_user_active_agents(user_id)
    for agent in agents:
        if agent.get("id") == tool_id:
            return agent
    return None

def _get_system_agent(user_id: Optional[str] = None) -> SystemAgent:
    """Get or create the SystemAgent instance with user's active agents."""
    global _system_agent_instance
    
    # Get user's active agents from database
    if user_id:
        agents = _get_user_active_agents(user_id)
        # Convert to tools format for SystemAgent
        tools = [
            {
                "name": agent.get("name", "Unknown"),
                "id": agent.get("id", ""),
                "local": agent.get("local", False)
            }
            for agent in agents
        ]
        
        # Create new SystemAgent with user's tools
        return SystemAgent(name="system_agent", tools=tools)
    _system_agent_instance = SystemAgent(name="system_agent", tools=[])
    return _system_agent_instance


# Global access token storage (set by orchestrator)
_access_token: Optional[str] = None

def set_access_token(token: Optional[str]) -> None:
    """Set the access token for system agent operations."""
    global _access_token
    _access_token = token

# Global device_id storage (set by orchestrator)
_device_id: Optional[str] = None

def set_device_id(token: Optional[str]) -> None:
    """Set the device id for system agent operations."""
    global _device_id
    _device_id = token

# Global user_id storage (set by orchestrator)
_user_id: Optional[str] = None

def set_user_id(user_id: Optional[str]) -> None:
    """Set the user id for system agent operations."""
    global _user_id
    _user_id = user_id
    # Clear cache when user changes to force refresh
    if user_id in _user_agents_cache:
        del _user_agents_cache[user_id]


def extract_intent_from_text(text: str) -> Optional[str]:
    """
    Extract intent from text that contains execute_system_intent function calls.
    
    Handles patterns like:
    - execute_system_intent(intent="open resume")
    - execute_system_intent(intent='open resume')
    - execute_system_intent(intent="open resume")
    
    Returns the extracted intent string, or None if no match found.
    """
    if not text or not isinstance(text, str):
        return None
    
    # Pattern to match execute_system_intent(intent="...") or execute_system_intent(intent='...')
    patterns = [
        r'execute_system_intent\s*\(\s*intent\s*=\s*["\']([^"\']+)["\']\s*\)',
        r'execute_system_intent\s*\(\s*["\']([^"\']+)["\']\s*\)',  # execute_system_intent("intent")
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            intent = match.group(1).strip()
            if intent:
                logger.debug(f"Extracted intent from text: {intent}")
                return intent
    
    return None


def _send_http_request(access_url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> str:
    """Synchronous helper function to send HTTP request."""
    try:
        response = requests.post(
            access_url,
            json=payload,
            headers=headers,
            timeout=30.0
        )
        response.raise_for_status()
        
        # Try to parse JSON response
        try:
            return json.dumps(response.json()) if isinstance(response.json(), dict) else response.text
        except (ValueError, json.JSONDecodeError):
            return response.text
    except requests.exceptions.Timeout:
        raise TimeoutError(f"Request to {access_url} timed out")
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"HTTP request failed: {str(e)}")

async def _send_to_remote_agent(access_url: str, action: str, data: Dict[str, Any], auth_token: Optional[str] = None) -> str:
    """Send action to remote agent via HTTP request."""
    try:
        headers = {
            "Content-Type": "application/json"
        }
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        
        payload = {
            "action": action,
            "data": data
        }
        
        logger.debug(f"Sending action '{action}' to remote agent at {access_url}")
        
        # Run the synchronous request in a thread pool
        response_text = await asyncio.to_thread(
            _send_http_request,
            access_url,
            payload,
            headers
        )
        logger.debug(f"Received response from remote agent at {access_url}")
        return response_text
    
    except (TimeoutError, ConnectionError) as e:
        logger.error(f"Failed to send action to remote agent at {access_url}: {e}")
        return f"Failed to send action to remote agent: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error sending to remote agent at {access_url}: {e}", exc_info=True)
        return f"Error sending to remote agent: {str(e)}"

async def _execute_with_token(user_intent: str) -> str:
    """Internal function to execute intent with optional token."""
    try:
        if not _user_id:
            logger.warning("System intent executed without user_id")
            return "Error: User context not available."
        
        logger.debug(f"Executing system intent for user {_user_id}: {user_intent[:100]}")
        
        # Get the SystemAgent instance with user's active agents
        try:
            system_agent = _get_system_agent(_user_id)
        except Exception as e:
            logger.error(f"Failed to get system agent for user {_user_id}: {e}", exc_info=True)
            return "Error: Failed to initialize system agent."
        
        # Create a TextMessage from the user intent
        intent_message = TextMessage(
            content=user_intent,
            source="user",
            created_at=datetime.now()
        )

        # Create a cancellation token
        cancellation_token = CancellationToken()

        # Call the SystemAgent's on_messages method
        try:
            response = await system_agent.on_messages(
                [intent_message],
                cancellation_token
            )
        except Exception as e:
            logger.error(f"SystemAgent execution failed for user {_user_id}: {e}", exc_info=True)
            return "Error: System agent execution failed."

        # Extract the content from the response
        if not response or not response.chat_message:
            logger.warning(f"No response from SystemAgent for user {_user_id}")
            return "No response generated from SystemAgent."
        
        content = response.chat_message.content
        content_str = content if isinstance(content, str) else str(content)
        
        # Parse the SystemAgent response (should be JSON: { "action": "", "id": "" })
        try:
            action_data = json.loads(content_str.strip())
            action = action_data.get("action", "")
            tool_id = action_data.get("id", "")
            
            # Check if tool exists
            if not tool_id:
                logger.debug(f"SystemAgent response has no tool_id, returning content as-is")
                return content_str
            
            # Get agent details from database
            agent = _get_tool_by_id(tool_id, _user_id)
            if not agent:
                logger.warning(f"Agent {tool_id} not found for user {_user_id}")
                return f"Agent with ID '{tool_id}' not found or not active for this user."
            
            is_local = agent.get("local", False)
            access_url = agent.get("access_url", "")
            agent_name = agent.get("name", tool_id)
            
            logger.info(f"Routing action '{action}' to {'local' if is_local else 'remote'} agent '{agent_name}' (ID: {tool_id})")
            
            # Update last_used_at timestamp
            installed_agent_id = agent.get("installed_agent_id")
            if installed_agent_id:
                _update_agent_last_used(installed_agent_id)
            
            # Route based on local/remote
            if is_local:
                # Local agent - send via relay server
                if action and _device_id:
                    try:
                        logger.debug(f"Sending action '{action}' to local agent via device {_device_id}")
                        device_response = await send_intent(
                            device_id=_device_id,
                            action=action,
                            data={"tool_id": tool_id,"user_intent": user_intent},
                            wait_for_response=True,
                            timeout=100.0,
                            auth_token=_access_token
                        )
                        if device_response:
                            logger.info(f"Received response from local agent '{agent_name}'")
                            return f"[Tool:{tool_id}]: {json.dumps(device_response) if isinstance(device_response, dict) else str(device_response)}"
                        else:
                            logger.warning(f"No response received from local agent '{agent_name}'")
                            return f"Action '{action}' sent to local agent {agent_name}, but no response received."
                    except Exception as e:
                        logger.error(f"Failed to send action to local agent '{agent_name}': {e}", exc_info=True)
                        return f"Action '{action}' determined, but failed to send to local agent: {str(e)}"
                else:
                    logger.warning(f"Action '{action}' requires device connection, but no device_id is set")
                    return f"Action '{action}' requires device connection, but no device_id is set."
            else:
                # Remote agent - send HTTP request to access_url
                if access_url:
                    try:
                        logger.debug(f"Sending action '{action}' to remote agent at {access_url}")
                        remote_response = await _send_to_remote_agent(
                            access_url=access_url,
                            action=action,
                            data={"tool_id": tool_id, "agent_id": tool_id, "user_intent": user_intent},
                            auth_token=_access_token
                        )
                        logger.info(f"Received response from remote agent '{agent_name}'")
                        return remote_response
                    except Exception as e:
                        logger.error(f"Failed to send action to remote agent '{agent_name}': {e}", exc_info=True)
                        return f"Failed to send action to remote agent '{agent_name}': {str(e)}"
                else:
                    logger.warning(f"Remote agent '{agent_name}' has no access_url configured")
                    return f"Remote agent '{agent_name}' has no access_url configured."
        
        except json.JSONDecodeError:
            # If response is not JSON, return as-is
            logger.debug(f"SystemAgent response is not JSON, returning as-is")
            return content_str

    except Exception as e:
        logger.error(f"Unexpected error executing system intent for user {_user_id}: {e}", exc_info=True)
        return f"Error executing system intent: {str(e)}"

async def execute_system_intent(intent: str | dict | None = None) -> str:
    """
    Execute a user intent through the SystemAgent.

    This tool takes a user's intent and passes it to the SystemAgent
    for execution. It supports both plain strings and structured JSON calls.
    Also automatically extracts intent from text containing function call syntax.

    Args:
        intent: Either a string (e.g. "open resume folder"), 
                a dict (e.g. { "intent": "open resume folder" }),
                or text containing execute_system_intent(intent="...") syntax.

    Returns:
        str: The response from the SystemAgent after processing the intent.
    """
    try:
        # Handle structured tool call input from model
        if isinstance(intent, dict):
            intent = intent.get("intent") or intent.get("intent")

        # If intent is a string, check if it contains function call syntax
        if isinstance(intent, str):
            # Try to extract intent from function call syntax
            extracted_intent = extract_intent_from_text(intent)
            if extracted_intent:
                logger.info(f"Extracted intent from function call syntax: {extracted_intent}")
                intent = extracted_intent

        if not intent:
            logger.warning("execute_system_intent called with no intent")
            return "Error: No intent provided."

        # Use global access token if available
        return await _execute_with_token(intent)
    except Exception as e:
        logger.error(f"Error in execute_system_intent: {e}", exc_info=True)
        return f"Error: Failed to execute system intent"
