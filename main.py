import sys
import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from core.chat_service import ChatService

# region Configuration and Setup
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress verbose third-party library logs
logging.getLogger("autogen_core").setLevel(logging.WARNING)
logging.getLogger("autogen_core.events").setLevel(logging.WARNING)
logging.getLogger("autogen_agentchat").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("websockets").setLevel(logging.WARNING)

# Add current directory to path first for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Add brain/src directory to path for orchestrator imports
brain_src_path = os.path.join(current_dir, 'brain', 'src')
brain_src_path = os.path.abspath(brain_src_path)
if brain_src_path not in sys.path:
    sys.path.insert(1, brain_src_path)

from core.health_service import health_check
from core.auth_service import require_auth

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
# endregion

@app.route('/health', methods=['GET'])
def health_check_endpoint():
    """Health check endpoint"""
    try:
        return health_check()
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return jsonify({"status": "unhealthy", "error": str(e)}), 500
    
@app.route('/chat', methods=['POST'])
@require_auth
def chat_stream_endpoint():
    """Process chat messages with streaming response (Server-Sent Events)"""
    try:
        user_id = request.user_id
        access_token = request.access_token
        logger.info(f"Chat stream request from user: {user_id}")
        return ChatService().process_chat_stream(user_id, access_token)
    except Exception as e:
        logger.error(f"Chat stream failed for user {request.user_id}: {e}", exc_info=True)
        return jsonify({"error": "Chat streaming failed"}), 500

@app.route('/clear-history', methods=['POST', 'DELETE'])
@require_auth
def clear_history_endpoint():
    """Clear all chat history and memory data for the authenticated user"""
    try:
        user_id = request.user_id
        logger.info(f"Clear history request from user: {user_id}")
        return ChatService().clear_history(user_id)
    except Exception as e:
        logger.error(f"Clear history failed for user {request.user_id}: {e}", exc_info=True)
        return jsonify({"error": "Failed to clear history"}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    logger.warning(f"404 error: {request.url}")
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}", exc_info=True)
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    logger.info("Starting Aven Speech API server...")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)