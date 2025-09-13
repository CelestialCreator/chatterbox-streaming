#!/usr/bin/env python3
"""
Enhanced WebSocket TTS Server with Public URL Exposure
This script runs the Chatterbox TTS WebSocket server and exposes it publicly using ngrok.
"""
import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def check_ngrok_token():
    """Check if ngrok token is available"""
    token = os.getenv('NGROK_TOKEN') or os.getenv('NGROK_AUTH_TOKEN')
    if not token:
        print("Warning: NGROK_TOKEN not found in environment variables")
        print("To expose your server publicly, set your ngrok token:")
        print("  export NGROK_TOKEN=your_ngrok_token_here")
        print("Get your token from: https://dashboard.ngrok.com/get-started/your-authtoken")
    return token

def start_server(host="0.0.0.0", port=8001, expose=False):
    """Start the WebSocket server with optional public exposure"""
    
    # Import here to avoid issues if dependencies aren't installed yet
    try:
        from src.chatterbox.streaming.enhanced_websocket_server import create_enhanced_streaming_server
        from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS
        import torch
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Make sure you've installed all dependencies with: pip install -e .")
        sys.exit(1)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("🚀 Initializing Chatterbox TTS model...")
    
    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✅ Using GPU: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
        print("✅ Using MPS (Apple Silicon)")
    else:
        device = "cpu"
        print("⚠️  Using CPU (slow - consider using a GPU)")
    
    try:
        # Load the model
        print("📥 Loading model...")
        tts_model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        print("✅ Model loaded successfully")
        
        # Create and start the server
        server = create_enhanced_streaming_server(tts_model, host=host, port=port)
        print(f"🌐 WebSocket server starting on {host}:{port}")
        print("💡 Connect to: ws://localhost:8001/ws/tts")
        
        # Expose publicly if requested
        public_url = None
        if expose:
            token = check_ngrok_token()
            if token:
                try:
                    from pyngrok import ngrok
                    # Set the token
                    ngrok.set_auth_token(token)
                    # Create tunnel
                    public_url = ngrok.connect(port, "http")
                    print(f"🌍 Public URL: {public_url}")
                    print("💡 WebSocket endpoint: {public_url}/ws/tts".format(public_url=public_url))
                except ImportError:
                    print("❌ pyngrok not installed. Install with: pip install pyngrok")
                    print("🔧 Falling back to local server only")
                except Exception as e:
                    print(f"❌ Error setting up ngrok: {e}")
                    print("🔧 Falling back to local server only")
            else:
                print("❌ NGROK_TOKEN not set. Cannot expose server publicly.")
                print("🔧 Falling back to local server only")
        
        # Run the server
        print("🎧 Server is ready!")
        if expose and public_url:
            print(f"🔗 Local: ws://localhost:{port}/ws/tts")
            print(f"🔗 Public: {public_url}/ws/tts")
        else:
            print(f"🔗 WebSocket endpoint: ws://localhost:{port}/ws/tts")
        print("🔢 Send JSON messages with type 'generate' and 'flush'")
        print("⏹️  Press Ctrl+C to stop")
        
        # Run the server
        server.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Chatterbox TTS WebSocket Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8001, help="Port to listen on (default: 8001)")
    parser.add_argument("--expose", action="store_true", help="Expose server publicly using ngrok")
    
    args = parser.parse_args()
    
    # Run the server
    # Since server.run() creates its own event loop, we don't need asyncio.run()
    start_server(args.host, args.port, args.expose)

if __name__ == "__main__":
    main()