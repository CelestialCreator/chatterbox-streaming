import asyncio
import sys
import os

# Add the project root and chatterbox directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
chatterbox_path = os.path.join(project_root, 'chatterbox', 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, chatterbox_path)

from chatterbox.mtl_tts import ChatterboxMultilingualTTS as ChatterboxTTS
from chatterbox.streaming.enhanced_websocket_server import create_enhanced_streaming_server
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running on device: {DEVICE}")

# Global Model Initialization
MODEL = None

def get_or_load_model():
    """Loads the ChatterboxTTS model if it hasn't been loaded already,
    and ensures it's on the correct device."""
    global MODEL
    if MODEL is None:
        print("Model not loaded, initializing...")
        try:
            MODEL = ChatterboxTTS.from_pretrained(DEVICE)
            if hasattr(MODEL, 'to') and str(MODEL.device) != DEVICE:
                MODEL.to(DEVICE)
            print(f"Model loaded successfully. Internal device: {getattr(MODEL, 'device', 'N/A')}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    return MODEL

def start_websocket_server(host="0.0.0.0", port=8001):
    """Start the WebSocket server."""
    current_model = get_or_load_model()
    if current_model is None:
        raise RuntimeError("TTS model is not loaded.")
    
    print(f"Starting WebSocket server on {host}:{port}...")
    websocket_server = create_enhanced_streaming_server(current_model, host, port)
    websocket_server.run()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Chatterbox TTS WebSocket Server")
    parser.add_argument("--port", type=int, default=8001, help="Port for WebSocket server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for WebSocket server")
    
    args = parser.parse_args()
    
    start_websocket_server(host=args.host, port=args.port)