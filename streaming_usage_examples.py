#!/usr/bin/env python3
"""
Streaming Usage Example

This script demonstrates how to use the streaming capabilities of Chatterbox TTS.
"""

import asyncio
import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS
import torch

async def basic_streaming_example():
    """Demonstrate basic streaming usage"""
    print("=== Basic Streaming Example ===")
    
    try:
        # Initialize model (use CPU for demo)
        print("Initializing model...")
        model = ChatterboxMultilingualTTS.from_pretrained("cpu")
        print("Model initialized successfully!")
        
        # For demo purposes, we'll create mock conditionals
        # In a real application, you would use:
        # model.prepare_conditionals("reference.wav")
        
        # Create mock conditionals
        from src.chatterbox.models.t3.modules.cond_enc import T3Cond
        mock_ref_dict = {
            'prompt_token': torch.zeros(1, 10, dtype=torch.long),
            'prompt_token_len': torch.tensor([10]),
            'prompt_feat': torch.zeros(1, 10, 80),
            'prompt_feat_len': None,
            'embedding': torch.zeros(1, 192)  # Fix: Should be 192 dimensions to match spk_embed_dim
        }
        
        t3_cond = T3Cond(
            speaker_emb=torch.zeros(1, 256),
            clap_emb=None,
            cond_prompt_speech_tokens=None,
            cond_prompt_speech_emb=None,
            emotion_adv=torch.tensor([[[0.5]]])
        )
        
        model.conds = type('Conditionals', (), {
            't3': t3_cond,
            'gen': mock_ref_dict
        })()
        
        # Stream generate audio
        text = "Hello, this is a demonstration of the Chatterbox streaming TTS system. "
        text += "You can use this for real-time voice agents and low-latency applications."
        
        print(f"Generating speech for: {text}")
        print("Streaming audio chunks...")
        
        chunk_count = 0
        async for audio_chunk in model.stream_generate(
            text=text,
            language_id="en",
            chunk_size=25,  # Small chunk size for demo
            sample_rate=16000
        ):
            chunk_count += 1
            print(f"Chunk {chunk_count}: {len(audio_chunk)} audio samples")
            
            # In a real application, you would:
            # 1. Send audio_chunk to an audio player
            # 2. Transmit over WebSocket
            # 3. Process for voice agent
            
            # For demo, we'll stop after a few chunks
            if chunk_count >= 5:
                break
        
        print(f"Streaming completed! Processed {chunk_count} chunks.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

async def websocket_server_example():
    """Demonstrate WebSocket server usage"""
    print("\n=== WebSocket Server Example ===")
    print("To run the WebSocket server, use:")
    print("  python ws_server_only.py --port 8001 --host 0.0.0.0")
    print("")
    print("Then connect with a WebSocket client to ws://localhost:8001/ws/tts")
    print("Send JSON messages like:")
    print('  {"type": "generate", "text": "Hello world", "language_id": "en"}')
    print("Receive audio chunks as binary data")

async def gradio_example():
    """Demonstrate Gradio usage"""
    print("\n=== Gradio Example ===")
    print("To run the Gradio interface, use:")
    print("  python multilingual_app.py")
    print("This will start a web server with a user-friendly TTS interface")

async def livekit_integration_example():
    """Demonstrate LiveKit integration"""
    print("\n=== LiveKit Integration Example ===")
    print("To integrate with LiveKit:")
    print("1. Start the WebSocket server:")
    print("   python ws_server_only.py --port 8001")
    print("")
    print("2. In your LiveKit agent, connect and stream:")
    print("""
import websockets
import json
import numpy as np

async def livekit_tts_handler(text):
    uri = "ws://localhost:8001/ws/tts"
    async with websockets.connect(uri) as websocket:
        # Send generation request
        request = {
            "type": "generate",
            "text": text,
            "language_id": "en",
            "sample_rate": 24000
        }
        await websocket.send(json.dumps(request))
        
        # Receive audio chunks
        while True:
            message = await websocket.recv()
            if isinstance(message, bytes):
                # Convert bytes to audio samples
                audio_chunk = np.frombuffer(message, dtype=np.float32)
                # Send to LiveKit participant
                yield audio_chunk
""")

async def main():
    """Main demonstration function"""
    print("Chatterbox TTS Streaming Usage Examples")
    print("=" * 50)
    
    await basic_streaming_example()
    await websocket_server_example()
    await gradio_example()
    await livekit_integration_example()
    
    print("\n" + "=" * 50)
    print("For more details, see:")
    print("- README_EXTENDED.md for comprehensive documentation")
    print("- test/README.md for testing information")
    print("- TENSOR_FIXES_SUMMARY.md for tensor fixes details")

if __name__ == "__main__":
    asyncio.run(main())