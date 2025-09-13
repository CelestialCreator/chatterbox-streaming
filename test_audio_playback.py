#!/usr/bin/env python3
"""
Audio playback verification script
"""
import asyncio
import json
import websockets
import numpy as np
import sounddevice as sd

async def test_audio_playback():
    uri = "ws://localhost:8001/ws/tts"
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to WebSocket server")
            
            # Send a simple generate message
            message = {
                "type": "generate",
                "text": "Hello, this is a test of the streaming audio system."
            }
            await websocket.send(json.dumps(message))
            print("Sent generate message")
            
            # Send flush message
            flush_msg = {"type": "flush"}
            await websocket.send(json.dumps(flush_msg))
            print("Sent flush message")
            
            # Audio playback parameters
            sample_rate = 24000  # Match the server's sample rate
            audio_buffer = []
            
            # Listen for responses
            try:
                async for message in websocket:
                    if isinstance(message, str):
                        data = json.loads(message)
                        msg_type = data.get("type")
                        print(f"Received text message: {msg_type}")
                        
                        if msg_type == "segment_start":
                            print(f"Starting segment: {data.get('text')}")
                            audio_buffer = []  # Reset buffer for new segment
                        elif msg_type == "segment_end":
                            print("Segment completed")
                            # Play the accumulated audio
                            if audio_buffer:
                                audio_data = np.concatenate(audio_buffer)
                                print(f"Playing audio with {len(audio_data)} samples at {sample_rate} Hz")
                                sd.play(audio_data, sample_rate)
                                sd.wait()  # Wait for playback to finish
                            break
                        elif msg_type == "error":
                            print(f"Error: {data.get('message')}")
                            break
                    else:
                        # Binary audio data
                        # Convert bytes to numpy array
                        audio_chunk = np.frombuffer(message, dtype=np.float32)
                        audio_buffer.append(audio_chunk)
                        print(f"Received audio chunk with {len(audio_chunk)} samples ({len(message)} bytes)")
                        
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_audio_playback())