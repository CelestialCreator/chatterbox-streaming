\"\"\"WebSocket server for streaming Chatterbox TTS.\"\"\"
import asyncio
import json
import logging
from typing import Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from ..tts import ChatterboxTTS

logger = logging.getLogger(__name__)

class StreamingTTSWebSocketServer:
    def __init__(self, tts_model: ChatterboxTTS, host: str = "0.0.0.0", port: int = 8000):
        self.tts_model = tts_model
        self.host = host
        self.port = port
        self.app = FastAPI(title="Chatterbox Streaming TTS API")
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self):
        @self.app.websocket("/ws/tts")
        async def websocket_tts_endpoint(websocket: WebSocket):
            await websocket.accept()
            logger.info("WebSocket connection established")
            
            try:
                while True:
                    # Receive message from client
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    # Handle different message types
                    if message.get("type") == "generate":
                        await self._handle_generate_request(websocket, message)
                    elif message.get("type") == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))
                    else:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": f"Unknown message type: {message.get('type')}"
                        }))
                        
            except WebSocketDisconnect:
                logger.info("WebSocket connection closed")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": str(e)
                }))
        
        @self.app.get("/health")
        async def health_check():
            return {"status": "ok"}
    
    async def _handle_generate_request(self, websocket: WebSocket, message: dict):
        \"\"\"
        Handle TTS generation request and stream audio chunks back to client.
        
        Expected message format:
        {
            "type": "generate",
            "text": "Hello, world!",
            "audio_prompt_path": "optional/path/to/reference.wav",
            "exaggeration": 0.5,
            "temperature": 0.8,
            "cfg_weight": 0.5,
            "chunk_size": 50,
            "sample_rate": 24000
        }
        \"\"\"
        try:
            # Extract parameters
            text = message.get("text", "")
            if not text:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Text is required"
                }))
                return
            
            # Optional parameters with defaults
            audio_prompt_path = message.get("audio_prompt_path")
            exaggeration = message.get("exaggeration", 0.5)
            temperature = message.get("temperature", 0.8)
            cfg_weight = message.get("cfg_weight", 0.5)
            chunk_size = message.get("chunk_size", 50)
            sample_rate = message.get("sample_rate", 24000)
            
            # Validate sample rate
            if sample_rate not in [16000, 24000, 48000]:
                sample_rate = 24000  # Default to 24kHz
            
            # Send start message
            await websocket.send_text(json.dumps({
                "type": "generation_start",
                "text": text
            }))
            
            # Generate and stream audio chunks
            chunk_count = 0
            async for audio_chunk in self.tts_model.stream_generate(
                text=text,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                temperature=temperature,
                cfg_weight=cfg_weight,
                chunk_size=chunk_size,
                sample_rate=sample_rate,
            ):
                # Convert tensor to bytes
                audio_bytes = audio_chunk.cpu().numpy().astype(np.float32).tobytes()
                
                # Send audio chunk as binary data
                await websocket.send_bytes(audio_bytes)
                chunk_count += 1
            
            # Send completion message
            await websocket.send_text(json.dumps({
                "type": "generation_complete",
                "chunk_count": chunk_count
            }))
            
            logger.info(f"Generated {chunk_count} audio chunks for text: {text[:50]}...")
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Generation failed: {str(e)}"
            }))
    
    def run(self, **kwargs):
        \"\"\"Start the WebSocket server.\"\"\"
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            **kwargs
        )

# Convenience function to create and run the server
def create_streaming_server(tts_model: ChatterboxTTS, host: str = "0.0.0.0", port: int = 8000):
    \"\"\"
    Create a streaming TTS WebSocket server.
    
    Args:
        tts_model: Initialized ChatterboxTTS model
        host: Host to bind to
        port: Port to listen on
        
    Returns:
        StreamingTTSWebSocketServer instance
    \"\"\"
    return StreamingTTSWebSocketServer(tts_model, host, port)