"""Enhanced WebSocket server for streaming Chatterbox TTS following Resemble's pattern."""
import asyncio
import base64
import json
import logging
import uuid
from typing import Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from ..mtl_tts import ChatterboxMultilingualTTS as ChatterboxTTS

logger = logging.getLogger(__name__)

class SynthesizeStream:
    def __init__(self, tts_model: ChatterboxTTS):
        self.tts_model = tts_model
        self.input_ch = asyncio.Queue()
        # Create a singleton instance of FlushSentinel
        self._flush_sentinel = type('FlushSentinel', (), {})()
    
    async def push_text(self, text: str):
        """Push text to the input channel."""
        await self.input_ch.put(text)
    
    async def flush(self):
        """Flush the current segment."""
        await self.input_ch.put(self._flush_sentinel)
    
    async def _tokenize_input(self, segments_ch: asyncio.Queue):
        """Tokenize text from the input channel to segments."""
        input_stream = None
        while True:
            try:
                item = await self.input_ch.get()
                if isinstance(item, str):
                    if input_stream is None:
                        # new segment (after flush for e.g)
                        input_stream = TextSegmentStream()
                        await segments_ch.put(input_stream)
                    input_stream.push_text(item)
                elif item is self._flush_sentinel:  # Use instance comparison instead of type checking
                    if input_stream is not None:
                        input_stream.end_input()
                    input_stream = None
            except asyncio.CancelledError:
                break
        
        if input_stream is not None:
            input_stream.end_input()

    def __del__(self):
        # Clean up any pending tasks
        pass

class TextSegmentStream:
    def __init__(self):
        self.text_buffer = []
        self.input_ended = False
    
    def push_text(self, text: str):
        """Add text to the buffer."""
        self.text_buffer.append(text)
    
    def end_input(self):
        """Mark the end of input for this segment."""
        self.input_ended = True
    
    def get_text(self) -> str:
        """Get the complete text for this segment."""
        return ''.join(self.text_buffer)

class EnhancedStreamingTTSWebSocketServer:
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
            
            # Create synthesis stream for this connection
            synth_stream = SynthesizeStream(self.tts_model)
            
            try:
                # Run the streaming session
                await self._run_streaming_session(websocket, synth_stream)
            except WebSocketDisconnect:
                logger.info("WebSocket connection closed")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                try:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": str(e)
                    }))
                except:
                    pass
        
        @self.app.get("/health")
        async def health_check():
            return {"status": "ok"}
    
    async def _run_streaming_session(self, websocket: WebSocket, synth_stream: SynthesizeStream):
        """Run a streaming session with separate send/receive tasks."""
        request_id = str(uuid.uuid4())[:8]
        
        # Channel for text segments
        segments_ch = asyncio.Queue()
        
        async def send_task():
            """Task to send messages to the client."""
            try:
                # Start tokenization task
                tokenize_task = asyncio.create_task(synth_stream._tokenize_input(segments_ch))
                
                segment_id = 0
                while True:
                    # Get next text segment
                    try:
                        input_stream = await asyncio.wait_for(segments_ch.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    
                    if input_stream is None:
                        break
                    
                    # Get text from segment
                    text = input_stream.get_text()
                    if not text.strip():
                        continue
                    
                    segment_id += 1
                    current_segment_id = f"{request_id}-{segment_id}"
                    
                    # Send segment start
                    await websocket.send_text(json.dumps({
                        "type": "segment_start",
                        "segment_id": current_segment_id,
                        "text": text
                    }))
                    
                    try:
                        # Generate and stream audio chunks
                        audio_chunk_count = 0
                        async for audio_chunk in self.tts_model.stream_generate(
                            text=text,
                            sample_rate=24000,  # Default to 24kHz, can be configured
                            chunk_size=50,
                        ):
                            audio_chunk_count += 1
                            logger.info(f"Generated audio chunk #{audio_chunk_count}")
                            
                            # Convert tensor to bytes (32-bit float PCM)
                            # Ensure proper tensor handling to avoid size mismatches
                            if torch.is_tensor(audio_chunk):
                                # Handle different tensor shapes
                                if audio_chunk.dim() == 0:
                                    # Scalar tensor
                                    audio_data = audio_chunk.unsqueeze(0).detach().cpu().numpy().astype(np.float32)
                                elif audio_chunk.dim() == 1:
                                    # 1D tensor
                                    audio_data = audio_chunk.detach().cpu().numpy().astype(np.float32)
                                elif audio_chunk.dim() == 2:
                                    # 2D tensor, flatten it
                                    audio_data = audio_chunk.flatten().detach().cpu().numpy().astype(np.float32)
                                elif audio_chunk.dim() == 3:
                                    # 3D tensor, flatten it
                                    audio_data = audio_chunk.flatten().detach().cpu().numpy().astype(np.float32)
                                else:
                                    # Higher dimensional tensor, flatten it
                                    audio_data = audio_chunk.flatten().detach().cpu().numpy().astype(np.float32)
                            else:
                                audio_data = np.array(audio_chunk, dtype=np.float32)
                            
                            logger.info(f"Audio data shape: {audio_data.shape}, size: {audio_data.size}")
                            
                            # Ensure we don't send empty data
                            if audio_data.size > 0:
                                try:
                                    await websocket.send_bytes(audio_data.tobytes())
                                    logger.info(f"Sent audio chunk #{audio_chunk_count} ({len(audio_data.tobytes())} bytes)")
                                except Exception as send_error:
                                    logger.error(f"Error sending audio data: {send_error}")
                                    continue
                            else:
                                logger.warning(f"Skipping empty audio chunk #{audio_chunk_count}")
                    
                    except Exception as e:
                        logger.error(f"Generation error for segment {current_segment_id}: {e}")
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "segment_id": current_segment_id,
                            "message": f"Generation failed: {str(e)}"
                        }))
                        continue
                    
                    logger.info(f"Completed audio generation for segment {current_segment_id} with {audio_chunk_count} chunks")
                    
                    # Send segment end
                    await websocket.send_text(json.dumps({
                        "type": "segment_end",
                        "segment_id": current_segment_id
                    }))
                
                # Clean up tokenization task
                tokenize_task.cancel()
                try:
                    await tokenize_task
                except asyncio.CancelledError:
                    pass
                    
            except Exception as e:
                logger.error(f"Send task error: {e}")
                raise
        
        async def recv_task():
            """Task to receive messages from the client."""
            try:
                while True:
                    data = await websocket.receive()
                    
                    if data["type"] == "websocket.disconnect":
                        break
                    elif data["type"] == "websocket.receive":
                        if "text" in data:
                            try:
                                # Handle potential control characters in JSON
                                text_data = data["text"].strip()
                                # Clean up any control characters that might cause issues
                                cleaned_text = ''.join(char for char in text_data if ord(char) >= 32 or char in '\n\r\t')
                                message = json.loads(cleaned_text)
                            except json.JSONDecodeError as e:
                                logger.error(f"JSON decode error: {e}")
                                logger.error(f"Problematic text: {data['text']}")
                                continue
                            msg_type = message.get("type")
                            msg_type = message.get("type")
                            
                            if msg_type == "generate":
                                text = message.get("text", "")
                                if text:
                                    await synth_stream.push_text(text)
                            elif msg_type == "flush":
                                await synth_stream.flush()
                            elif msg_type == "ping":
                                try:
                                    await websocket.send_text(json.dumps({"type": "pong"}))
                                except Exception as e:
                                    logger.error(f"Error sending pong: {e}")
                        elif "bytes" in data:
                            # Handle binary data if needed
                            pass
            except Exception as e:
                logger.error(f"Receive task error: {e}")
                raise
        
        # Run send and receive tasks concurrently
        send_task_handle = asyncio.create_task(send_task())
        recv_task_handle = asyncio.create_task(recv_task())
        
        try:
            await asyncio.gather(send_task_handle, recv_task_handle)
        except Exception as e:
            logger.error(f"Streaming session error: {e}")
        finally:
            # Cancel tasks if they're still running
            for task in [send_task_handle, recv_task_handle]:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
    
    def run(self, **kwargs):
        """Start the WebSocket server."""
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            **kwargs
        )

# Convenience function to create and run the server
def create_enhanced_streaming_server(tts_model: ChatterboxTTS, host: str = "0.0.0.0", port: int = 8000):
    """
    Create an enhanced streaming TTS WebSocket server following Resemble's pattern.
    
    Args:
        tts_model: Initialized ChatterboxTTS model
        host: Host to bind to
        port: Port to listen on
        
    Returns:
        EnhancedStreamingTTSWebSocketServer instance
    """
    return EnhancedStreamingTTSWebSocketServer(tts_model, host, port)