import asyncio
import random
import threading
import numpy as np
import torch
import sys
import os

# Add the project root and chatterbox directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
chatterbox_path = os.path.join(project_root, 'chatterbox', 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, chatterbox_path)

from chatterbox.mtl_tts import ChatterboxMultilingualTTS as ChatterboxTTS
# from chatterbox.tts import ChatterboxTTS as ChatterboxTTS
from chatterbox.streaming.enhanced_websocket_server import create_enhanced_streaming_server
import gradio as gr
import spaces

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running on device: {DEVICE}")

# --- Global Model Initialization ---
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

# Attempt to load the model at startup.
try:
    get_or_load_model()
except Exception as e:
    print(f"CRITICAL: Failed to load model on startup. Application may not function. Error: {e}")

# WebSocket server (initialized lazily)
WEBSOCKET_SERVER = None
WEBSOCKET_THREAD = None

def set_seed(seed: int):
    """Sets the random seed for reproducibility across torch, numpy, and random."""
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def start_websocket_server(host="0.0.0.0", port=8001):
    """Start the WebSocket server in a separate thread."""
    global WEBSOCKET_SERVER, WEBSOCKET_THREAD
    
    if WEBSOCKET_SERVER is not None:
        print("WebSocket server is already running.")
        return
    
    current_model = get_or_load_model()
    if current_model is None:
        raise RuntimeError("TTS model is not loaded.")
    
    print(f"Starting WebSocket server on {host}:{port}...")
    WEBSOCKET_SERVER = create_enhanced_streaming_server(current_model, host, port)
    
    def run_server():
        WEBSOCKET_SERVER.run()
    
    WEBSOCKET_THREAD = threading.Thread(target=run_server, daemon=True)
    WEBSOCKET_THREAD.start()
    print("WebSocket server started.")

def launch_app(colab_mode=False, share=False, server_port=7860, ws_port=8001):
    """
    Launch the application with both Gradio and WebSocket servers.
    
    Args:
        colab_mode: Whether to run in Colab mode with public endpoints
        share: Whether to create public URLs (for Colab)
        server_port: Port for Gradio interface
        ws_port: Port for WebSocket server
    """
    # Start WebSocket server
    start_websocket_server(port=ws_port)
    
    # Launch Gradio app
    if colab_mode:
        print("Launching in Colab mode with public endpoints...")
        demo.launch(share=share, server_port=server_port, prevent_thread_lock=True)
        
        # Print connection information
        if share:
            # Wait a moment for Gradio to generate the public URL
            import time
            time.sleep(2)
            print(f"🔗 Gradio interface will be available at the public URL provided by Gradio")
            print(f"🔗 WebSocket endpoint: ws://localhost:{ws_port}/ws/tts (use the same host as Gradio)")
        else:
            print(f"🔗 Gradio interface: http://localhost:{server_port}")
            print(f"🔗 WebSocket endpoint: ws://localhost:{ws_port}/ws/tts")
    else:
        print("Launching in local mode...")
        demo.launch(server_port=server_port, prevent_thread_lock=True)
        print(f"🔗 Gradio interface: http://localhost:{server_port}")
        print(f"🔗 WebSocket endpoint: ws://localhost:{ws_port}/ws/tts")

@spaces.GPU
def generate_tts_audio(
    text_input: str,
    audio_prompt_path_input: str = None,
    exaggeration_input: float = 0.5,
    temperature_input: float = 0.8,
    seed_num_input: int = 0,
    cfgw_input: float = 0.5
) -> tuple[int, np.ndarray]:
    """
    Generate high-quality speech audio from text using ChatterboxTTS model with optional reference audio styling.
    
    This tool synthesizes natural-sounding speech from input text. When a reference audio file 
    is provided, it captures the speaker's voice characteristics and speaking style. The generated audio 
    maintains the prosody, tone, and vocal qualities of the reference speaker, or uses default voice if no reference is provided.

    Args:
        text_input (str): The text to synthesize into speech (maximum 300 characters)
        audio_prompt_path_input (str, optional): File path or URL to the reference audio file that defines the target voice style. Defaults to None.
        exaggeration_input (float, optional): Controls speech expressiveness (0.25-2.0, neutral=0.5, extreme values may be unstable). Defaults to 0.5.
        temperature_input (float, optional): Controls randomness in generation (0.05-5.0, higher=more varied). Defaults to 0.8.
        seed_num_input (int, optional): Random seed for reproducible results (0 for random generation). Defaults to 0.
        cfgw_input (float, optional): CFG/Pace weight controlling generation guidance (0.2-1.0). Defaults to 0.5.

    Returns:
        tuple[int, np.ndarray]: A tuple containing the sample rate (int) and the generated audio waveform (numpy.ndarray)
    """
    current_model = get_or_load_model()

    if current_model is None:
        raise RuntimeError("TTS model is not loaded.")

    if seed_num_input != 0:
        set_seed(int(seed_num_input))

    print(f"Generating audio for text: '{text_input[:50]}...'")
    
    # Handle optional audio prompt
    generate_kwargs = {
        "language_id": "en",  # Default to English
        "exaggeration": exaggeration_input,
        "temperature": temperature_input,
        "cfg_weight": cfgw_input,
        "repetition_penalty": 2.0,
        "min_p": 0.05,
        "top_p": 1.0,
    }
    
    if audio_prompt_path_input:
        generate_kwargs["audio_prompt_path"] = audio_prompt_path_input
    
    wav = current_model.generate(
        text_input[:300],  # Truncate text to max chars
        **generate_kwargs
    )
    print("Audio generation complete.")
    return (current_model.sr, wav.squeeze(0).numpy())

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Chatterbox TTS Demo
        Generate high-quality speech from text with reference audio styling.
        """
    )
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                value="Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. A sprinkle of olive oil and some tomato ketchup. Now smell that. Oh boy this is going to be incredible.",
                label="Text to synthesize (max chars 300)",
                max_lines=5
            )
            ref_wav = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Reference Audio File (Optional)",
                value="https://storage.googleapis.com/chatterbox-demo-samples/prompts/female_shadowheart4.flac"
            )
            exaggeration = gr.Slider(
                0.25, 2, step=.05, label="Exaggeration (Neutral = 0.5, extreme values can be unstable)", value=.5
            )
            cfg_weight = gr.Slider(
                0.2, 1, step=.05, label="CFG/Pace", value=0.5
            )

            with gr.Accordion("More options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 5, step=.05, label="Temperature", value=.8)

            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Output Audio")

    run_btn.click(
        fn=generate_tts_audio,
        inputs=[
            text,
            ref_wav,
            exaggeration,
            temp,
            seed_num,
            cfg_weight,
        ],
        outputs=[audio_output],
    )

import argparse

def main():
    # Check if running in Colab
    try:
        import google.colab
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Chatterbox TTS Demo")
    parser.add_argument("--share", action="store_true", help="Create public URLs for Colab")
    parser.add_argument("--server-port", type=int, default=7860, help="Port for Gradio interface")
    parser.add_argument("--ws-port", type=int, default=8001, help="Port for WebSocket server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for WebSocket server")
    
    args = parser.parse_args()
    
    # Launch the application
    if IN_COLAB or args.share:
        # Launch in Colab mode with public endpoints
        launch_app(colab_mode=True, share=args.share, server_port=args.server_port, ws_port=args.ws_port)
    else:
        # Launch in local mode
        launch_app(colab_mode=False, server_port=args.server_port, ws_port=args.ws_port)
    
    # Keep the main thread alive
    try:
        import asyncio
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        print("Shutting down...")

if __name__ == "__main__":
    main()