<img width="1200" height="600" alt="Chatterbox-Multilingual" src="https://www.resemble.ai/wp-content/uploads/2025/09/Chatterbox-Multilingual-1.png" />

# Chatterbox TTS - Voice Agent Enhanced

[![Alt Text](https://img.shields.io/badge/listen-demo_samples-blue)](https://resemble-ai.github.io/chatterbox_demopage/)
[![Alt Text](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/ResembleAI/Chatterbox)
[![Alt Text](https://static-public.podonos.com/badges/insight-on-pdns-sm-dark.svg)](https://podonos.com/resembleai/chatterbox)
[![Discord](https://img.shields.io/discord/1377773249798344776?label=join%20discord&logo=discord&style=flat)](https://discord.gg/rJq9cRJBJ6)

_Made with ♥️ by <a href="https://resemble.ai" target="_blank"><img width="100" alt="resemble-logo-horizontal" src="https://github.com/user-attachments/assets/35cf756b-3506-4943-9c72-c05ddfa4e525" /></a>

This is an enhanced fork of **Chatterbox Multilingual**, [Resemble AI's](https://resemble.ai) first production-grade open source TTS model supporting **23 languages** out of the box. This version includes additional enhancements for voice agent applications, particularly real-time streaming capabilities and integration with platforms like LiveKit.

We're excited to introduce **Chatterbox Multilingual**, [Resemble AI's](https://resemble.ai) first production-grade open source TTS model supporting **23 languages** out of the box. Licensed under MIT, Chatterbox has been benchmarked against leading closed-source systems like ElevenLabs, and is consistently preferred in side-by-side evaluations.

Whether you're working on memes, videos, games, or AI agents, Chatterbox brings your content to life across languages. It's also the first open source TTS model to support **emotion exaggeration control** with robust **multilingual zero-shot voice cloning**. Try the english only version now on our [English Hugging Face Gradio app.](https://huggingface.co/spaces/ResembleAI/Chatterbox). Or try the multilingual version on our [Multilingual Hugging Face Gradio app.](https://huggingface.co/spaces/ResembleAI/Chatterbox-Multilingual-TTS).

If you like the model but need to scale or tune it for higher accuracy, check out our competitively priced TTS service (<a href="https://resemble.ai">link</a>). It delivers reliable performance with ultra-low latency of sub 200ms—ideal for production use in agents, applications, or interactive media.

## Enhancements in This Fork

This fork includes several enhancements specifically designed for voice agent applications:

### 1. Fixed Tensor Dimension Issues
- Resolved critical tensor dimension inconsistencies in the streaming implementation
- Improved token buffer processing in `S3Gen.stream_inference` method
- Enhanced tensor creation from mixed scalar/tensor inputs
- Added robust error handling for different tensor shapes

### 2. Real-time Streaming Capabilities
- Added WebSocket server implementations for real-time audio streaming
- Implemented segment-based streaming with proper start/end messages
- Created comprehensive streaming API in `ChatterboxMultilingualTTS`

### 3. Voice Agent Integration
- Enhanced support for LiveKit and other voice agent platforms
- Added Gradio interface for web-based TTS
- Improved documentation for integration use cases

## Key Details
- Multilingual, zero-shot TTS supporting 23 languages
- SoTA zeroshot English TTS
- 0.5B Llama backbone
- Unique exaggeration/intensity control
- Ultra-stable with alignment-informed inference
- Trained on 0.5M hours of cleaned data
- Watermarked outputs
- Easy voice conversion script
- [Outperforms ElevenLabs](https://podonos.com/resembleai/chatterbox)

# Supported Languages 
Arabic (ar) • Danish (da) • German (de) • Greek (el) • English (en) • Spanish (es) • Finnish (fi) • French (fr) • Hebrew (he) • Hindi (hi) • Italian (it) • Japanese (ja) • Korean (ko) • Malay (ms) • Dutch (nl) • Norwegian (no) • Polish (pl) • Portuguese (pt) • Russian (ru) • Swedish (sv) • Swahili (sw) • Turkish (tr) • Chinese (zh)

# Tips
- **General Use (TTS and Voice Agents):**
  - Ensure that the reference clip matches the specified language tag. Otherwise, language transfer outputs may inherit the accent of the reference clip’s language. To mitigate this, set `cfg_weight` to `0`.
  - The default settings (`exaggeration=0.5`, `cfg_weight=0.5`) work well for most prompts across all languages.
  - If the reference speaker has a fast speaking style, lowering `cfg_weight` to around `0.3` can improve pacing.

- **Expressive or Dramatic Speech:**
  - Try lower `cfg_weight` values (e.g. `~0.3`) and increase `exaggeration` to around `0.7` or higher.
  - Higher `exaggeration` tends to speed up speech; reducing `cfg_weight` helps compensate with slower, more deliberate pacing.

# Installation
```shell
pip install chatterbox-tts
```

Alternatively, you can install from source:
```shell
# conda create -yn chatterbox python=3.11
# conda activate chatterbox

git clone https://github.com/resemble-ai/chatterbox.git
cd chatterbox
pip install -e .
```
We developed and tested Chatterbox on Python 3.11 on Debian 11 OS; the versions of the dependencies are pinned in `pyproject.toml` to ensure consistency. You can modify the code or dependencies in this installation mode.

# Usage

## Basic Usage (Same as Original)
```python
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# English example
model = ChatterboxTTS.from_pretrained(device="cuda")

text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
wav = model.generate(text)
ta.save("test-english.wav", wav, model.sr)

# Multilingual examples
multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=device)

french_text = "Bonjour, comment ça va? Ceci est le modèle de synthèse vocale multilingue Chatterbox, il prend en charge 23 langues."
wav_french = multilingual_model.generate(french_text, language_id="fr")
ta.save("test-french.wav", wav_french, model.sr)

chinese_text = "你好，今天天气真不错，希望你有一个愉快的周末。"
wav_chinese = multilingual_model.generate(chinese_text, language_id="zh")
ta.save("test-chinese.wav", wav_chinese, model.sr)

# If you want to synthesize with a different voice, specify the audio prompt
AUDIO_PROMPT_PATH = "YOUR_FILE.wav"
wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
ta.save("test-2.wav", wav, model.sr)
```

## Streaming Usage (Enhanced Feature)
```python
import asyncio
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

async def stream_tts():
    # Initialize model
    model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
    
    # Prepare conditionals (voice cloning)
    model.prepare_conditionals("reference.wav")
    
    # Stream generate audio
    async for audio_chunk in model.stream_generate(
        text="Hello, this is a streaming TTS example.",
        language_id="en",
        chunk_size=50,
        sample_rate=24000
    ):
        # Process audio chunk (1D tensor)
        print(f"Received audio chunk with {len(audio_chunk)} samples")
        # Send to audio player, WebSocket, etc.

# Run the streaming example
asyncio.run(stream_tts())
```

## WebSocket Server (Enhanced Feature)
Start the enhanced WebSocket server for real-time streaming:

```bash
# Start the WebSocket server
python ws_server_only.py --port 8001 --host 0.0.0.0
```

Connect to the server using a WebSocket client:
- Endpoint: `ws://localhost:8001/ws/tts`
- Send JSON messages with `type: "generate"` and text content
- Receive audio chunks as binary data

## Gradio Integration (Enhanced Feature)
Run the Gradio app for a web-based TTS interface:

```bash
# For multilingual TTS
python multilingual_app.py

# For English-only TTS
python gradio_tts_app.py
```

# LiveKit Integration

To use Chatterbox with LiveKit for voice agents:

1. Start the WebSocket streaming server:
```bash
python ws_server_only.py --port 8001
```

2. In your LiveKit agent, connect to the WebSocket server and stream audio:
```python
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
```

# Documentation

For more detailed information about the enhancements in this fork:

- [Extended README](README_EXTENDED.md) - Comprehensive documentation
- [Change Log](CHANGELOG.md) - Detailed list of changes
- [Tensor Fixes Summary](TENSOR_FIXES_SUMMARY.md) - Technical details of tensor fixes
- [Test Documentation](test/README.md) - Information about test coverage
- [Usage Examples](streaming_usage_examples.py) - Practical examples for different use cases

# Acknowledgements
- [Cosyvoice](https://github.com/FunAudioLLM/CosyVoice)
- [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning)
- [HiFT-GAN](https://github.com/yl4579/HiFTNet)
- [Llama 3](https://github.com/meta-llama/llama3)
- [S3Tokenizer](https://github.com/xingchensong/S3Tokenizer)

# Built-in PerTh Watermarking for Responsible AI

Every audio file generated by Chatterbox includes [Resemble AI's Perth (Perceptual Threshold) Watermarker](https://github.com/resemble-ai/perth) - imperceptible neural watermarks that survive MP3 compression, audio editing, and common manipulations while maintaining nearly 100% detection accuracy.


## Watermark extraction

You can look for the watermark using the following script.

```python
import perth
import librosa

AUDIO_PATH = "YOUR_FILE.wav"

# Load the watermarked audio
watermarked_audio, sr = librosa.load(AUDIO_PATH, sr=None)

# Initialize watermarker (same as used for embedding)
watermarker = perth.PerthImplicitWatermarker()

# Extract watermark
watermark = watermarker.get_watermark(watermarked_audio, sample_rate=sr)
print(f"Extracted watermark: {watermark}")
# Output: 0.0 (no watermark) or 1.0 (watermarked)
```


# Official Discord

👋 Join us on [Discord](https://discord.gg/rJq9cRJBJ6) and let's build something awesome together!

# Citation
If you find this model useful, please consider citing.
```
@misc{chatterboxtts2025,
  author       = {{Resemble AI}},
  title        = {{Chatterbox-TTS}},
  year         = {2025},
  howpublished = {\url{https://github.com/resemble-ai/chatterbox}},
  note         = {GitHub repository}
}
```
# Disclaimer
Don't use this model to do bad things. Prompts are sourced from freely available data on the internet.