import torch
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Device detection (Apple Silicon or CPU)
device = "cpu" 
map_location = torch.device(device)

# Patch torch.load for MPS/CPU
torch_load_original = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = map_location
    return torch_load_original(*args, **kwargs)
torch.load = patched_torch_load

# Load multilingual model
model = ChatterboxMultilingualTTS.from_pretrained(device=device)

# Hindi text examples
hindi_texts = [
    "आज का दिन बहुत खास है। मैं हर पल को जियोंगा जैसे यह आखिरी हो।",  # Neutral
    "मैं बहुत खुश हूँ! ये तो मेरे लिए अद्भुत पल है!",                 # Happy
    "यह बहुत दुखद है। मुझे विश्वास नहीं हो रहा।",                   # Sad
    "मैं बहुत गुस्से में हूँ! यह बिल्कुल अस्वीकार्य है!",               # Angry
]

# Test with different exaggeration and cfg_weight
for i, text in enumerate(hindi_texts):
    # exaggeration: 0.0 (calm) → 2.0 (very dramatic)
    # cfg_weight: 0.0 (ignore reference voice) → 1.0 (strictly follow reference)
    wav = model.generate(
        text,
        language_id="hi",       # Hindi
        exaggeration=1.5,       # increase drama/emotion
        cfg_weight=0.5           # controls reference style influence
    )
    filename = f"test_hindi_{i+1}.wav"
    ta.save(filename, wav, model.sr)
    print(f"✅ Saved {filename}")
