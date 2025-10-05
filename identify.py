# identify.py
import json, numpy as np, sounddevice as sd, torch, librosa
from pathlib import Path
from speechbrain.inference import EncoderClassifier

# -------- CONFIG --------
TARGET_SR = 16000
REC_DUR = 4.0
THRESH = 0.60
PROFILES = Path("voice_profiles.json")
INPUT_DEV = 14

# -------- SAMPLE RATE --------
def choose_samplerate(dev_idx, prefer=TARGET_SR):
    try:
        sd.check_input_settings(device=dev_idx, samplerate=prefer)
        return prefer
    except Exception:
        dev = sd.query_devices(dev_idx)
        sr = int(dev.get('default_samplerate', 48000) or 48000)
        for r in (sr, 48000, 44100, 32000, 22050, 16000):
            try:
                sd.check_input_settings(device=dev_idx, samplerate=r)
                return r
            except Exception:
                pass
        raise RuntimeError("No supported sample rate found")

REC_SR = choose_samplerate(INPUT_DEV)
sd.default.device = (INPUT_DEV, None)
sd.default.samplerate = REC_SR
sd.default.channels = 1

# -------- MODELS --------
device = "cuda" if torch.cuda.is_available() else "cpu"
enc = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb",
                                     run_opts={"device": device})
vad_model, vad_utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)
vad_model.to(device).eval()
get_speech_timestamps, _, _, _, collect_chunks = vad_utils

# -------- HELPERS --------
def record(sec=REC_DUR):
    input(f"Press Enter → speak {sec:.1f}s… ")
    x = sd.rec(int(sec * REC_SR), dtype='float32'); sd.wait()
    y = x.squeeze()
    if REC_SR != TARGET_SR:
        y = librosa.resample(y, orig_sr=REC_SR, target_sr=TARGET_SR)
    return y

def embed(y: np.ndarray) -> np.ndarray:
    if y.ndim > 1:
        y = y.mean(axis=1)
    wav_t = torch.as_tensor(y, dtype=torch.float32).to(device)
    ts = get_speech_timestamps(wav_t, vad_model, sampling_rate=TARGET_SR)
    chunks = collect_chunks(ts, wav_t)
    speech = chunks if isinstance(chunks, torch.Tensor) else torch.cat(chunks)
    t = speech.unsqueeze(0)
    with torch.no_grad():
        e = enc.encode_batch(t).squeeze().cpu().numpy()
    return e

def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

# -------- PUBLIC FUNCTION --------
def identify_profile() -> str:
    """Identify the speaker and return a summary string."""
    if not PROFILES.exists():
        return "⚠️ No profiles found. Please enroll users first."

    profiles = json.loads(PROFILES.read_text())
    if not profiles:
        return "⚠️ No profiles stored in JSON."

    y = record()
    e = embed(y)

    # Compare to all stored profiles
    ranked = sorted(
        ((n, cosine(e, np.array(v))) for n, v in profiles.items()),
        key=lambda x: x[1],
        reverse=True
    )

    name, score = ranked[0]              # ✅ now defined here
    decision = "✅ MATCH" if score >= THRESH else "⚠️ BELOW THRESHOLD"

    print(f"\nBest match → {name} (score={score:.3f})")
    print("Top 3:", ranked[:3])
    print("Decision:", decision)

    # return for GUI display
    return f"Best match: {name} (score={score:.3f}) | {decision}"

# Optional: for standalone testing
if __name__ == "__main__":
    print(identify_profile())
