
import json, numpy as np, sounddevice as sd, soundfile as sf, torch, librosa
from pathlib import Path
from speechbrain.inference import EncoderClassifier


TARGET_SR = 16000           # model expects 16 kHz
INPUT_DEV = 14              # your mic index
REC_DUR_DEFAULT = 4.0       # seconds per take
TAKES_DEFAULT = 4           # takes to average
PROFILES = Path("voice_profiles.json")

#  Sample rate negotiation 
def choose_samplerate(dev_idx, prefer=TARGET_SR):
    try:
        sd.check_input_settings(device=dev_idx, samplerate=prefer)
        return prefer
    except Exception:
        dev = sd.query_devices(dev_idx)
        sr = int(dev.get('default_samplerate', 48000) or 48000)
        try:
            sd.check_input_settings(device=dev_idx, samplerate=sr)
            return sr
        except Exception:
            for r in (48000, 44100, 32000, 22050, 16000):
                try:
                    sd.check_input_settings(device=dev_idx, samplerate=r)
                    return r
                except Exception:
                    pass
            raise RuntimeError("No supported input samplerate found.")

REC_SR = choose_samplerate(INPUT_DEV)
sd.default.device = (INPUT_DEV, None)
sd.default.samplerate = REC_SR
sd.default.channels = 1


device = "cuda" if torch.cuda.is_available() else "cpu"
enc = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb",
                                     run_opts={"device": device})
vad_model, vad_utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)
vad_model.to(device).eval()
get_speech_timestamps, _, _, _, collect_chunks = vad_utils


def record(sec):
    input(f"Press Enter and speak {sec:.1f}s… ")
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


def enroll_profile(name: str | None = None, takes: int = TAKES_DEFAULT, dur: float = REC_DUR_DEFAULT):
    """Record `takes` clips, average embeddings, and save under `name`."""
    print(f"🎤 Using device {INPUT_DEV} at {REC_SR} Hz (→ {TARGET_SR} Hz)")
    if not name:
        name = input("Profile name: ").strip()
        if not name:
            raise ValueError("Name cannot be empty.")

    embs = []
    for i in range(takes):
        y = record(dur)
        sf.write(f"{name}_take{i+1}.wav", y, TARGET_SR)
        e = embed(y)
        embs.append(e)
        print(f"✓ Take {i+1}/{takes}")

    
    mean_emb = np.mean(np.stack(embs, axis=0), axis=0)

    profiles = json.loads(PROFILES.read_text()) if PROFILES.exists() else {}
    profiles[name] = mean_emb.tolist()
    PROFILES.write_text(json.dumps(profiles, indent=2))
    print(f"✅ Saved profile '{name}' to {PROFILES}")


if __name__ == "__main__":
    enroll_profile()

