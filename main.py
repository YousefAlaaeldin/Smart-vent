import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import sys, json
from pathlib import Path

import numpy as np
import librosa
import sounddevice as sd
import torch

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QLabel, QListWidget, QInputDialog, QMessageBox, QHBoxLayout,
    QProgressBar, QTextEdit, QLineEdit
)
from PySide6.QtCore import QThread, Signal, Slot
import json as _json

from speechbrain.inference import EncoderClassifier  # SpeechBrain v1.0+

# ====================== CONFIG ======================
TARGET_SR = 16000
REC_DUR   = 8.0          # seconds per sample
TAKES     = 4            # number of samples per user (enroll)
THRESH    = 0.60         # match threshold (cosine)
PROFILES  = Path("voice_profiles.json")
INPUT_DEV = 14           # your mic index (change if needed)

# Your English ASR model (GigaSpeech)
VOSK_MODEL_DIR = Path(r"C:\Users\Ahmad\Downloads\vosk-model-en-us-0.42-gigaspeech")

# ====================== VOSK ASR HELPERS ======================
asr_model = None
VOSK_AVAILABLE = False

def init_vosk_model(model_dir: Path):
    """
    Initialize Vosk ASR from model_dir.
    Returns (ok: bool, msg: str). On success sets global asr_model.
    """
    global asr_model, VOSK_AVAILABLE
    try:
        import vosk
        VOSK_AVAILABLE = True
        vosk.SetLogLevel(-1)  # quiet logs
    except Exception as e:
        return False, f"Vosk not installed: {e}"

    if not model_dir or not model_dir.exists():
        return False, f"ASR model path not found: {model_dir}"

    # Some models are nested as <root>/model/...
    inner = model_dir / "model"
    if inner.exists() and inner.is_dir():
        model_dir = inner

    try:
        from vosk import Model
        asr_model = Model(str(model_dir))
        return True, f"ASR ready: {model_dir}"
    except Exception as e:
        return False, f"Failed to load Vosk model from {model_dir}: {e}"

def transcribe_vosk(y: np.ndarray) -> str:
    """Offline ASR via Vosk. y must be 16 kHz mono float32 in [-1,1]."""
    global asr_model, VOSK_AVAILABLE
    if not (VOSK_AVAILABLE and asr_model is not None):
        return ""
    from vosk import KaldiRecognizer
    pcm = (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    rec = KaldiRecognizer(asr_model, TARGET_SR)
    step = 8000  # ~0.25s per chunk
    for i in range(0, len(pcm), step):
        rec.AcceptWaveform(pcm[i:i+step])
    out = json.loads(rec.FinalResult() or "{}")
    return (out.get("text") or "").strip()

# Initialize ASR once (safe log)
ok, msg = init_vosk_model(VOSK_MODEL_DIR)
print(msg)

# ====================== AUDIO DEVICE PICKER ======================
def pick_input_config(dev_idx, preferred_sr=TARGET_SR, preferred_ch=1):
    """Return (channels, samplerate) that the device accepts."""
    dev = sd.query_devices(dev_idx)
    max_in = int(dev.get('max_input_channels', 0))
    if max_in < 1:
        raise RuntimeError(f"Device {dev_idx} has no input channels.")

    try_ch = preferred_ch if preferred_ch <= max_in else max_in
    try:
        sd.check_input_settings(device=dev_idx, channels=try_ch, samplerate=preferred_sr)
        sr = preferred_sr
        ch = try_ch
    except Exception:
        sr = int(dev.get('default_samplerate', 48000) or 48000)
        sd.check_input_settings(device=dev_idx, channels=try_ch, samplerate=sr)
        ch = try_ch
    return ch, sr

try:
    IN_CH, REC_SR = pick_input_config(INPUT_DEV, preferred_sr=TARGET_SR, preferred_ch=1)
except Exception:
    dev = sd.query_devices(INPUT_DEV)
    fallback_sr = int(dev.get('default_samplerate', 48000) or 48000)
    sd.check_input_settings(device=INPUT_DEV, channels=2, samplerate=fallback_sr)
    IN_CH, REC_SR = 2, fallback_sr

sd.default.device = (INPUT_DEV, None)  # we pass channels & samplerate explicitly later

# ====================== MODELS ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
enc = EncoderClassifier.from_hparams(
    "speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": device}
)
vad_model, vad_utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)
vad_model.to(device).eval()
get_speech_timestamps, _, _, _, collect_chunks = vad_utils

# ====================== CORE HELPERS ======================
def record_audio(seconds=REC_DUR):
    """Record audio and resample to 16k."""
    x = sd.rec(
        int(seconds * REC_SR),
        samplerate=REC_SR,
        channels=IN_CH,
        dtype='float32',
        device=INPUT_DEV
    )
    sd.wait()
    y = x.squeeze()
    if y.ndim > 1:
        y = y.mean(axis=1)
    if REC_SR != TARGET_SR:
        y = librosa.resample(y, orig_sr=REC_SR, target_sr=TARGET_SR)
    return y

def embed(y: np.ndarray):
    """Compute ECAPA embedding with Silero VAD trimming (expects 16 kHz)."""
    if y.ndim > 1:
        y = y.mean(axis=1)
    wav_t = torch.as_tensor(y, dtype=torch.float32).to(device)
    ts = get_speech_timestamps(wav_t, vad_model, sampling_rate=TARGET_SR)
    chunks = collect_chunks(ts, wav_t)
    speech = chunks if isinstance(chunks, torch.Tensor) else (torch.cat(chunks) if chunks else wav_t)
    t = speech.unsqueeze(0)
    with torch.no_grad():
        e = enc.encode_batch(t).squeeze().cpu().numpy()
    return e

def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

def load_profiles():
    return json.loads(PROFILES.read_text()) if PROFILES.exists() else {}

def save_profiles(p):
    PROFILES.write_text(json.dumps(p, indent=2))

# ====================== STREAMING WORKER (wake/sleep + live) ======================
class ListenerWorker(QThread):
    # Signals
    partial_text = Signal(str)   # combined (final_accum + partial_cur) when active
    final_text = Signal(str)     # combined (final_accum) when active
    status = Signal(str)
    identified = Signal(str, float, str)   # name, score, transcript
    error = Signal(str)
    stopped = Signal()

    def __init__(self, wake_word: str, sleep_word: str, model_ready: bool,
                 input_dev: int, in_ch: int, sample_rate: int,
                 asr_model, THRESH, load_profiles_fn, embed_fn, cosine_fn, parent=None):
        super().__init__(parent)
        # Normalize phrases
        self.wake_phrase = (wake_word or "").strip().lower()
        self.sleep_phrase = (sleep_word or "").strip().lower()

        self.model_ready = model_ready
        self.input_dev = input_dev
        self.in_ch = in_ch
        self.stream_sr = sample_rate   # actual device rate (REC_SR)
        self.asr_model = asr_model
        self.THRESH = THRESH
        self.load_profiles = load_profiles_fn
        self.embed = embed_fn
        self.cosine = cosine_fn

        # streaming state
        self._running = True
        self.active = False
        self.cap_buf = []              # float32 chunks at stream_sr
        self.final_accum = ""          # finalized transcript during active window
        self.partial_cur = ""          # latest partial (not appended to final_accum)

    def stop(self):
        self._running = False

    def _emit_text(self):
        """Emit the combined text (final + partial) to the UI when active."""
        if not self.active:
            return
        combo = (self.final_accum + (" " + self.partial_cur if self.partial_cur else "")).strip()
        if combo:
            self.partial_text.emit(combo)

    def _finalize_current(self):
        """Identify speaker for the captured utterance and emit result, then clear buffers."""
        if not self.cap_buf:
            # Clear UI text anyway at end of turn
            self.final_text.emit(self.final_accum.strip())
            self.final_accum = ""
            self.partial_cur = ""
            return
        try:
            y = np.concatenate(self.cap_buf, axis=0)   # y is at stream_sr
        except ValueError:
            self.final_text.emit(self.final_accum.strip())
            self.final_accum = ""
            self.partial_cur = ""
            return

        # Resample to 16k for ECAPA
        if self.stream_sr != 16000:
            y = librosa.resample(y, orig_sr=self.stream_sr, target_sr=16000)

        text = self.final_accum.strip()
        try:
            e = self.embed(y)  # expects 16k
            profiles = self.load_profiles()
            if not profiles:
                self.status.emit("No profiles enrolled.")
            else:
                ranked = sorted(
                    ((n, self.cosine(e, np.array(v))) for n, v in profiles.items()),
                    key=lambda x: x[1], reverse=True
                )
                best_name, best_score = ranked[0]
                if best_score < self.THRESH:
                    self.identified.emit("User not identified", best_score, text)
                else:
                    self.identified.emit(best_name, best_score, text)
        except Exception as ex:
            self.error.emit(f"Identify error: {ex}")
        finally:
            # Clear all capture state
            self.cap_buf = []
            self.final_accum = ""
            self.partial_cur = ""

    def run(self):
        # Guards
        if not self.model_ready or self.asr_model is None:
            self.error.emit("ASR not initialized")
            self.stopped.emit()
            return

        try:
            from vosk import KaldiRecognizer
        except Exception as e:
            self.error.emit(f"Vosk import error: {e}")
            self.stopped.emit()
            return

        # Recognizer 1: full dictation (live transcription) — ONLY used while active
        rec_full = KaldiRecognizer(self.asr_model, self.stream_sr)

        # Recognizer 2: small grammar just for wake/sleep words (reliable triggers)
        kw_list = []
        if self.wake_phrase:
            kw_list.append(self.wake_phrase)
        if self.sleep_phrase and self.sleep_phrase != self.wake_phrase:
            kw_list.append(self.sleep_phrase)
        if not kw_list:
            kw_list = ["[unk]"]
        grammar_str = json.dumps(kw_list)
        rec_kw = KaldiRecognizer(self.asr_model, self.stream_sr, grammar_str)

        # 100 ms blocks for snappy partials
        dtype = 'int16'
        blocksize = max(1, int(0.10 * self.stream_sr))

        def reset_full_and_clear_text():
            rec_full.Reset()
            self.final_accum = ""
            self.partial_cur = ""
            self.cap_buf = []
            self._emit_text()

        def callback(indata, frames, time_info, status_flags):
            # Convert to bytes and downmix if needed
            if self.in_ch > 1:
                arr = np.frombuffer(indata, dtype=np.int16).reshape(-1, self.in_ch)
                mono = arr.mean(axis=1).astype(np.int16)
                raw_bytes = mono.tobytes()
            else:
                raw_bytes = bytes(indata)  # ensure bytes, not cffi buffer

            try:
                # --- Keyword recognizer (wake/sleep) ---
                accepted_kw = rec_kw.AcceptWaveform(raw_bytes)
                kw_text = ""
                if accepted_kw:
                    r = _json.loads(rec_kw.Result() or "{}")
                    kw_text = (r.get("text") or "").strip().lower()

                    if not self.active:
                        if self.wake_phrase and self.wake_phrase in kw_text:
                            # Enter active mode: clear dictation, buffers, and start fresh
                            self.active = True
                            reset_full_and_clear_text()
                            self.cap_buf = []
                            self.status.emit(f"Wake word detected: '{self.wake_phrase}' — listening…")
                            # Reset keyword recognizer so it doesn't instantly retrigger
                            rec_kw.Reset()
                    else:
                        if self.sleep_phrase and self.sleep_phrase in kw_text:
                            # Leave active mode: finalize captured audio
                            self.status.emit(f"Sleep word detected: '{self.sleep_phrase}' — processing…")
                            self.active = False
                            self._finalize_current()
                            # Reset recognizers and avoid more text until next wake
                            rec_full.Reset()
                            rec_kw.Reset()
                            return  # stop processing this block for dictation
                # If kw partials are desired (often noisy), we could also check PartialResult here,
                # but for reliability we stick to finalized kw matches only.

                # --- Full dictation recognizer (only while active) ---
                if self.active:
                    accepted_full = rec_full.AcceptWaveform(raw_bytes)
                    if accepted_full:
                        res = _json.loads(rec_full.Result() or "{}")
                        t_final = (res.get("text") or "").strip()
                        if t_final:
                            # Append to final_accum ONCE
                            self.final_accum = (self.final_accum + " " + t_final).strip() if self.final_accum else t_final
                            self.partial_cur = ""
                            # Emit the consolidated text
                            self.final_text.emit(self.final_accum)
                    else:
                        pres = _json.loads(rec_full.PartialResult() or "{}")
                        ptxt = (pres.get("partial") or "").strip()
                        # Only show partial overlay; do not append to final_accum
                        if ptxt and ptxt != self.partial_cur:
                            self.partial_cur = ptxt
                            self._emit_text()

                    # Collect raw audio for ID while active
                    s = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    self.cap_buf.append(s)

            except Exception as e:
                self.error.emit(f"Recognizer error: {e}")

            if not self._running:
                raise sd.CallbackAbort

        try:
            with sd.RawInputStream(samplerate=self.stream_sr,
                                   channels=self.in_ch,
                                   dtype=dtype,
                                   blocksize=blocksize,
                                   device=self.input_dev,
                                   callback=callback):
                self.status.emit("Streaming… (say wake word to start)")
                while self._running:
                    sd.sleep(100)
        except Exception as e:
            self.error.emit(f"Audio stream error: {e}")
        finally:
            if self.active:
                # If stopped while active, finalize what we have
                self._finalize_current()
            self.stopped.emit()

# ====================== GUI ======================
class VoiceApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎙 Voice Recognition System")
        self.setGeometry(400, 200, 700, 560)

        layout = QVBoxLayout(self)
        self.title = QLabel("Voice Profile Manager", self)
        self.title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(self.title)

        # Buttons row
        btn_layout = QHBoxLayout()
        self.btn_enroll = QPushButton("➕ Enroll User")
        self.btn_identify = QPushButton("🔍 Identify Speaker")
        self.btn_listen_identify = QPushButton("🎤 Listen & Identify (ASR + ID)")
        self.btn_delete = QPushButton("🗑 Delete")
        btn_layout.addWidget(self.btn_enroll)
        btn_layout.addWidget(self.btn_identify)
        btn_layout.addWidget(self.btn_listen_identify)
        btn_layout.addWidget(self.btn_delete)
        layout.addLayout(btn_layout)

        # Wake/Sleep + Start/Stop streaming row (NEW)
        ws = QHBoxLayout()
        ws.addWidget(QLabel("Wake word:"))
        self.le_wake = QLineEdit()
        self.le_wake.setPlaceholderText("e.g., hey home")
        ws.addWidget(self.le_wake)

        ws.addWidget(QLabel("Sleep word:"))
        self.le_sleep = QLineEdit()
        self.le_sleep.setPlaceholderText("e.g., stop listening")
        ws.addWidget(self.le_sleep)

        self.btn_listen_stream = QPushButton("▶ Start Listening")
        ws.addWidget(self.btn_listen_stream)
        layout.addLayout(ws)

        self.listener = None
        self.btn_listen_stream.clicked.connect(self.toggle_stream)

        self.list = QListWidget()
        layout.addWidget(QLabel("Registered Profiles:"))
        layout.addWidget(self.list)

        layout.addWidget(QLabel("Transcript:"))
        self.asr_box = QTextEdit()
        self.asr_box.setReadOnly(True)
        self.asr_box.setPlaceholderText("(speech-to-text will appear here)")
        layout.addWidget(self.asr_box)

        self.status = QLabel("Ready.")
        layout.addWidget(self.status)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        layout.addWidget(self.progress)

        # Events
        self.btn_enroll.clicked.connect(self.enroll_user)
        self.btn_identify.clicked.connect(self.identify_user)
        self.btn_listen_identify.clicked.connect(self.listen_and_identify)
        self.btn_delete.clicked.connect(self.delete_user)

        self.refresh_profiles()

        # ASR status
        if ok:
            self.status.setText("ASR model initialized.")
        else:
            self.status.setText(f"ASR not initialized: {msg}")

    def refresh_profiles(self):
        self.list.clear()
        profiles = load_profiles()
        for name in profiles.keys():
            self.list.addItem(name)
        if not profiles:
            self.status.setText("⚠️ No profiles registered yet.")

    # ====== Continuous streaming controls ======
    @Slot()
    def toggle_stream(self):
        if self.listener and self.listener.isRunning():
            self.listener.stop()
            self.btn_listen_stream.setText("▶ Start Listening")
            return

        wake = (self.le_wake.text() or "").strip()
        sleep = (self.le_sleep.text() or "").strip()
        if not wake:
            QMessageBox.warning(self, "Wake word", "Please enter a wake word.")
            return

        self.asr_box.clear()
        self.status.setText("Starting continuous listener…")
        self.listener = ListenerWorker(
            wake_word=wake,
            sleep_word=sleep,
            model_ready=(VOSK_AVAILABLE and asr_model is not None),
            input_dev=INPUT_DEV,
            in_ch=IN_CH,
            sample_rate=REC_SR,          # use device-supported rate
            asr_model=asr_model,
            THRESH=THRESH,
            load_profiles_fn=load_profiles,
            embed_fn=embed,
            cosine_fn=cosine,
            parent=self
        )
        self.listener.partial_text.connect(self.on_partial_text)
        self.listener.final_text.connect(self.on_final_text)
        self.listener.status.connect(self.on_listener_status)
        self.listener.error.connect(self.on_listener_error)
        self.listener.identified.connect(self.on_listener_identified)
        self.listener.stopped.connect(self.on_listener_stopped)
        self.btn_listen_stream.setText("■ Stop Listening")
        self.listener.start()

    @Slot(str)
    def on_partial_text(self, combined: str):
        # Replace entire box with combined (final + partial) to avoid duplicates
        self.asr_box.setPlainText(combined)

    @Slot(str)
    def on_final_text(self, combined_final: str):
        # Replace entire box with final-only (no partial)
        self.asr_box.setPlainText(combined_final)

    @Slot(str)
    def on_listener_status(self, s: str):
        self.status.setText(s)

    @Slot(str)
    def on_listener_error(self, e: str):
        QMessageBox.critical(self, "Listener Error", e)

    @Slot()
    def on_listener_stopped(self):
        self.status.setText("Listener stopped.")
        self.btn_listen_stream.setText("▶ Start Listening")

    @Slot(str, float, str)
    def on_listener_identified(self, name: str, score: float, text: str):
        if name == "User not identified":
            self.user_not_identified(score)
        else:
            QMessageBox.information(
                self, "Result",
                f"Speaker: {name} ({score:.3f})\n\nText:\n{text or '(empty)'}"
            )
            self.status.setText(f"🗣 {name}: {text or '(empty)'}")

    # ====== Shared GUI helpers ======
    def user_not_identified(self, score: float):
        QMessageBox.warning(
            self,
            "User not identified",
            f"No profile passed the threshold.\n\n"
            f"Best similarity: {score:.3f}\n"
            f"Threshold: {THRESH:.2f}"
        )
        self.status.setText(f"❌ User not identified (best {score:.3f} < {THRESH:.2f})")

    # ---------- Enroll ----------
    def enroll_user(self):
        name, ok_ = QInputDialog.getText(self, "New User", "Enter user name:")
        if not ok_ or not name.strip():
            return
        name = name.strip()
        self.status.setText(f"Recording voice for {name}...")
        embs = []
        for i in range(TAKES):
            QMessageBox.information(self, "Recording",
                                    f"Speak sample {i+1}/{TAKES} for {REC_DUR:.0f} seconds.")
            try:
                y = record_audio(REC_DUR)
            except Exception as e:
                QMessageBox.critical(self, "Audio Error", f"Failed to open microphone:\n{e}")
                return
            e = embed(y)
            embs.append(e)
            self.progress.setValue(int((i + 1) / TAKES * 100))
            QApplication.processEvents()
        mean_emb = np.mean(np.stack(embs, axis=0), axis=0)
        profiles = load_profiles()
        profiles[name] = mean_emb.tolist()
        save_profiles(profiles)
        self.refresh_profiles()
        self.status.setText(f"✅ {name} enrolled successfully.")
        self.progress.setValue(0)

    # ---------- Identify only (fixed duration) ----------
    def identify_user(self):
        profiles = load_profiles()
        if not profiles:
            QMessageBox.warning(self, "No profiles", "Please enroll users first.")
            return
        QMessageBox.information(self, "Recording",
                                f"Speak for {REC_DUR:.0f}s to identify yourself.")
        try:
            y = record_audio(REC_DUR)
        except Exception as e:
            QMessageBox.critical(self, "Audio Error", f"Failed to open microphone:\n{e}")
            return
        e = embed(y)
        ranked = sorted(
            ((n, cosine(e, np.array(v))) for n, v in profiles.items()),
            key=lambda x: x[1], reverse=True
        )
        best_name, best_score = ranked[0]
        if best_score < THRESH:
            self.user_not_identified(best_score)
            return
        QMessageBox.information(self, "Result",
                                f"Best match: {best_name}\nScore: {best_score:.3f}\n✅ MATCH")
        self.status.setText(f"🎯 Identified as {best_name} ({best_score:.3f})")

    # ---------- Listen -> ASR -> Identify (fixed duration) ----------
    def listen_and_identify(self):
        QMessageBox.information(self, "Recording",
                                f"Speak for {REC_DUR:.0f}s. I will transcribe and identify.")
        try:
            y = record_audio(REC_DUR)
        except Exception as e:
            QMessageBox.critical(self, "Audio Error", f"Failed to open microphone:\n{e}")
            return

        # 1) Transcribe (y is already 16k here)
        text = transcribe_vosk(y) if (VOSK_AVAILABLE and asr_model is not None) else "(ASR not available)"
        self.asr_box.setPlainText(text if text else "(no speech detected)")

        # 2) Identify
        profiles = load_profiles()
        if not profiles:
            QMessageBox.warning(self, "No profiles", "Please enroll users first.")
            return
        e = embed(y)
        ranked = sorted(
            ((n, cosine(e, np.array(v))) for n, v in profiles.items()),
            key=lambda x: x[1], reverse=True
        )
        best_name, best_score = ranked[0]
        if best_score < THRESH:
            self.user_not_identified(best_score)
            return
        QMessageBox.information(
            self, "Result",
            f"Speaker: {best_name} ({best_score:.3f})\n\nText:\n{text or '(empty)'}"
        )
        self.status.setText(f"🗣 {best_name}: {text or '(empty)'}")

    # ---------- Delete ----------
    def delete_user(self):
        selected = self.list.currentItem()
        if not selected:
            QMessageBox.warning(self, "Delete", "Select a profile to delete.")
            return
        name = selected.text()
        confirm = QMessageBox.question(self, "Confirm Delete",
                                       f"Delete {name}?",
                                       QMessageBox.Yes | QMessageBox.No)
        if confirm == QMessageBox.Yes:
            profiles = load_profiles()
            profiles.pop(name, None)
            save_profiles(profiles)
            self.refresh_profiles()
            self.status.setText(f"🗑 Deleted {name}.")

# ====================== RUN ======================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = VoiceApp()
    win.show()
    sys.exit(app.exec())
