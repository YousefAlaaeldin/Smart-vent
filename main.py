import sys, json, numpy as np, torch, sounddevice as sd, librosa
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QLabel, QListWidget, QInputDialog, QMessageBox, QHBoxLayout, QProgressBar
)
from PySide6.QtCore import Qt
from speechbrain.inference import EncoderClassifier

# ============== CONFIG ==============
TARGET_SR = 16000
REC_DUR = 4.0          # seconds per sample
TAKES = 4              # number of samples per user
THRESH = 0.60          # match threshold
PROFILES = Path("voice_profiles.json")
INPUT_DEV = 14         # your mic index (update if needed)

# ============== AUDIO SETUP ==============
def choose_samplerate(dev_idx, prefer=TARGET_SR):
    try:
        sd.check_input_settings(device=dev_idx, samplerate=prefer)
        return prefer
    except Exception:
        dev = sd.query_devices(dev_idx)
        sr = int(dev.get('default_samplerate', 48000) or 48000)
        return sr
REC_SR = choose_samplerate(INPUT_DEV)
sd.default.device = (INPUT_DEV, None)
sd.default.channels = 1

# ============== MODELS ==============
device = "cuda" if torch.cuda.is_available() else "cpu"
enc = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb",
                                     run_opts={"device": device})
vad_model, vad_utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)
vad_model.to(device).eval()
get_speech_timestamps, _, _, _, collect_chunks = vad_utils

# ============== HELPERS ==============
def record_audio(seconds=REC_DUR):
    """Record audio for a given duration and resample to 16k."""
    x = sd.rec(int(seconds * REC_SR), dtype='float32')
    sd.wait()
    y = x.squeeze()
    if REC_SR != TARGET_SR:
        y = librosa.resample(y, orig_sr=REC_SR, target_sr=TARGET_SR)
    return y

def embed(y: np.ndarray):
    """Compute ECAPA embedding with Silero VAD trimming."""
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

def load_profiles():
    return json.loads(PROFILES.read_text()) if PROFILES.exists() else {}

def save_profiles(p):
    PROFILES.write_text(json.dumps(p, indent=2))

# ============== GUI APP ==============
class VoiceApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎙 Voice Recognition System")
        self.setGeometry(400, 200, 500, 400)

        layout = QVBoxLayout(self)
        self.title = QLabel("Voice Profile Manager", self)
        self.title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(self.title)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_enroll = QPushButton("➕ Enroll User")
        self.btn_identify = QPushButton("🔍 Identify Speaker")
        self.btn_delete = QPushButton("🗑 Delete")
        btn_layout.addWidget(self.btn_enroll)
        btn_layout.addWidget(self.btn_identify)
        btn_layout.addWidget(self.btn_delete)
        layout.addLayout(btn_layout)

        self.list = QListWidget()
        layout.addWidget(QLabel("Registered Profiles:"))
        layout.addWidget(self.list)

        self.status = QLabel("Ready.")
        layout.addWidget(self.status)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        layout.addWidget(self.progress)

        # Events
        self.btn_enroll.clicked.connect(self.enroll_user)
        self.btn_identify.clicked.connect(self.identify_user)
        self.btn_delete.clicked.connect(self.delete_user)

        self.refresh_profiles()

    def refresh_profiles(self):
        """Refresh user list from JSON file."""
        self.list.clear()
        profiles = load_profiles()
        for name in profiles.keys():
            self.list.addItem(name)
        if not profiles:
            self.status.setText("⚠️ No profiles registered yet.")

    # ---------- Enroll ----------
    def enroll_user(self):
        name, ok = QInputDialog.getText(self, "New User", "Enter user name:")
        if not ok or not name.strip():
            return
        self.status.setText(f"Recording voice for {name}...")
        embs = []
        for i in range(TAKES):
            QMessageBox.information(self, "Recording",
                                    f"Speak sample {i+1}/{TAKES} for {REC_DUR:.0f} seconds.")
            y = record_audio(REC_DUR)
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

    # ---------- Identify ----------
    def identify_user(self):
        profiles = load_profiles()
        if not profiles:
            QMessageBox.warning(self, "No profiles", "Please enroll users first.")
            return
        QMessageBox.information(self, "Recording",
                                f"Speak for {REC_DUR:.0f}s to identify yourself.")
        y = record_audio(REC_DUR)
        e = embed(y)
        ranked = sorted(
            ((n, cosine(e, np.array(v))) for n, v in profiles.items()),
            key=lambda x: x[1], reverse=True
        )
        name, score = ranked[0]
        decision = "✅ MATCH" if score >= THRESH else "⚠️ BELOW THRESHOLD"
        QMessageBox.information(self, "Result",
                                f"Best match: {name}\nScore: {score:.3f}\n{decision}")
        self.status.setText(f"🎯 Identified as {name} ({score:.3f})")

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

# ============== RUN APP ==============
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = VoiceApp()
    win.show()
    sys.exit(app.exec())
