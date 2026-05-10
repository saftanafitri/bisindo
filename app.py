import os
import pickle
import re
import tempfile

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="BISINDO SLT",
    page_icon="🤟",
    layout="centered",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

:root {
    --bg: #0d0f14;
    --surface: #161921;
    --border: #252a35;
    --accent: #5cffb0;
    --accent2: #4a9eff;
    --warn: #ffb347;
    --text: #e8eaf0;
    --muted: #6b7280;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] { background: transparent !important; }

h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(92,255,176,0.1);
    border: 1px solid rgba(92,255,176,0.3);
    color: var(--accent);
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    padding: 0.3rem 0.8rem;
    border-radius: 2rem;
    margin-bottom: 1rem;
}
.hero h1 {
    font-size: 2.2rem;
    line-height: 1.2;
    margin: 0.5rem 0;
    background: linear-gradient(135deg, #e8eaf0 0%, var(--accent) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p {
    color: var(--muted);
    font-size: 0.95rem;
    max-width: 480px;
    margin: 0 auto;
}

.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.result-box {
    background: rgba(92,255,176,0.05);
    border: 1px solid rgba(92,255,176,0.25);
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-top: 0.75rem;
}
.result-sentence {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--accent);
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.02em;
}
.conf-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-top: 0.75rem;
    font-size: 0.82rem;
    color: var(--muted);
}
.conf-bar-wrap {
    flex: 1;
    height: 6px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
}
.conf-bar {
    height: 100%;
    background: linear-gradient(90deg, var(--accent2), var(--accent));
    border-radius: 3px;
    transition: width 0.8s ease;
}
.chip {
    display: inline-block;
    background: rgba(74,158,255,0.12);
    border: 1px solid rgba(74,158,255,0.25);
    color: var(--accent2);
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 0.2rem 0.55rem;
    border-radius: 4px;
    margin: 0.15rem;
}

.info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
    margin-top: 0.5rem;
}
.info-item label {
    font-size: 0.72rem;
    color: var(--muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    display: block;
    margin-bottom: 0.2rem;
}
.info-item span {
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem;
    color: var(--text);
}

.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.25rem 0;
}

/* Streamlit button */
.stButton > button {
    background: var(--accent) !important;
    color: #0d0f14 !important;
    border: none !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important;
    width: 100%;
}
.stButton > button:hover {
    background: #3de89a !important;
    box-shadow: 0 0 20px rgba(92,255,176,0.3) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

/* Progress / spinner */
.stSpinner > div { color: var(--accent) !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p {
    color: var(--text) !important;
}

/* Alerts */
.stAlert {
    border-radius: 10px !important;
    font-size: 0.88rem !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: var(--surface) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(256, hidden, batch_first=True)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1d(x))
        x = x.permute(0, 2, 1)
        outputs, hidden = self.gru(x)
        return outputs, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.W1 = nn.Linear(hidden, hidden)
        self.W2 = nn.Linear(hidden, hidden)
        self.V = nn.Linear(hidden, 1)

    def forward(self, hidden, enc_out):
        hidden = hidden.permute(1, 0, 2)
        score = self.V(torch.tanh(self.W1(enc_out) + self.W2(hidden)))
        weights = torch.softmax(score, dim=1)
        context = (weights * enc_out).sum(dim=1)
        return context


class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.gru = nn.GRU(hidden * 2, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, vocab_size)
        self.attn = BahdanauAttention(hidden)

    def forward(self, token, hidden, enc_out):
        emb = self.embedding(token).unsqueeze(1)
        context = self.attn(hidden, enc_out).unsqueeze(1)
        x = torch.cat([emb, context], dim=2)
        out, hidden = self.gru(x, hidden)
        pred = self.fc(out.squeeze(1))
        return pred, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        enc_out, hidden = self.encoder(src)
        input_token = trg[:, 0]
        outputs = []
        for t in range(1, trg.shape[1]):
            pred, hidden = self.decoder(input_token, hidden, enc_out)
            outputs.append(pred)
            input_token = trg[:, t]
        return torch.stack(outputs).permute(1, 0, 2)


# ─────────────────────────────────────────────
# PIPELINE HELPERS
# ─────────────────────────────────────────────
def temporal_resample(seq, target_len=100):
    if len(seq) == 0:
        return np.zeros((target_len, seq.shape[1] if len(seq.shape) > 1 else 1))
    idx = np.linspace(0, len(seq) - 1, target_len).astype(int)
    return seq[idx]


def compute_velocity(seq):
    vel = np.diff(seq, axis=0)
    vel = np.vstack([np.zeros(seq.shape[1]), vel])
    return np.concatenate([seq, vel], axis=1)


def normalize(seq):
    mean = seq.mean(axis=0)
    std = seq.std(axis=0) + 1e-6
    return (seq - mean) / std


@st.cache_resource
def load_mediapipe_detectors():
try:
base_dir = os.path.dirname(os.path.abspath(**file**))

```
    hand_path = os.path.join(base_dir, "hand_landmarker.task")
    pose_path = os.path.join(base_dir, "pose_landmarker_lite.task")

    if not os.path.exists(hand_path):
        st.error(f"❌ hand_landmarker.task tidak ditemukan di: {hand_path}")
        return None, None

    if not os.path.exists(pose_path):
        st.error(f"❌ pose_landmarker_lite.task tidak ditemukan di: {pose_path}")
        return None, None

    # FORCE CPU
    base_options_hand = python.BaseOptions(
        model_asset_path=hand_path,
        delegate=python.BaseOptions.Delegate.CPU
    )

    hand_detector = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=base_options_hand,
            num_hands=2
        )
    )

    base_options_pose = python.BaseOptions(
        model_asset_path=pose_path,
        delegate=python.BaseOptions.Delegate.CPU
    )

    pose_detector = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=base_options_pose
        )
    )

    return pose_detector, hand_detector

except Exception as e:
    st.error(f"❌ Error load MediaPipe: {e}")
    return None, None
```



def extract_keypoints(frame, pose_detector, hand_detector):
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    pose_result = pose_detector.detect(image)
    hand_result = hand_detector.detect(image)

    pose_kp = np.zeros(33 * 3)
    hand_kp = np.zeros(21 * 3 * 2)

    if pose_result.pose_landmarks:
        pts = [[lm.x, lm.y, lm.z] for lm in pose_result.pose_landmarks[0]]
        pose_kp = np.array(pts).flatten()

    if hand_result.hand_landmarks:
        pts = [[lm.x, lm.y, lm.z] for hand in hand_result.hand_landmarks for lm in hand]
        flat = np.array(pts).flatten()
        hand_kp[:len(flat)] = flat

    return np.concatenate([pose_kp, hand_kp])


def process_video(path, pose_detector, hand_detector):
    cap = cv2.VideoCapture(path)
    sequence = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    progress = st.progress(0, text="Mengekstrak keypoints...")
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        sequence.append(extract_keypoints(frame_rgb, pose_detector, hand_detector))
        frame_idx += 1
        if total_frames > 0:
            progress.progress(min(frame_idx / total_frames, 1.0), text=f"Frame {frame_idx}/{total_frames}")

    cap.release()
    progress.empty()

    seq = np.array(sequence)
    seq = temporal_resample(seq)
    seq = compute_velocity(seq)
    seq = normalize(seq)
    return seq, total_frames


def predict_with_confidence(sequence, model, device, word2idx, idx2word, max_len=30):
    model.eval()
    src = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src)

    trg_idx = [word2idx["<SOS>"]]
    confidences = []

    for _ in range(max_len):
        trg_tensor = torch.tensor([trg_idx[-1]], dtype=torch.long).to(device)
        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, hidden, encoder_outputs)

        probs = F.softmax(output, dim=1)
        pred_token = probs.argmax(1).item()
        conf = probs.max().item()

        trg_idx.append(pred_token)
        confidences.append(conf)

        if pred_token == word2idx["<EOS>"]:
            break

    tokens = [idx2word.get(idx, "") for idx in trg_idx]
    sentence_tokens = [t for t in tokens if t not in ["<SOS>", "<EOS>", "<PAD>", ""]]
    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    return " ".join(sentence_tokens), avg_conf, confidences

# ─────────────────────────────────────────────
# KONFIGURASI PATH
# ─────────────────────────────────────────────
_BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_BASE_DIR, "model", "best_model.pth")
VOCAB_PATH = os.path.join(_BASE_DIR, "model", "vocab.pkl")
INPUT_DIM   = 450
HIDDEN_DIM  = 256
MAX_DECODE  = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# LOAD OTOMATIS SAAT STARTUP
# ─────────────────────────────────────────────
@st.cache_resource
def load_vocab():
    with open(VOCAB_PATH, "rb") as f:
        data = pickle.load(f)
    word2idx = data["word2idx"]
    idx2word = data["idx2word"]
    return word2idx, idx2word

@st.cache_resource
def load_model(vocab_size):
    encoder = Encoder(INPUT_DIM, HIDDEN_DIM)
    decoder = Decoder(vocab_size, HIDDEN_DIM)
    model   = Seq2Seq(encoder, decoder).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model

model_ready = False
vocab_ready = False
word2idx, idx2word, slt_model = {}, {}, None

if not os.path.exists(VOCAB_PATH):
    st.error("❌ File vocab.pkl tidak ditemukan di folder model/")
elif not os.path.exists(MODEL_PATH):
    st.error("❌ File best_model.pth tidak ditemukan di folder model/")
else:
    try:
        word2idx, idx2word = load_vocab()
        vocab_ready = True
    except Exception as e:
        st.error(f"Gagal memuat vocab: {e}")

    if vocab_ready:
        try:
            slt_model = load_model(len(word2idx))
            model_ready = True
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")

# ─────────────────────────────────────────────
# SIDEBAR – Info saja
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤟 BISINDO SLT")
    st.markdown("---")
    if model_ready:
        st.success("Model aktif")
        st.caption(f"Vocab: {len(word2idx)} token")
        st.caption(f"Device: {device}")
    else:
        st.warning("Model belum siap")
    st.markdown("---")
    st.caption("CNN-GRU + Bahdanau Attention")

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🤟 BISINDO · Sign Language Translation</div>
    <h1>Video → Teks</h1>
    <p>Pipeline CNN-GRU dengan Bahdanau Attention untuk menerjemahkan gestur BISINDO menjadi kalimat Bahasa Indonesia.</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ARSITEKTUR INFO (collapsed)
# ─────────────────────────────────────────────
with st.expander("📐 Arsitektur Model", expanded=False):
    st.markdown("""
<div class="info-grid">
  <div class="info-item"><label>Encoder</label><span>Conv1D → GRU</span></div>
  <div class="info-item"><label>Attention</label><span>Bahdanau</span></div>
  <div class="info-item"><label>Decoder</label><span>GRU + Embedding</span></div>
  <div class="info-item"><label>Keypoints</label><span>33 Pose + 42 Hand</span></div>
  <div class="info-item"><label>Input Dim</label><span>450 (+ velocity)</span></div>
  <div class="info-item"><label>Seq Len</label><span>100 (resampled)</span></div>
</div>
    """, unsafe_allow_html=True)

    st.markdown("""
<hr class="divider">
<b>Token Khusus:</b><br>
<span class="chip">&lt;PAD&gt;</span>
<span class="chip">&lt;SOS&gt;</span>
<span class="chip">&lt;EOS&gt;</span>
    """, unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# VIDEO UPLOAD & PREDICT
# ─────────────────────────────────────────────
st.markdown("### 🎬 Upload Video Isyarat")
video_file = st.file_uploader(
    "Pilih file video (.mp4, .avi, .mov)",
    type=["mp4", "avi", "mov"],
    label_visibility="collapsed"
)

if video_file:
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.video(video_file)
    with col2:
        st.markdown(f"""
<div class="card">
  <div class="info-item"><label>Nama File</label><span>{video_file.name}</span></div>
  <div style="margin-top:.75rem" class="info-item"><label>Ukuran</label><span>{video_file.size/1024:.1f} KB</span></div>
  <div style="margin-top:.75rem" class="info-item"><label>Tipe</label><span>{video_file.type}</span></div>
</div>
        """, unsafe_allow_html=True)

    run_btn = st.button("🔍 Terjemahkan Video", disabled=not model_ready)

    if run_btn and model_ready:
        pose_det, hand_det = load_mediapipe_detectors()

        if pose_det is None or hand_det is None:
            st.error("❌ MediaPipe landmarker tidak ditemukan. Pastikan file `hand_landmarker.task` dan `pose_landmarker_lite.task` ada di direktori yang sama dengan `app.py`.")
        else:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(video_file.read())
                tmp_path = tmp.name

            try:
                with st.spinner("Memproses video..."):
                    seq, n_frames = process_video(tmp_path, pose_det, hand_det)

                with st.spinner("Mendekode kalimat..."):
                    sentence, avg_conf, per_token_conf = predict_with_confidence(
                        seq, slt_model, device, word2idx, idx2word, max_len=MAX_DECODE
                    )

                conf_pct = avg_conf * 100
                color = "#5cffb0" if conf_pct >= 70 else "#ffb347" if conf_pct >= 40 else "#ff6b6b"

                st.markdown(f"""
<div class="result-box">
  <div style="font-size:.75rem;color:#6b7280;letter-spacing:.08em;text-transform:uppercase;margin-bottom:.5rem">Hasil Terjemahan</div>
  <div class="result-sentence">"{sentence if sentence else '(tidak terdeteksi)'}"</div>
  <div class="conf-row">
    <span>Confidence</span>
    <div class="conf-bar-wrap">
      <div class="conf-bar" style="width:{conf_pct:.1f}%;background:linear-gradient(90deg,#4a9eff,{color})"></div>
    </div>
    <span style="font-family:'Space Mono',monospace;color:{color};font-size:.8rem">{conf_pct:.1f}%</span>
  </div>
</div>
                """, unsafe_allow_html=True)

                with st.expander("📊 Detail Per Token", expanded=False):
                    tokens_out = sentence.split() if sentence else []
                    if tokens_out and per_token_conf:
                        cols = st.columns(min(len(tokens_out), 6))
                        for i, (tok, c) in enumerate(zip(tokens_out, per_token_conf[:len(tokens_out)])):
                            with cols[i % len(cols)]:
                                st.metric(label=tok, value=f"{c*100:.0f}%")
                    else:
                        st.info("Tidak ada token yang dihasilkan.")

                    st.markdown(f"""
<div class="info-grid" style="margin-top:.75rem">
  <div class="info-item"><label>Total Frame</label><span>{n_frames}</span></div>
  <div class="info-item"><label>Seq Len (resampled)</label><span>100</span></div>
  <div class="info-item"><label>Jumlah Token</label><span>{len(tokens_out)}</span></div>
  <div class="info-item"><label>Device</label><span>{device}</span></div>
</div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Error saat prediksi: {e}")
            finally:
                os.unlink(tmp_path)

else:
    st.markdown("""
<div class="card" style="text-align:center;padding:2rem;border-style:dashed">
  <div style="font-size:2.5rem;margin-bottom:.5rem">🎥</div>
  <div style="color:#6b7280;font-size:.9rem">Belum ada video yang dipilih.<br>Upload video gestur BISINDO untuk memulai.</div>
</div>
    """, unsafe_allow_html=True)
