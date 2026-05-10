# BISINDO SLT — Streamlit App

## Cara Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

## File yang Dibutuhkan di Direktori yang Sama dengan `app.py`

| File | Keterangan |
|------|-----------|
| `hand_landmarker.task` | MediaPipe Hand Landmarker model |
| `pose_landmarker_lite.task` | MediaPipe Pose Landmarker model |

Download otomatis di notebook:
```bash
wget -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

## Upload via Sidebar

- **Model `.pth`** — file `best_model.pth` hasil training
- **Vocab `.npz`** — file vocab, buat dengan kode di bawah

## Cara Simpan Vocab ke `.npz`

Tambahkan kode ini setelah blok "Vocabulary Construction" di notebook:

```python
import numpy as np

np.savez("vocab.npz", vocab=np.array(vocab))
print("Vocab disimpan ke vocab.npz")
```

---

Arsitektur: **CNN-GRU Encoder** → **Bahdanau Attention** → **GRU Decoder**  
Keypoints: 33 Pose + 42 Hand (2 tangan) = 75 × 3 = 225 coords + velocity → 450 dim
