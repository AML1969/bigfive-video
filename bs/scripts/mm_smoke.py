"""Smoke test of the stage-2 code on a couple of real clips: faces -> CLIP, wav -> CLAP, text -> EmoRoBERTa,
then a forward/backward pass of the fusion model on the resulting vectors.
Usage: mm_smoke.py [VIDEO_DIR_WITH_MP4] [N]
"""
import sys
import time
from pathlib import Path

import torch

from bs_bigfive.mm.data import DEFAULT_ROOT, load_split_table
from bs_bigfive.mm.extractors import ClapAudioEncoder, ClipFaceEncoder, EmoRobertaTextEncoder, load_wav_mono
from bs_bigfive.mm.faces import get_face_crops
from bs_bigfive.mm.model import ModelConfig, PersonalityFusionModel
import subprocess

vdir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ROOT / "video" / "test"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
device = "cuda" if torch.cuda.is_available() else "cpu"
df = load_split_table("test")
texts = dict(zip(df["video_name"], df["text"]))
descs = dict(zip(df["video_name"], df["text_llm"]))
clips = [p for p in sorted(vdir.glob("*.mp4")) if p.stem in texts][:n]
print("clips:", [c.name for c in clips])

t = time.time(); face_enc = ClipFaceEncoder(device); clap = ClapAudioEncoder(device); txt = EmoRobertaTextEncoder(device)
print(f"encoders loaded in {time.time()-t:.1f}s; CLAP sr={clap.sample_rate}")
feats = {m: [] for m in ("face", "audio", "text", "behavior")}
for p in clips:
    t = time.time()
    crops, st = get_face_crops(str(p), n_frames=30)
    f = face_enc(crops)
    t_face = time.time() - t
    wav = Path("/tmp") / (p.stem + ".wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(p), "-vn", "-ac", "1", "-ar", str(clap.sample_rate),
                    "-acodec", "pcm_s16le", str(wav)], check=True)
    t = time.time(); a = clap.from_file(str(wav)); t_audio = time.time() - t
    t = time.time(); x_t = txt(texts[p.stem]); x_b = txt(descs[p.stem]); t_text = time.time() - t
    print(f"{p.name}: faces {st} face_vec {tuple(f.shape)} ({t_face:.2f}s) | audio_vec {tuple(a.shape)} ({t_audio:.2f}s) "
          f"| text {tuple(x_t.shape)} behavior {tuple(x_b.shape)} ({t_text:.2f}s)")
    feats["face"].append(f); feats["audio"].append(a); feats["text"].append(x_t); feats["behavior"].append(x_b)

batch = {m: torch.stack(v).to(device) for m, v in feats.items()}
model = PersonalityFusionModel(ModelConfig(modality_dims={m: v.shape[1] for m, v in batch.items()})).to(device)
out = model(batch)
loss = torch.mean(torch.abs(out - torch.rand_like(out)))
loss.backward()
print("model params:", sum(p.numel() for p in model.parameters()), "| output", out.detach().cpu().numpy().round(3), "| loss", float(loss))
print("smoke ok")
