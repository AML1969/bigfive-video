"""Debug run of OCEAN-AI on the first N clips of the eval folder with verbose library output."""
import os
import sys
import tempfile
from pathlib import Path

from bs_bigfive.backend_oceanai import BackendConfig, OceanAIBackend

eval_dir = Path(sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/bs/eval/fi_test200"))
n = int(sys.argv[2]) if len(sys.argv) > 2 else 1

tmp = Path(tempfile.mkdtemp(prefix="bs_dbg_"))
for p in sorted(eval_dir.glob("*.mp4"))[:n]:
    os.symlink(p, tmp / p.name)          # .txt stays next to the real file; OCEAN-AI resolves symlinks
print("tmp dir:", tmp, sorted(os.listdir(tmp)))

be = OceanAIBackend(BackendConfig(lang="en")).load()
df = be.predict_dir(tmp, asr=False, verbose=True)
print(df.to_string())
