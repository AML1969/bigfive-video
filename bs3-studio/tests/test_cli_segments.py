"""`bs3 infer` with several long videos and --out: every video gets its own segment folder. One shared folder made the
analyzer reuse the first video's segment files (seg01_0-20s.mp4 and so on repeat from video to video) and the behaviour
descriptions of its timeline.json, so the second video was scored on the first one's segments. Nothing here loads a
model or cuts a video: the analyzer and the backend are fakes that record the work dir they were given."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import types
from pathlib import Path

from bs3 import cli, longvideo, media
from bs3.norms import TRAIT_KEYS

SCORES = {k: 0.4 + 0.02 * i for i, k in enumerate(TRAIT_KEYS)}


def _result(video):
    return {"scores": dict(SCORES), "transcript": f"речь {Path(video).stem}", "seconds": 2.0, "duration_sec": 100.0,
            "segments": 5, "timeline": [{"segment": 1, "start": 0.0, "end": 20.0, "scores": dict(SCORES)}]}


class _FakeAnalyzer:
    calls: list[tuple[str, str]] = []
    single_max = 30.0

    def __init__(self, backend, lang="ru", seg_len=20.0, asr_model=""):
        self.backend = backend

    def analyze(self, video, work_dir, progress=None, should_stop=None):
        Path(work_dir).mkdir(parents=True, exist_ok=True)      # the real analyzer creates it and writes segments here
        (Path(work_dir) / "timeline.json").write_text("[]", encoding="utf-8")
        _FakeAnalyzer.calls.append((Path(video).name, str(work_dir)))
        return _result(video)


def _run(argv):
    be = types.SimpleNamespace(cfg=types.SimpleNamespace(corpus="own checkpoints", lang="ru"), load_seconds=1.0,
                               predict_video=lambda v, asr=True, transcript=None: _result(v))
    saved = (cli._backend, media.check_upload, longvideo.LongVideoAnalyzer, longvideo.video_duration)
    cli._backend = lambda a, **kw: be
    media.check_upload = lambda *a, **k: None
    longvideo.LongVideoAnalyzer = _FakeAnalyzer
    longvideo.video_duration = lambda p: 100.0                    # longer than single_max: the segment path is taken
    _FakeAnalyzer.calls = []
    try:
        a = cli.parse_args(argv)
        with contextlib.redirect_stdout(io.StringIO()):
            a.fn(a)
    finally:
        cli._backend, media.check_upload, longvideo.LongVideoAnalyzer, longvideo.video_duration = saved
    return list(_FakeAnalyzer.calls)


def test_two_videos_with_out_get_separate_segment_folders():
    with tempfile.TemporaryDirectory(prefix="bs3_cli_seg_") as d:
        clips = []
        for name in ("R0147.mp4", "R0148.mp4"):
            p = Path(d) / name
            p.write_bytes(b"")
            clips.append(str(p))
        out = Path(d) / "batch.json"
        calls = _run(["infer", *clips, "--out", str(out)])
        assert [c[0] for c in calls] == ["R0147.mp4", "R0148.mp4"]
        dirs = [Path(c[1]) for c in calls]
        assert dirs[0] != dirs[1], dirs
        assert dirs[0] == Path(d) / "batch" / "R0147" / "segments" and dirs[1] == Path(d) / "batch" / "R0148" / "segments"
        reports = json.loads(out.read_text(encoding="utf-8"))
        assert isinstance(reports, list) and len(reports) == 2 and reports[1]["transcript"] == "речь R0148"


def test_one_video_keeps_the_old_segment_folder_layout():
    """<out without suffix>/segments, as the docs and the earlier runs have it."""
    with tempfile.TemporaryDirectory(prefix="bs3_cli_seg_") as d:
        clip = Path(d) / "R0147.mp4"
        clip.write_bytes(b"")
        out = Path(d) / "R0147.json"
        calls = _run(["infer", str(clip), "--out", str(out)])
        assert calls == [("R0147.mp4", str(Path(d) / "R0147" / "segments"))]


def test_without_out_every_video_gets_a_fresh_temporary_folder():
    with tempfile.TemporaryDirectory(prefix="bs3_cli_seg_") as d:
        clips = []
        for name in ("a.mp4", "b.mp4"):
            p = Path(d) / name
            p.write_bytes(b"")
            clips.append(str(p))
        calls = _run(["infer", *clips])
        assert len(calls) == 2 and calls[0][1] != calls[1][1]
        assert all("bs_seg_" in c[1] for c in calls)
