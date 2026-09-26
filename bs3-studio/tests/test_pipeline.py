"""BS Profiler 3.1 pipeline without a GPU (change request 3.1, sections 1, 2, 6): the Studio builds the backend of the
chosen member only, run_analysis(member=...) runs one model, fixes the speech language to Russian, records the model
in result.json (model.selected / selected_title / primary, one member in variant_scores, an mbti section of schema 3
with one source), and asks for explanations only from AMLAI 1.0. The heavy backend modules are replaced by fakes in
sys.modules for the Studio test; run_analysis gets a fake Studio and a canned analyzer result."""
from __future__ import annotations

import inspect
import json
import logging
import sys
import tempfile
import threading
import types
from pathlib import Path

from samples import rep

import bs3
from bs3 import pipeline, pool
from bs3.norms import TRAIT_KEYS

MEMBERS = ("oceanai", "mm")


# ------------------------------------------------------------------------------------------------- fakes ---

class _FakeEnsemble:
    """Stands in for backend_ensemble.EnsembleBackend: records the config, builds one inner backend per member."""
    built: list = []

    def __init__(self, cfg):
        self.cfg = cfg
        self.backends = {m: types.SimpleNamespace(cfg=types.SimpleNamespace(corpus=f"corpus-of-{m}")) for m in cfg.members}
        self.loaded = 0
        _FakeEnsemble.built.append(self)

    def load(self):
        self.loaded += 1
        return self


def _fake_modules():
    """bs3.backend_mm and bs3.backend_oceanai without torch / OCEAN-AI: config dataclasses that record their kwargs;
    the real EnsembleConfig (its primary rule is what is tested) with the fake EnsembleBackend."""
    from bs3.backend_ensemble import EnsembleConfig

    def cfg_class(name):
        def __init__(self, **kw):
            self.__dict__.update(kw)
        return type(name, (), {"__init__": __init__})

    mm = types.ModuleType("bs3.backend_mm")
    mm.MMConfig = cfg_class("MMConfig")
    oa = types.ModuleType("bs3.backend_oceanai")
    oa.BackendConfig = cfg_class("BackendConfig")
    ens = types.ModuleType("bs3.backend_ensemble")
    ens.EnsembleConfig, ens.EnsembleBackend = EnsembleConfig, _FakeEnsemble
    return {"bs3.backend_mm": mm, "bs3.backend_oceanai": oa, "bs3.backend_ensemble": ens}


class _Patched:
    def __init__(self, mods: dict):
        self.mods, self.saved = mods, {}

    def __enter__(self):
        for name, mod in self.mods.items():
            self.saved[name] = sys.modules.get(name)
            sys.modules[name] = mod
        return self

    def __exit__(self, *exc):
        for name, old in self.saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


class _FakeAnalyzer:
    """LongVideoAnalyzer stand-in: a short clip scored as one segment by the member of its backend."""

    def __init__(self, member: str):
        self.member = member
        self.calls = []

    def analyze(self, video, work_dir, progress=None, should_stop=None):
        self.calls.append(str(video))
        if progress:
            progress(0.5, "отрезок 1/1")
        scores = {k: 0.2 + 0.1 * i for i, k in enumerate(TRAIT_KEYS)}
        if self.member == "mm":
            scores["interview"] = 0.44
        return {"scores": dict(scores), "seconds": 1.0, "variants": {self.member: {k: scores[k] for k in TRAIT_KEYS}},
                "members_used": [self.member], "members_failed": {}, "primary": self.member,
                "primary_used": self.member, "transcript": "", "duration_sec": 15.0, "segments": 1, "timeline": []}


class _FakeStudio:
    """A Studio whose backends never load: remembers which member was asked for, the analyses fail harmlessly (the
    pipeline logs and goes on), explanations are never asked for from OCEAN-AI."""
    asr_model = "fake-asr"

    def __init__(self):
        self.stop_event = threading.Event()
        self.asked = []
        self.explained = []
        self._an = {}

    def backend(self, member):
        self.asked.append(member)
        return types.SimpleNamespace(cfg=types.SimpleNamespace(corpus=f"ensemble({member}), main={member}",
                                                                members=(member,)),
                                     backends={member: types.SimpleNamespace(cfg=types.SimpleNamespace(
                                         corpus="mupta" if member == "oceanai" else "own checkpoints"))})

    def analyzer(self, member):
        return self._an.setdefault(member, _FakeAnalyzer(member))

    def mm_backend(self, member):
        if member != "mm":
            raise AssertionError("explanations were asked for from a model other than AMLAI 1.0")
        studio = self

        class _MM:
            def explain_video(self, *a, **kw):
                studio.explained.append(a[0])
                raise RuntimeError("no GPU in the test: the pipeline logs and goes on")
        return _MM()

    @property
    def text_emotion(self):
        raise RuntimeError("no model in the test")

    voice_emotion = face_expression = text_emotion


# ------------------------------------------------------------------------------------------------- tests ---

def test_signatures_one_member_russian_only():
    sig = inspect.signature(pipeline.run_analysis)
    assert "member" in sig.parameters and "lang" not in sig.parameters
    assert sig.parameters["member"].default == "mm" == bs3.DEFAULT_MODEL      # a run without a model: AMLAI 1.0
    assert "members" not in inspect.signature(pipeline.Studio.__init__).parameters
    assert bs3.LANG == "ru" and pipeline.Studio().lang == "ru"
    for bad in ("en", "ensemble", "scene", ""):
        try:
            pipeline.check_member(bad)
        except RuntimeError as e:
            assert "Неизвестная модель" in str(e) and "OCEAN-AI" in str(e) and "AMLAI 1.0" in str(e)
        else:
            raise AssertionError(bad)


def test_studio_builds_one_backend_per_member():
    """One backend of one member at a time: asking for the other member drops the cached one (the GPU holds one
    Big Five model even after the radio is switched in a running server); the same member is not rebuilt."""
    with _Patched(_fake_modules()):
        _FakeEnsemble.built.clear()
        st = pipeline.Studio(asr_model="asr-x", ollama_model="qwen-x", mm_ckpt="/tmp/ckpt.pt")
        mm = st.backend("mm")
        assert mm.cfg.members == ("mm",) and mm.cfg.primary == "mm" and mm.cfg.lang == "ru"
        assert mm.cfg.oceanai_cfg is None and mm.cfg.mm_cfg.lang == "ru"
        assert mm.cfg.mm_cfg.checkpoint == "/tmp/ckpt.pt" and mm.cfg.mm_cfg.ollama_model == "qwen-x"
        assert list(mm.backends) == ["mm"] and mm.loaded == 1
        assert st.mm_backend("mm") is mm.backends["mm"]
        assert st.backend("mm") is mm and list(st._be) == ["mm"]          # cached, not rebuilt
        oa = st.backend("oceanai")
        assert oa is not mm and oa.cfg.members == ("oceanai",) and oa.cfg.primary == "oceanai"
        assert oa.cfg.mm_cfg is None and oa.cfg.oceanai_cfg.lang == "ru" and oa.cfg.oceanai_cfg.asr_model == "asr-x"
        assert list(oa.backends) == ["oceanai"]
        assert list(st._be) == ["oceanai"] and "mm" not in st._an           # the other member is unloaded
        assert st.mm_backend("oceanai") is None                            # explanations exist for AMLAI 1.0 only
        assert st.backend("oceanai") is oa and len(_FakeEnsemble.built) == 2
        mm2 = st.backend("mm")                                             # back again: built anew, OCEAN-AI dropped
        assert mm2 is not mm and list(st._be) == ["mm"] and len(_FakeEnsemble.built) == 3
        assert all(len(b.cfg.members) == 1 for b in _FakeEnsemble.built)
        try:
            st.backend("ensemble")
        except RuntimeError:
            pass
        else:
            raise AssertionError("an unknown member was accepted")


def test_studio_hands_whisper_over_between_analyzers():
    """The segment analyzer of the next member takes the Whisper pipeline of the dropped one instead of loading a
    second copy; at most one analyzer is kept."""
    with _Patched(_fake_modules()):
        st = pipeline.Studio()
        an_mm = st.analyzer("mm")
        assert list(st._an) == ["mm"] and an_mm.backend is st._be["mm"] and an_mm.lang == "ru"
        an_mm._asr = whisper = object()                                     # «loaded» Whisper of this analyzer
        an_oa = st.analyzer("oceanai")
        assert list(st._an) == ["oceanai"] and list(st._be) == ["oceanai"]
        assert an_oa is not an_mm and an_oa._asr is whisper and st._asr_shared is None
        assert st.analyzer("oceanai") is an_oa


def test_ensemble_config_single_member_is_primary():
    from bs3.backend_ensemble import EnsembleConfig
    for m in MEMBERS:
        assert EnsembleConfig(members=(m,), lang="ru").primary == m
        assert EnsembleConfig(members=(m,), lang="en").primary == m
    assert EnsembleConfig(members=("oceanai", "mm"), lang="ru").primary == "oceanai"      # the 3.0 rule is kept
    assert EnsembleConfig(members=("oceanai", "mm"), lang="en").primary is None


def _run(member: str, explain: bool, tmp: Path):
    studio = _FakeStudio()
    tmp.mkdir(parents=True, exist_ok=True)
    video = tmp / "clip.mp4"
    video.write_bytes(b"\x00" * 2048)
    work = tmp / "jobs" / member
    old_pool = pool.POOL_DIR
    pool.POOL_DIR = tmp / "pool"
    log = logging.getLogger("bs3.pipeline")
    old_level = log.level
    log.setLevel(logging.CRITICAL)            # the fakes fail on purpose; the pipeline logs and goes on
    try:
        rep_ = pipeline.run_analysis(studio, work, str(video), member=member, explain=explain, progress=lambda f, d: None)
    finally:
        pool.POOL_DIR = old_pool
        log.setLevel(old_level)
    return studio, rep_, work


def test_run_analysis_one_member_each():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        for member in MEMBERS:
            studio, r, work = _run(member, explain=True, tmp=tmp)
            assert set(studio.asked) == {member}, (member, studio.asked)
            m = r["model"]
            assert m["selected"] == member and m["primary"] == member
            assert m["selected_title"] == bs3.MODEL_TITLES[member] and m["lang"] == "ru"
            assert m["backend"] == member and m["product"] == "BS Profiler 3.1" and m["version"] == "3.1.0a1"
            assert r["modalities_used"] == list(bs3.MODALITIES[member])     # what the model looked at, not its name
            assert set(r["variant_scores"]) == {member}
            # nothing of 2.0 that the tab «Данные» would have to leave out: no stored summary, no percentiles
            assert "narrative" not in r
            if member == "mm":
                assert set(r["interview"]) == {"score", "name_ru", "disclaimer"}
            for k in TRAIT_KEYS:
                assert abs(r["traits"][k]["score"] - r["variant_scores"][member][k]) < 1e-9
                assert "percentile" not in r["traits"][k]                   # Russian speech: the score only
            mb = r["mbti"]
            assert mb["schema_version"] == 3 and mb["model"] == member and "second" not in mb and "agreement" not in mb
            assert mb["source"] == {"oceanai": "ocean_ai", "mm": "own_model"}[member]
            assert mb["model_title"] == bs3.MODEL_TITLES[member] and mb["primary_missing"] is False
            assert ("interview" in r) == (member == "mm")
            # explanations: asked for from AMLAI 1.0 only (they failed harmlessly here)
            assert (len(studio.explained) == 1) == (member == "mm")
            # written where the page reads it
            job = Path(r["job_dir"])
            assert job.parent == work and (job / "result.json").exists()
            saved = json.loads((job / "result.json").read_text(encoding="utf-8"))
            assert saved["model"]["selected"] == member and set(saved["variant_scores"]) == {member}
            assert saved["mbti"]["schema_version"] == 3
            text = json.dumps(saved, ensure_ascii=False)
            for bad in ("второе мнение", "Второе мнение", "своя модель", "среднее двух систем", "MM-PSYCHE"):
                assert bad not in text.replace("по рецепту MM-PSYCHE", ""), (member, bad)
            # the view of a fresh 3.1 job is the job itself, and the tab «Данные» shows the file whole (no note)
            from bs3.scores import clean_view, data_json
            v = clean_view(saved)
            assert v["view_meta"]["main_system"] == member and v["view_meta"]["primary_missing"] is False
            assert ("interview" in v) == (member == "mm")
            shown, trimmed = data_json(saved)
            assert shown == saved and trimmed is False


def test_run_analysis_oceanai_skips_explanations_even_when_asked():
    with tempfile.TemporaryDirectory() as d:
        studio, r, _ = _run("oceanai", explain=True, tmp=Path(d))
        assert studio.explained == [] and r["key_frames"] == []
        studio, r, _ = _run("mm", explain=False, tmp=Path(d) / "b")
        assert studio.explained == [] and r["key_frames"] == []
