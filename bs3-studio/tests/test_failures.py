"""bs3.errors (refactoring plan of 3.1, stage 20): one table that turns any exception of an analysis into a calm Russian
sentence, and is_fatal, which says whether the server must be restarted. Every message is Russian, without a traceback,
the word «Error» or a server path; the models' English exceptions are mapped, never shown. The order of the table is
pinned here, because a wrong order would give the wrong advice (e.g. «нет места» instead of «повреждённое видео»)."""
from __future__ import annotations

import contextlib
import re
import subprocess

from bs3 import errors
from bs3.errors import (AnalysisCancelled, NoSegmentsAnalysed, OllamaUnavailable, UserFacingError, is_fatal,
                        user_message)


class _Usage:
    def __init__(self, free: int):
        self.free = free


@contextlib.contextmanager
def _disk(free_mb: float):
    """shutil.disk_usage(...).free fixed at `free_mb` megabytes while the block runs."""
    real = errors.shutil.disk_usage
    errors.shutil.disk_usage = lambda p: _Usage(int(free_mb * 1024 * 1024))
    try:
        yield
    finally:
        errors.shutil.disk_usage = real


CANONICAL = (errors.CANCELLED, errors.OOM, errors.OLLAMA, errors.NO_SIGNAL, errors.ENSEMBLE, errors.BAD_VIDEO,
             errors.CUDA, errors.DISK, errors.NO_WRITE, errors.GENERIC)


def test_the_canonical_messages_are_clean_russian_sentences():
    """Each mapped sentence is Russian, ends with a full stop, and carries no traceback, «Error» or path (Latin product
    names such as «Ollama» / «AMLAI 1.0» are allowed)."""
    for m in CANONICAL:
        assert re.search(r"[А-Яа-яЁё]", m), m
        assert m.strip().endswith(".") and "\n" not in m, m
        for bad in ("Error", "Traceback", "Exception", "/", "\\"):
            assert bad not in m, (bad, m)


def test_user_message_table():
    low_disk = errors._DISK_FREE_MIN / (1024 * 1024) - 1        # just under the threshold, in MB
    # (exception, work_dir, expected message)
    rows = [
        (AnalysisCancelled("остановлено пользователем"), None, errors.CANCELLED),
        # a UserFacingError is already meant for the user and passes through, Latin names and all
        (UserFacingError("Файл не найден: /home/u/x.mp4"), None, "Файл не найден: /home/u/x.mp4"),
        (UserFacingError("Не отвечает Ollama — модель для AMLAI 1.0."), None,
         "Не отвечает Ollama — модель для AMLAI 1.0."),
        (OllamaUnavailable("connection refused"), None, errors.OLLAMA),
        (RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"), None, errors.OOM),
        # NoSegmentsAnalysed: the shared reason of the skipped segments decides the advice
        (NoSegmentsAnalysed("no segment", errors=["CUDA out of memory", "seg2: cuda out of memory"]), None, errors.OOM),
        (NoSegmentsAnalysed("no segment", errors=["Ollama timed out", "ollama connection refused"]), None,
         errors.OLLAMA),
        (NoSegmentsAnalysed("no segment", errors=["no face found", "no speech found"]), None, errors.NO_SIGNAL),
        (NoSegmentsAnalysed("no segment", errors=[]), None, errors.NO_SIGNAL),
        (RuntimeError("CUDA error: device-side assert triggered"), None, errors.CUDA),
        (OSError(28, "No space left on device"), None, errors.DISK),
        (subprocess.CalledProcessError(1, ["ffmpeg", "in.mp4"]), "/work", errors.DISK),      # ffmpeg + low disk
        (OSError(5, "I/O error"), "/work", errors.DISK),                                     # any OSError + low disk
        (OSError(13, "Permission denied"), None, errors.NO_WRITE),
        (OSError(30, "Read-only file system"), None, errors.NO_WRITE),
        (RuntimeError("No segment could be analysed"), None, errors.NO_SIGNAL),
        (RuntimeError("no predictions for any file"), None, errors.NO_SIGNAL),
        (RuntimeError("no frames decoded"), None, errors.NO_SIGNAL),
        (RuntimeError("All ensemble members failed: oceanai"), None, errors.ENSEMBLE),
        (subprocess.CalledProcessError(1, ["ffprobe", "x.mp4"]), None, errors.BAD_VIDEO),    # no work dir: not disk
        (RuntimeError("Неизвестная модель. Выберите другую."), None, "Неизвестная модель. Выберите другую."),
        (ValueError("unexpected value 3"), None, errors.GENERIC),
        (KeyError("traits"), None, errors.GENERIC),
    ]
    with _disk(low_disk):
        for e, wd, want in rows:
            assert user_message(e, wd) == want, (repr(e), user_message(e, wd))
    # with plenty of free space an ffmpeg failure is a broken video, not a full disk
    with _disk(errors._DISK_FREE_MIN / (1024 * 1024) + 1000):
        assert user_message(subprocess.CalledProcessError(1, ["ffmpeg", "in.mp4"]), "/work") == errors.BAD_VIDEO


def test_the_old_texts_are_word_for_word():
    """The four texts and the generic fallback that the page showed before this stage are unchanged (baseline)."""
    assert user_message(RuntimeError("CUDA out of memory")) == \
        "Не хватило памяти видеокарты. Подождите минуту и запустите анализ заново."
    assert user_message(RuntimeError("no frames decoded")) == \
        "В ролике не найдено ни лица, ни речи, поэтому оценить его нельзя. Проверьте файл."
    assert user_message(RuntimeError("All ensemble members failed: oceanai")) == \
        "Модель не смогла обработать ролик: чаще всего в кадре не найдено лицо или не слышна речь. Проверьте файл."
    assert user_message(subprocess.CalledProcessError(1, ["ffprobe"])) == \
        "Не удалось прочитать видеофайл: возможно, он повреждён или записан в неподдерживаемом формате."
    assert user_message(ValueError("boom")) == \
        "Не удалось обработать ролик из-за внутренней ошибки. Подробности записаны в журнал сервера."


def test_a_user_facing_error_passes_through_with_its_latin_names():
    e = UserFacingError("Не отвечает Ollama — модель для AMLAI 1.0. Запустите Ollama.")
    assert user_message(e) is str(e) or user_message(e) == str(e)
    assert "Ollama" in user_message(e) and "AMLAI 1.0" in user_message(e)


def test_is_fatal():
    """Fatal (the server must be restarted): a broken Ollama connection and a CUDA fault other than out of memory.
    Not fatal: a timeout, out of memory, and ordinary failures."""
    refused = OllamaUnavailable("ollama down")
    refused.__cause__ = ConnectionRefusedError(111, "Connection refused")
    assert is_fatal(refused) is True
    assert is_fatal(OllamaUnavailable("HTTP 404 model not found")) is True
    assert is_fatal(OllamaUnavailable("connection refused by the daemon")) is True
    assert is_fatal(OllamaUnavailable("read timed out")) is False
    assert is_fatal(OllamaUnavailable("something odd")) is False
    assert is_fatal(RuntimeError("CUDA error: device-side assert triggered")) is True
    assert is_fatal(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")) is False
    assert is_fatal(RuntimeError("a plain error")) is False
    assert is_fatal(subprocess.CalledProcessError(1, ["ffprobe"])) is False
