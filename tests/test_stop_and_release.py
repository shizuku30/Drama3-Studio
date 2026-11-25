"""Stop the job and release its model, including during streamed preview."""
import json
from threading import Event, Thread
import time
from types import SimpleNamespace
from urllib.request import Request, urlopen
from unittest.mock import Mock
import wave

import ui.server as server


def test_stop_and_release_while_preview_is_publishing(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(server, "load_native_runtime", lambda _: {"protocol": 5})
    state = server.AppState()
    monkeypatch.setattr(server, "STATE", state)
    state.checkpoint = tmp_path / "model.gguf"
    state.checkpoint.touch()
    engine = state._engine = SimpleNamespace(
        is_gguf=True, loaded=True, close=Mock(), build={"protocol": 5}, status_callback=None)
    first_part = Event()

    def generate(mode, _text, output, *, streaming_audio, segment_ready, cancelled, **_):
        assert mode == "tts" and streaming_audio
        part = output.with_name("part.wav")
        with wave.open(str(part), "wb") as sink:
            sink.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
            sink.writeframes(b"\x01\x01" * 4096)
        segment_ready({"index": 0, "path": str(part), "samples": 4096, "sample_rate": 44100,
                       "start_sample": 0, "end_sample": 4096})
        first_part.set()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if cancelled():
                raise server.GenerationCancelled("Stopped during preview")
            time.sleep(0.01)
        raise AssertionError("Active generation was not cancelled")

    monkeypatch.setattr(server, "GGUFWorkflows", lambda _: SimpleNamespace(run=generate))
    listener = server.StudioHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = Thread(target=listener.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{listener.server_port}"
    try:
        job_id = state.create_job({"text": "Speak a streaming sample.", "preview_audio": True,
                                   "silence_retry": False})
        assert first_part.wait(3)
        with urlopen(f"{base}/api/jobs/{job_id}/preview", timeout=2) as reply:
            preview = json.load(reply)
        assert preview["status"] == "running" and preview["parts"] and not preview["done"]
        request = Request(f"{base}/api/jobs/{job_id}/stop-and-release", data=b"{}",
                          method="POST", headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=4) as reply:
            stopped = json.load(reply)
        assert stopped == {"id": job_id, "cancel_requested": True, "resources_released": True}
        assert state.job(job_id)["status"] == "cancelled"
        tail = state.preview_audio(job_id, 1, timeout=0)
        assert tail == {"parts": [], "status": "cancelled", "done": True}
        engine.close.assert_called_once()
        assert state._engine is None and not state._releasing_resources
    finally:
        listener.shutdown()
        listener.server_close()
        thread.join(timeout=2)
        state.shutdown()


def test_stop_releases_after_generation_has_already_completed(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    state = server.AppState()
    engine = state._engine = SimpleNamespace(close=Mock())
    job_id = "a" * 32
    wav = tmp_path / "finished.wav"
    wav.write_bytes(b"finished audio")
    state._jobs[job_id] = {"status": "complete", "output": str(wav)}
    try:
        result = state.release_resources(stop_job=job_id)
        assert result == {"id": job_id, "cancel_requested": False, "resources_released": True}
        engine.close.assert_called_once()
        assert state._engine is None
        assert state.job(job_id) == {"status": "complete", "output": str(wav)}
        assert wav.read_bytes() == b"finished audio"
    finally:
        state.shutdown()


def test_stop_does_not_cancel_or_release_other_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    state = server.AppState()
    engine = state._engine = SimpleNamespace(close=Mock())
    current, other = "a" * 32, "b" * 32
    state._jobs[current] = {"status": "running", "events": []}
    state._jobs[other] = {"status": "queued", "events": []}
    (state.jobs_root / current).mkdir()
    try:
        result = state.release_resources(stop_job=current)
        assert result == {"id": current, "cancel_requested": True, "resources_released": False,
                          "other_jobs_active": True}
        assert state.cancelled(current) and not state.cancelled(other)
        assert state.job(other)["status"] == "queued"
        assert state._engine is engine and not state._releasing_resources
        engine.close.assert_not_called()
    finally:
        state._jobs.clear()
        state.shutdown()
