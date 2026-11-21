"""Recorded-file CPU smoke for the pinned vendor Q8 runtime (no microphone)."""
from __future__ import annotations

import argparse
from hashlib import file_digest, sha256
import io
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter
import wave

from .nemotron_cache_smoke import (
    CLIP_PINS, LOCALES, MANIFEST_PIN, MODEL_REVISION, RATE, _check, _real, verify_audio,
)
from .whisper_fleurs_dev import DiagnosticError, edit_distance, words

Q8_PIN = "3fc991d3badad7277c11030a7519832cddaf2057aafed6d4b25147e953a070b1"
RUNTIME_RELEASE = "v0.3.0"
RUNTIME_ARCHIVE_PIN = "255f12b722ab15b2a38a51cf125d56e3fa97058b528f7e895b39d6a473deb06e"
RUNTIME_EXE_PIN = "b38ff5adbe5faba96dea1482e1ba9f9c048d3da07f38b1ca2b092a06c968de61"
RUNTIME_TREE_PIN = "3ca1eb98ffb986ff4e3edf24aeeb621035c4cc1cafbec20499984c09a3e38130"
THREADS = 2


def _pin(path, expected, ceiling):
    file = _real(path)
    with file.open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        _check(0 < size <= ceiling, "artifact_size_out_of_bounds")
        digest = file_digest(stream, "sha256").hexdigest()
    _check(digest == expected, "artifact_pin_mismatch")
    return {"sha256": digest, "bytes": size}


def _canonical_wav(pcm):
    with io.BytesIO() as target:
        with wave.open(target, "wb") as stream:
            stream.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
            stream.writeframes(pcm)
        return target.getvalue()


def _verify_runtime(exe):
    identity = _pin(exe, RUNTIME_EXE_PIN, 100_000_000)
    root = exe.parent.parent
    parts = [exe, *(p for p in _real(root / "lib", is_dir=True).iterdir()
                    if not p.is_dir() or p.is_symlink())]
    _check(len(parts) == 30 and exe.relative_to(root).as_posix() == "bin/nemo-speech",
           "runtime_installation_shape_mismatch")
    records = []
    for p in sorted(parts):
        if p.is_symlink():
            kind, detail = "link", str(p.readlink())
        else:
            with _real(p).open("rb") as stream:
                _check(0 < os.fstat(stream.fileno()).st_size < 100_000_000, "invalid_runtime_file_size")
                kind, detail = "file", file_digest(stream, "sha256").hexdigest()
        records.append((p.relative_to(root).as_posix(), [kind, detail]))
    pin = sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()
    _check(pin == RUNTIME_TREE_PIN, "runtime_binary_or_library_mismatch")
    return {**identity, "binary_and_library_tree_sha256": pin}


def _run_one(exe, model, wav, lang, env, save, repeats):
    args = [str(exe), "--quiet", "bench", "asr", str(wav), "--model", str(model),
            "--device", "cpu", "--mode", "stream", "--language", lang, "--chunk-ms", "320",
            "--asr.streaming.rnnt_right_context", "3", "--asr.backend.threads", str(THREADS),
            "--concurrency", "1", "--repetitions", str(repeats), "--warmup", "0",
            "--save", str(save), "--json"]
    t = perf_counter()
    try:
        process = subprocess.run(args, stdin=subprocess.DEVNULL, capture_output=True,
                                 env=env, timeout=45, check=False)
    except subprocess.TimeoutExpired:
        raise DiagnosticError("native_clip_timed_out") from None
    elapsed = perf_counter() - t
    _check(process.returncode == 0, "native_transcription_failed")
    return process.stdout, elapsed



def _read_result(output, expected_seconds, repeats):
    try:
        item = json.loads(output)
        runs = item["runs"]
        run = runs[0]
        latency = run["latency_ms"]
        mean = latency["mean"]
        corpus_seconds = item["corpus_audio_seconds"]
    except (UnicodeError, ValueError, RecursionError, KeyError, TypeError, IndexError):
        raise DiagnosticError("native_bench_json_invalid") from None
    _check(item.get("mode") == "stream" and item.get("chunk_ms") == 320
           and item.get("files") == 1 and len(runs) == 1 and run.get("concurrency") == 1
           and run.get("items") == repeats and run.get("transcript_mismatches") == 0,
           "native_bench_protocol_mismatch")
    _check(type(corpus_seconds) in (int, float) and abs(corpus_seconds - expected_seconds) < 0.01
           and type(mean) in (int, float) and 0 < mean < 45_000,
           "native_bench_measurement_invalid")
    return round(mean, 3)

def run(args):
    _check(platform.system() == "Linux" and platform.machine() == "x86_64",
           "native_runtime_linux_x86_64_required")
    exe = _verify_runtime(args.exe)
    model = _pin(args.model, Q8_PIN, 800_000_000)
    rows = verify_audio(args.manifest, args.audio_root)
    clip_hashes = {language: [v for k, v in CLIP_PINS.items() if k.startswith(language+"-")]
                   for language in rows}
    summary = {}
    with TemporaryDirectory(prefix="bridgecaption-q8-") as directory:
        tmp = Path(directory)
        # Use no user config, cached model shortcuts, GPU, or provider credentials.
        env = {"HOME": directory, "XDG_CONFIG_HOME": directory, "XDG_CACHE_HOME": directory,
               "LC_ALL": "C.UTF-8", "OMP_NUM_THREADS": str(THREADS), "NO_COLOR": "1"}
        for language, items in sorted(rows.items()):
            locale = LOCALES[language]
            results = []
            for i, (pcm, reference) in enumerate(items):
                wav = tmp / "clip.wav"
                canonical = _canonical_wav(pcm)
                _check(sha256(canonical).hexdigest() == clip_hashes[language][i],
                       "reconstructed_wav_pin_mismatch")
                wav.write_bytes(canonical)
                seconds = len(pcm) / (2 * RATE)
                save = tmp / f"{language}-{i}"
                output, elapsed = _run_one(args.exe, args.model, wav, locale, env, save, args.repeats)
                mean = _read_result(output, seconds, args.repeats)
                with _real(save / "clip.txt").open(encoding="utf-8") as transcript:
                    observed = words(transcript.read(8193))
                expected = words(reference)
                _check(0 < len(observed) <= 2048 and bool(expected), "empty_or_excessive_transcription")
                results.append({"seconds_audio": seconds, "mean_unpaced_stream_ms": mean,
                                "full_cli_wall_ms_for_all_repeats": round(elapsed * 1000, 3),
                                "word_errors": edit_distance(expected, observed),
                                "reference_words": len(expected)})
            baseline = sum(x["reference_words"] for x in results)
            summary[language] = {
                "clips": results,
                "wer_pct": round(100 * sum(x["word_errors"] for x in results)/baseline, 4),
                "audio_seconds": round(sum(x["seconds_audio"] for x in results), 4),
                "sum_mean_unpaced_stream_ms": round(sum(x["mean_unpaced_stream_ms"]
                                                       for x in results), 3),
            }
    return {
        "schema_version": 1, "kind": "bridgecaption_nemotron_q8_recorded_asr_smoke",
        "scope": "pinned_publisher_development_asr_only",
        "vendor_model_revision": MODEL_REVISION, "model": model,
        "vendored_release": RUNTIME_RELEASE, "vendor_release_archive_sha256": RUNTIME_ARCHIVE_PIN,
        "vendor_cli": exe, "manifest_sha256": MANIFEST_PIN, "threads": THREADS,
        "repeat_count": args.repeats, "repeated_text_identical": args.repeats > 1,
        "execution_scope": "unpaced_recorded_stream_native_320ms_new_cli_process_per_clip",
        "timing_scope": "native_per_item_stream_compute_after_engine_warmup",
        "child_peak_rss_kib_so_far": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "paced_live_latency_measured": False, "translation_head_present": False,
        "training_executed": False, "asr": summary,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("exe", "model", "manifest", "audio-root", "report"):
        p.add_argument("--"+key, type=Path, required=True)
    p.add_argument("--repeats", type=int, choices=(1, 2), default=2)
    args = p.parse_args(argv)
    try:
        _check(not args.report.exists() and not args.report.is_symlink(), "existing_report_refused")
        result = run(args)
        fd = os.open(args.report, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as report:
            json.dump(result, report, indent=2, sort_keys=True, allow_nan=False)
            report.write("\n")
    except DiagnosticError as exc:
        p.exit(2, f"Q8 smoke blocked: {exc}\n")
    except Exception as exc:
        p.exit(2, f"Q8 smoke failed without exposing inputs: {type(exc).__name__}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
