"""Capture-clock ASR diagnostic; no microphone, training, or app integration."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from hashlib import sha256
import json
import os
from pathlib import Path
import resource
import subprocess
from tempfile import TemporaryDirectory

from .nemotron_cache_smoke import (
    LOCALES, MANIFEST_PIN, MODEL_REVISION, RATE, _check, _real, verify_audio,
)
from .nemotron_q8_smoke import Q8_PIN, RUNTIME_RELEASE, _pin, _verify_runtime
from .whisper_fleurs_dev import DiagnosticError, edit_distance, words

HEADER_PIN = "ce1addffe445e7061dad090ef2f3d6845303f536661137a875b95d2109cf8f95"
CHUNK = 5120
LOOKAHEAD = 3
NS_PER_SAMPLE = 1_000_000_000 // RATE
MAX_TRACE_BYTES = 2 << 20


def _ordered(rows, repeats):
    es, fr = rows["es"], rows["fr"]
    _check(len(es) == len(fr) == 4, "unexpected_pinned_audio_inventory")
    sequence = [(lang, ix, *rows[lang][ix]) for ix in range(4) for lang in ("es", "fr")]
    return sequence + sequence[:2] * repeats


_FIELDS = {
    "model": {"load_ns", "api_version", "cpu_affinity_count"},
    "clip_start": {"clip", "locale", "samples"},
    "step": {"clip", "seq", "start_sample", "end_sample", "due_ns", "dispatch_ns",
             "push_return_ns", "done_ns"},
    "result": {"clip", "seq", "ready_ns", "final", "processed_samples", "text"},
    "clip_done": {"clip", "released_samples", "steps", "capture_end_ns",
                  "finish_start_ns", "finish_done_ns", "final_count"},
}


def _number(value):
    return type(value) is int and 0 <= value <= 30_000_000_000


def _ms(ns):
    return round(ns / 1_000_000, 3)


def _summarize(raw, sequence):
    expected_index = 0
    model = state = None
    output = []
    finals = {}
    for line in raw.splitlines():
        try:
            e = json.loads(line)
        except (UnicodeError, ValueError, RecursionError):
            raise DiagnosticError("invalid_paced_event_json") from None
        _check(type(e) is dict and type(e.get("kind")) is str and e["kind"] in _FIELDS,
               "unexpected_paced_event")
        kind = e["kind"]
        _check(set(e) == _FIELDS[kind] | {"kind"}, "unexpected_paced_event_fields")
        for k, v in e.items():
            if k not in {"kind", "locale", "api_version", "final", "text"}:
                _check(_number(v), "invalid_paced_event_number")
        if kind == "model":
            _check(model is None and state is None and e["cpu_affinity_count"] == 2,
                   "inconsistent_native_model_lifetime")
            _check(e["api_version"] == "nemo-speech-asr 0.3.0", "invalid_native_api_version")
            model = e
            continue
        _check(model is not None and expected_index < len(sequence)
               and e["clip"] == expected_index, "unexpected_paced_clip")
        lang, ordinal, pcm, reference = sequence[expected_index]
        samples = len(pcm) // 2
        end_ns = samples * NS_PER_SAMPLE
        bounds = list(range(CHUNK, samples, CHUNK)) + [samples]
        due = [b * NS_PER_SAMPLE for b in bounds]
        if kind == "clip_start":
            _check(state is None and e["samples"] == samples and e["locale"] == LOCALES[lang],
                   "unexpected_paced_clip_start")
            state = {"steps": [], "first": None, "final": None, "maxlag": 0, "maxqueue": 0,
                     "maxservice": 0, "native_wait": 0, "last_native_progress": 0, "last_result_ns": 0}
            continue
        _check(state is not None and (state["final"] is None or kind == "clip_done"),
               "paced_event_without_active_clip")
        st = state
        if kind == "step":
            seq = len(st["steps"])
            start = 0 if seq == 0 else bounds[seq-1]
            prev = st["steps"][-1]["done_ns"] if seq else 0
            _check(seq < len(bounds) and e["seq"] == seq and e["start_sample"] == start
                   and e["end_sample"] == bounds[seq] and e["due_ns"] == due[seq]
                   and max(e["due_ns"], prev) <= e["dispatch_ns"] <= e["push_return_ns"] <= e["done_ns"],
                   "invalid_paced_release_or_completion")
            st["steps"].append(e)
            st["maxlag"] = max(st["maxlag"], e["dispatch_ns"]-e["due_ns"])
            newest = bisect_right(due, e["done_ns"])
            st["maxqueue"] = max(st["maxqueue"], bounds[newest-1] - e["end_sample"])
            st["maxservice"] = max(st["maxservice"], e["done_ns"]-e["dispatch_ns"])
        elif kind == "result":
            _check(type(e["text"]) is str and len(e["text"]) <= 8192 and type(e["final"]) is bool
                   and e["ready_ns"] >= st["last_result_ns"] and bool(st["steps"]),
                   "invalid_paced_result")
            seq = len(st["steps"])-1
            step = st["steps"][seq]
            st["last_result_ns"] = e["ready_ns"]
            if e["final"]:
                _check(e["seq"] == len(bounds) == len(st["steps"])
                       and e["processed_samples"] == samples and bool(words(e["text"])),
                       "invalid_paced_final")
                st["final"] = e
            else:
                _check(e["seq"] == seq and st["last_native_progress"] <= e["processed_samples"] <= bounds[seq]
                       and step["push_return_ns"] <= e["ready_ns"] <= step["done_ns"],
                       "invalid_paced_partial")
                st["last_native_progress"] = e["processed_samples"]
                st["native_wait"] = max(st["native_wait"], bounds[seq] - e["processed_samples"])
                if st["first"] is None and words(e["text"]):
                    st["first"] = e["ready_ns"]
        else:
            final = st["final"]
            _check(final is not None and st["steps"] and e["steps"] == len(bounds) == len(st["steps"])
                   and e["released_samples"] == samples and e["final_count"] == 1
                   and e["capture_end_ns"] == end_ns
                   and max(end_ns, st["steps"][-1]["done_ns"]) <= e["finish_start_ns"]
                       <= final["ready_ns"] <= e["finish_done_ns"], "invalid_paced_clip_done")
            ref, hyp = words(reference), words(final["text"])
            key = (lang, ordinal)
            repeat = key in finals
            matches = finals[key] == final["text"] if repeat else None
            finals.setdefault(key, final["text"])
            output.append({
                "language": lang, "repeat_for_reset_check": repeat, "repeat_final_equals_initial": matches,
                "audio_samples": samples, "released_chunks": len(bounds),
                "first_nonempty_partial_ms_from_clip_start": _ms(st["first"]) if st["first"] is not None else None,
                "final_ms_after_scheduled_eof": _ms(final["ready_ns"]-end_ns),
                "max_dispatch_lateness_ms": _ms(st["maxlag"]),
                "max_waiting_audio_ms_at_step_completion": _ms(st["maxqueue"]*NS_PER_SAMPLE),
                "max_step_service_ms": _ms(st["maxservice"]),
                "max_native_unreported_progress_ms_when_partial_emits": _ms(st["native_wait"]*NS_PER_SAMPLE),
                "word_errors": edit_distance(ref, hyp), "reference_words": len(ref),
            })
            state = None
            expected_index += 1
    _check(model is not None and state is None and expected_index == len(sequence),
           "incomplete_paced_clip_sequence")
    return {"observed_model_loads": 1, "load_ms": _ms(model["load_ns"]),
            "native_c_api_version": model["api_version"], "clips": output}

def _invoke(args, sequence):
    exe = _real(args.exe)
    native = _verify_runtime(exe)
    wheel = _pin(args.model, Q8_PIN, 800_000_000)
    sdk = exe.parent.parent
    header = _pin(sdk / "include/nemo_speech/asr.h", HEADER_PIN, 100_000)
    source = Path(__file__).with_suffix(".cpp").resolve(strict=True)
    cpus = sorted(os.sched_getaffinity(0))[:2]
    _check(len(cpus) == 2, "two_cpu_cores_required")
    with TemporaryDirectory(prefix="bridgecaption-q8-paced-") as directory:
        tmp = Path(directory)
        bench = tmp / "paced"
        toolchain = ["/usr/bin/c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                     "-I", str(sdk / "include"), str(source), "-L", str(sdk / "lib"),
                     "-Wl,-rpath," + str(sdk / "lib"), "-lnemo_speech_asr_c", "-pthread", "-Wl,--exclude-libs,ALL", "-static-libstdc++", "-static-libgcc",
                     "-o", str(bench)]
        comp = subprocess.run(toolchain, stdin=subprocess.DEVNULL, capture_output=True,
                              env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"}, timeout=30)
        _check(comp.returncode == 0, "native_helper_build_failed")
        env = {"HOME": directory, "XDG_CONFIG_HOME": directory, "XDG_CACHE_HOME": directory,
               "LC_ALL": "C.UTF-8", "OMP_NUM_THREADS": "2", "OMP_THREAD_LIMIT": "2", "NO_COLOR": "1"}
        command = ["/usr/bin/taskset", "-c", ",".join(map(str, cpus)), str(bench),
                   str(args.model), str(CHUNK), str(LOOKAHEAD), "25000"]
        for n, (lang, _, pcm, _) in enumerate(sequence):
            p = tmp / f"{n}.pcm"
            p.write_bytes(pcm)
            command.extend((LOCALES[lang], str(p)))
        worker = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                                env=env, timeout=min(600, 25 * len(sequence) + 30))
        _check(worker.returncode == 0, "paced_native_execution_failed")
        _check(0 < len(worker.stdout) <= MAX_TRACE_BYTES, "invalid_native_trace_size")
        result = _summarize(worker.stdout, sequence)
    return {
        "schema_version": 1, "kind": "bridgecaption_nemotron_q8_paced_asr",
        "scope": "pinned_publisher_development_asr_capture_simulation",
        "vendor_model_revision": MODEL_REVISION, "model": wheel, "native_cli": native,
        "native_release": RUNTIME_RELEASE, "native_c_header": header,
        "helper_source_sha256": sha256(source.read_bytes()).hexdigest(),
        "manifest_sha256": MANIFEST_PIN, "cpu_affinity_cores": len(cpus),
        "native_backend_threads": "c_api_default_not_specified", "omp_thread_limit": 2,
        "chunk_samples": CHUNK, "lookahead_tokens": LOOKAHEAD,
        "child_peak_rss_kib_including_build": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "microphone_used": False, "gui_measured": False, "translation_head_present": False,
        "training_executed": False, **result,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("exe", "model", "manifest", "audio-root", "report"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--repeat-pairs", type=int, choices=range(3), default=1,
                   help="Repeat the initial ES/FR pair after the 8 primary clips")
    args = p.parse_args(argv)
    try:
        _check(not args.report.exists() and not args.report.is_symlink(), "existing_report_refused")
        sequence = _ordered(verify_audio(args.manifest, args.audio_root), args.repeat_pairs)
        result = _invoke(args, sequence)
        fd = os.open(args.report, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as report:
            json.dump(result, report, indent=2, sort_keys=True, allow_nan=False)
            report.write("\n")
    except DiagnosticError as exc:
        p.exit(2, f"Paced ASR blocked: {exc}\n")
    except Exception as exc:
        p.exit(2, f"Paced ASR failed without exposing private inputs: {type(exc).__name__}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
