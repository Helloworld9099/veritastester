"""Compare declared CPU performance receipts, with no downloads or model execution.

A receipt describes one direction on one isolated host, not an app-wide benchmark.
Its workload, model pins and execution method are independently bound before a
10% gate can run. Cold measurements and audio measurements are separately
summarized; neither can stand in for warm text-only performance.
"""

from collections.abc import Mapping, Sequence
from fractions import Fraction
import hashlib
import json
import re
from statistics import median


SCHEMA_VERSION = 1
MAX_REGRESSION = Fraction(1, 10)
MIN_REPETITIONS = 3
MIN_REQUESTS_PER_REPETITION = 40
MAX_REPEAT_RELATIVE_RANGE = {"warm_p95_ns": Fraction(1, 10), "peak_process_rss_bytes": Fraction(1, 20)}
MAX_PAIR_REGRESSION_RANGE = Fraction(1, 10)
_HOST = {
    "fingerprint_sha256", "cpu", "logical_cpus", "memory_bytes", "os", "kernel",
    "architecture", "cpu_affinity", "power_policy", "runtime_fingerprint_sha256", "isolated",
}
_METHOD = {
    "device": "cpu",
    "workload": "text-only-translation",
    "timer": "monotonic-ns",
    "rss": "peak-process-rss-bytes",
    "isolation": "fresh-process-per-arm-repetition",
    "pairing": "alternating-ab-ba",
}
_REQUIRED = {
    "schema_version", "direction", "checkpoint", "fixture_sha256", "input_mix_sha256",
    "host", "decode", "threads", "methodology", "warm_runs",
}
_OPTIONAL = {"cold_runs", "audio_runs"}
_PIN_KEYS = {"base_revision", "files", "adapter_sha256", "rollback_sha256"}
_WARM_KEYS = {"pair_id", "position", "requests_ns", "peak_rss_bytes", "pressure_ok"}


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _hex(value, length):
    return isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None


def _integer(value, minimum=1):
    return type(value) is int and minimum <= value < 1 << 63


def _s(value):
    return isinstance(value, str) and bool(value.strip())


def _direction(value):
    return (
        isinstance(value, str)
        and re.fullmatch(r"[a-z]{2,3}->[a-z]{2,3}", value) is not None
        and len(set(value.split("->"))) == 2
    )


def _canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))


def input_mix_sha256(inputs: Sequence[str]):
    """Fingerprint ordered exact request strings, including repetitions.

    Strings should encode the entire effective request if it has inputs besides
    source text. The fixture suite digest and decode options are pinned separately.
    This helper never returns or logs the original request strings.
    """
    _check(not isinstance(inputs, (str, bytes)) and isinstance(inputs, Sequence) and bool(inputs),
           "nonempty ordered input sequence required")
    _check(all(type(x) is str for x in inputs), "input strings required")
    inputs_hashed = [hashlib.sha256(x.encode("utf-8")).hexdigest() for x in inputs]
    return hashlib.sha256(_canonical_json(inputs_hashed).encode("utf-8")).hexdigest()


def _validate_pins(value, label):
    _check(isinstance(value, Mapping) and set(value) == _PIN_KEYS, f"{label}: checkpoint pin fields missing or unknown")
    files = value["files"]
    _check(_hex(value["base_revision"], 40), f"{label}: invalid base revision")
    _check(isinstance(files, Mapping) and bool(files)
           and all(_s(k) and _hex(v, 64) for k, v in files.items()), f"{label}: missing or invalid base/tokenizer file hashes")
    _check(all(value[k] is None or _hex(value[k], 64) for k in ("adapter_sha256", "rollback_sha256")),
           f"{label}: invalid adapter or rollback SHA-256")


def _json_values(value):
    # Enforce actual JSON data types so Python's True == 1 cannot make different
    # decode declarations equivalent, and no NaN/infinite values get past comparison.
    if type(value) is dict:
        return all(type(k) is str and _json_values(v) for k, v in value.items())
    if type(value) is list:
        return all(_json_values(v) for v in value)
    return value is None or type(value) in (str, int, bool) or (
        type(value) is float and value == value and abs(value) != float("inf")
    )


def _validate_warm(receipt, pins, label, expected_direction, expected_fixture_sha256, expected_input_mix_sha256, expected_warm_requests):
    _check(isinstance(receipt, Mapping) and _REQUIRED <= set(receipt) <= (_REQUIRED | _OPTIONAL),
           f"{label}: performance receipt fields missing or unknown")
    _check(type(receipt["schema_version"]) is int and receipt["schema_version"] == SCHEMA_VERSION,
           f"{label}: unsupported performance receipt schema")
    direction = receipt["direction"]
    _check(_direction(direction) and (expected_direction is None or direction == expected_direction),
           f"{label}: direction is invalid or does not match the expected direction")
    fixture = receipt["fixture_sha256"]
    _check(_hex(fixture, 64) and (expected_fixture_sha256 is None or fixture == expected_fixture_sha256),
           f"{label}: fixture SHA-256 is invalid or does not match the expected frozen suite")
    mix = receipt["input_mix_sha256"]
    _check(_hex(mix, 64) and (expected_input_mix_sha256 is None or mix == expected_input_mix_sha256),
           f"{label}: ordered input mix SHA-256 is invalid or does not match the expected workload")
    _validate_pins(receipt["checkpoint"], label)
    _check(receipt["checkpoint"] == pins, f"{label}: actual checkpoint is not the independently expected checkpoint")
    host = receipt["host"]
    _check(isinstance(host, Mapping) and set(host) == _HOST and host["isolated"] is True,
           f"{label}: isolated environment has not been documented")
    _check(all(_hex(host[k], 64) for k in ("fingerprint_sha256", "runtime_fingerprint_sha256")),
           f"{label}: missing host or runtime identity")
    _check(all(_s(host[k]) for k in ("cpu", "os", "kernel", "architecture", "power_policy")),
           f"{label}: incomplete CPU host description")
    _check(_integer(host["logical_cpus"]) and _integer(host["memory_bytes"]),
           f"{label}: invalid host CPU count or memory")
    aff = host["cpu_affinity"]
    _check(type(aff) is list and bool(aff) and all(_integer(cpu, 0) for cpu in aff)
           and len(set(aff)) == len(aff) and len(aff) <= host["logical_cpus"],
           f"{label}: invalid CPU affinity")
    thread = receipt["threads"]
    _check(isinstance(thread, Mapping) and set(thread) == {"intraop", "interop"}
           and all(_integer(v) and v <= len(aff) for v in thread.values()),
           f"{label}: invalid CPU thread counts for affinity")
    decode = receipt["decode"]
    _check(type(decode) is dict and bool(decode) and _json_values(decode), f"{label}: exact JSON decode options required")
    method = receipt["methodology"]
    _check(isinstance(method, Mapping) and set(method) == set(_METHOD) | {"warmup_requests"}
           and all(method[k] == v for k, v in _METHOD.items()) and _integer(method["warmup_requests"]),
           f"{label}: warm text-only CPU methodology missing or unsupported")

    runs = receipt["warm_runs"]
    _check(type(runs) is list and len(runs) >= MIN_REPETITIONS,
           f"{label}: at least {MIN_REPETITIONS} independent warm repetitions required")
    seen = set()
    for r in runs:
        _check(isinstance(r, Mapping) and set(r) == _WARM_KEYS, f"{label}: malformed warm trial")
        pair = r["pair_id"]
        _check(_s(pair) and pair not in seen and type(r["position"]) is int and r["position"] in (1, 2),
               f"{label}: invalid pair identifier or trial position")
        seen.add(pair)
        _check(r["pressure_ok"] is True, f"{label}: isolated run pressure qualification missing or failed")
        ns = r["requests_ns"]
        _check(type(ns) is list and len(ns) >= MIN_REQUESTS_PER_REPETITION and all(_integer(x) for x in ns),
               f"{label}: at least {MIN_REQUESTS_PER_REPETITION} positive request timings per repetition required")
        _check(len(ns) == expected_warm_requests,
               f"{label}: warm sample count does not match the independently expected workload")
        _check(_integer(r["peak_rss_bytes"]), f"{label}: missing/invalid peak process RSS in bytes")


def _nearest_rank_p95(values):
    return sorted(values)[(95 * len(values) + 99) // 100 - 1]


def _spread(values):
    ordered = sorted(values)
    middle = Fraction(ordered[(len(ordered) - 1) // 2] + ordered[len(ordered) // 2], 2)
    return Fraction(max(values) - min(values)) / middle


def _metric(name, b, c, b_rounds, c_rounds):
    """Qualify repeat stability independently of a claimed aggregate regression."""
    paired = [Fraction(c1 - b1, b1) for b1, c1 in zip(b_rounds, c_rounds)]
    reg = Fraction(c - b, b)
    issues = []
    b_range, c_range = _spread(b_rounds), _spread(c_rounds)
    if b_range > MAX_REPEAT_RELATIVE_RANGE[name] or c_range > MAX_REPEAT_RELATIVE_RANGE[name]:
        issues.append("per-arm warm repeats too noisy")
    if max(paired) - min(paired) > MAX_PAIR_REGRESSION_RANGE:
        issues.append("paired ratios too noisy")
    pass_direction = reg <= MAX_REGRESSION
    if any((x <= MAX_REGRESSION) != pass_direction for x in paired):
        issues.append("paired repeats disagree with aggregate at the 10% threshold")
    return {
        "status": "incomparable" if issues else ("pass" if pass_direction else "fail"),
        "baseline": b,
        "candidate": c,
        "regression_fraction": float(reg),
        "baseline_repeat_range_fraction": float(b_range),
        "candidate_repeat_range_fraction": float(c_range),
        "paired_regression_fractions": [float(x) for x in paired],
        "reasons": issues,
    }


def _separate_measurement(receipt, key):
    """Optional evidence cannot make warm measurements pass or fail."""
    if not isinstance(receipt, Mapping) or key not in receipt:
        return {"status": "not_measured"}
    rows = receipt[key]
    if type(rows) is not list or not rows:
        return {"status": "invalid", "reason": "nonempty measurement list required"}
    if key == "cold_runs":
        fields = ("load_ns", "first_request_ns", "peak_rss_bytes")
        if not all(isinstance(r, Mapping) and set(r) == set(fields) and all(_integer(r[k]) for k in fields) for r in rows):
            return {"status": "invalid", "reason": "invalid cold load, first inference or peak-process units"}
        return {
            "status": "reported_separately", "n": len(rows), "model_load_median_ns": median(r["load_ns"] for r in rows),
            "first_request_median_ns": median(r["first_request_ns"] for r in rows),
            "peak_process_rss_bytes": max(r["peak_rss_bytes"] for r in rows),
        }
    if not all(isinstance(r, Mapping) and set(r) == {"end_to_end_ns", "fixture_sha256"}
               and _integer(r["end_to_end_ns"]) and _hex(r["fixture_sha256"], 64) for r in rows):
        return {"status": "invalid", "reason": "invalid audio-to-caption timing or audio fixture SHA-256"}
    fixtures = {r["fixture_sha256"] for r in rows}
    if len(fixtures) != 1:
        return {"status": "invalid", "reason": "audio fixture hash varied within receipt"}
    return {
        "status": "reported_separately", "n": len(rows), "fixture_sha256": next(iter(fixtures)),
        "audio_to_caption_p95_ns": _nearest_rank_p95([r["end_to_end_ns"] for r in rows]),
    }


def compare_performance(baseline, candidate, *, baseline_pins, candidate_pins, change_kind,
                        expected_direction=None, expected_fixture_sha256=None, expected_input_mix_sha256=None,
                        expected_warm_requests=None):
    """Compare two single-direction receipts against separate immutable checkpoint pins.

    Infrastructure replay is only comparable at identical model, tokenizer, and
    rollback pins. Model changes may have distinct, independently expected pins;
    that exception does not permit decode, CPU runtime or input changes. Caller
    must take the expected pins from frozen evidence, not from the receipts.
    """
    result = {
        "status": "incomparable", "reasons": [], "metrics": {}, "change_kind": change_kind,
        "max_regression_fraction": float(MAX_REGRESSION), "model_pin_changes": [],
        "cold": {label: _separate_measurement(r, "cold_runs") for label, r in (("baseline", baseline), ("candidate", candidate))},
        "audio": {label: _separate_measurement(r, "audio_runs") for label, r in (("baseline", baseline), ("candidate", candidate))},
    }
    try:
        _check(change_kind in ("infrastructure", "model"), "declare infrastructure or model comparison")
        _check(expected_direction is None or _direction(expected_direction), "invalid expected direction")
        _check(expected_fixture_sha256 is None or _hex(expected_fixture_sha256, 64), "invalid expected fixture SHA-256")
        _check(expected_input_mix_sha256 is None or _hex(expected_input_mix_sha256, 64), "invalid expected input mix SHA-256")
        _check(_integer(expected_warm_requests, MIN_REQUESTS_PER_REPETITION),
               "independently expected warm request count required")
        _validate_pins(baseline_pins, "expected baseline")
        _validate_pins(candidate_pins, "expected candidate")
        changed = sorted(k for k in _PIN_KEYS if baseline_pins[k] != candidate_pins[k])
        result["model_pin_changes"] = changed
        _check(change_kind == "model" or not changed, "infrastructure lane requires identical checkpoint pins")
        _validate_warm(baseline, baseline_pins, "baseline", expected_direction, expected_fixture_sha256, expected_input_mix_sha256, expected_warm_requests)
        _validate_warm(candidate, candidate_pins, "candidate", expected_direction, expected_fixture_sha256, expected_input_mix_sha256, expected_warm_requests)
        for k, description in (
            ("direction", "direction"), ("fixture_sha256", "fixture"), ("input_mix_sha256", "ordered input mix"),
            ("host", "host or runtime identity"), ("threads", "thread counts"), ("methodology", "measurement methodology"),
            ("decode", "decode configuration"),
        ):
            a, b = baseline[k], candidate[k]
            same = _canonical_json(a) == _canonical_json(b) if k == "decode" else a == b
            _check(same, f"different {description}")
        b_runs, c_runs = baseline["warm_runs"], candidate["warm_runs"]
        _check([r["pair_id"] for r in b_runs] == [r["pair_id"] for r in c_runs], "warm trial pair IDs or order differ")
        _check(len(b_runs[0]["requests_ns"]) == len(c_runs[0]["requests_ns"]), "warm trial request counts differ")
        _check(all(a["position"] != b["position"] for a, b in zip(b_runs, c_runs))
               and all(a["position"] != b["position"] for a, b in zip(b_runs, b_runs[1:])),
               "baseline and candidate were not run in matched alternating AB/BA order")
    except (ValueError, TypeError) as exc:
        result["reasons"].append(str(exc))
        return result

    result["direction"] = baseline["direction"]
    result["repetitions_per_arm"] = len(b_runs)
    result["warm_requests_per_repetition"] = len(b_runs[0]["requests_ns"])
    result["metrics"]["warm_p95_ns"] = _metric(
        "warm_p95_ns", _nearest_rank_p95([x for r in b_runs for x in r["requests_ns"]]),
        _nearest_rank_p95([x for r in c_runs for x in r["requests_ns"]]),
        [_nearest_rank_p95(r["requests_ns"]) for r in b_runs],
        [_nearest_rank_p95(r["requests_ns"]) for r in c_runs],
    )
    b_rss = [r["peak_rss_bytes"] for r in b_runs]
    c_rss = [r["peak_rss_bytes"] for r in c_runs]
    result["metrics"]["peak_process_rss_bytes"] = _metric("peak_process_rss_bytes", max(b_rss), max(c_rss), b_rss, c_rss)
    statuses = {m["status"] for m in result["metrics"].values()}
    result["status"] = "fail" if "fail" in statuses else "incomparable" if "incomparable" in statuses else "pass"
    for name, m in result["metrics"].items():
        result["reasons"].extend(f"{name}: {r}" for r in m["reasons"])
        if m["status"] == "fail":
            result["reasons"].append(f"{name}: comparable regression exceeds 10%")
    return result
