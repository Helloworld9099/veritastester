"""Offline DEV-only interoperability CLI. No check ever opens a sealed/final dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

M0_SUITE_SHA256 = "d5fce03b8c49d9dd071da0619661ea57106c6847b750101cb4d0693b3ae51a0c"
M0_MANIFEST_SHA256 = "7dc7be2a0df65f8507a67f9cf1e61b558fefdecad3bb81d8552cd4b62b5d36f4"
M0_REVISION = "ca755aec8f7111f03145e18ef9948cd6127aaec0"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_LANG = re.compile(r"[a-z]{2,3}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_MAX_FILE_BYTES = 64 * 1024 * 1024


class Rejected(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class RedactedParser(argparse.ArgumentParser):
    def error(self, message):
        # argparse's stock diagnostics echo arbitrary local paths/arguments.
        raise Rejected("invalid_arguments")


def _require(condition, code):
    if not condition:
        raise Rejected(code)


def _pin(s):
    return isinstance(s, str) and _SHA.fullmatch(s) is not None


def _hash(b):
    return hashlib.sha256(b).hexdigest()


def _data_hash(obj):
    return _hash(json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True,
                            allow_nan=False).encode("utf-8"))


def _raw(path: Path, expected: str, label: str) -> bytes:
    _require(_pin(expected), f"{label}_checksum_required")
    try:
        # Read with a size bound even if a file changes after stat.
        _require(path.is_file() and not path.is_symlink(), f"{label}_unavailable")
        with path.open("rb") as stream:
            data = stream.read(_MAX_FILE_BYTES + 1)
    except OSError:
        raise Rejected(f"{label}_unavailable") from None
    _require(0 < len(data) <= _MAX_FILE_BYTES, f"{label}_invalid_size")
    _require(_hash(data) == expected, f"{label}_checksum_mismatch")
    return data


def _unique(items):
    out = {}
    for k, v in items:
        if k in out:
            raise ValueError("duplicate field")
        out[k] = v
    return out


def _invalid_number(_):
    raise ValueError("nonstandard JSON constant")


def _parse(blob, label):
    try:
        value = json.loads(blob.decode("utf-8"), object_pairs_hook=_unique,
                           parse_constant=_invalid_number)
        pending = [value]
        while pending:
            item = pending.pop()
            if isinstance(item, str):
                if "\0" in item:
                    raise ValueError("NUL in input metadata or text")
                item.encode("utf-8")  # rejects escaped unpaired surrogates
            elif type(item) is float and not math.isfinite(item):
                raise ValueError("unrepresentable number")
            elif isinstance(item, dict):
                pending.extend(item.keys())
                pending.extend(item.values())
            elif isinstance(item, list):
                pending.extend(item)
        return value
    except (ValueError, UnicodeError, RecursionError):
        raise Rejected(f"{label}_invalid_json") from None


def _records(blob, label):
    _require(blob.startswith(b"{"), f"{label}_invalid_jsonl")
    rows = [_parse(line, label) for line in blob.split(b"\n") if line.strip()]
    _require(rows and all(isinstance(row, dict) for row in rows), f"{label}_invalid_jsonl")
    return rows


def _lang(a):
    return isinstance(a, str) and _LANG.fullmatch(a) is not None


def _route(a, b):
    _require(_lang(a) and _lang(b) and a != b, "invalid_direction")
    return f"{a}->{b}"


def _wire_route(name):
    _require(isinstance(name, str), "invalid_direction")
    parts = name.split("->")
    _require(len(parts) == 2, "invalid_direction")
    return _route(*parts)


def _dev(meta, *, kind, label):
    _require(isinstance(meta, dict), f"{label}_invalid_metadata")
    # Deny unknown scope/split by requiring explicit markers. No permissive
    # aliases such as 'validation', 'gold', 'test', or 'holdout'.
    _require(meta.get("kind") == kind and type(meta.get("schema_version")) is int
             and meta["schema_version"] == 1, f"{label}_invalid_metadata")
    _require(meta.get("scope") == "development" and meta.get("split") == "dev"
             and meta.get("sealed") is False, f"{label}_not_open_development")
    for key in ("holdout", "holdout_opened", "sealed_holdout", "independent_final",
                "test_opened", "independent_holdout_opened", "final_holdout"):
        _require(key not in meta or meta[key] is False, f"{label}_not_open_development")


def _checkrow(row, *, m0=False):
    if m0:
        _require(set(row) == {"id", "direction", "source_lang", "target_lang", "source",
                             "reference", "tags", "baseline_output", "generation_retry_for_loop"},
                 "fixture_invalid_row")
        source, target = row["source_lang"], row["target_lang"]
        _require(row["direction"] == f"{source}-{target}", "fixture_invalid_row")
        row = {"id": row["id"], "source_language": source, "target_language": target,
               "source": row["source"], "reference": row["reference"], "categories": row["tags"]}
    else:
        required = {"id", "source_language", "target_language", "source", "reference"}
        allowed = required | {"categories", "checks", "context"}
        _require(required <= row.keys() <= allowed, "fixture_invalid_row")
    key = row["id"]
    for item in ("source", "reference"):
        _require(isinstance(row[item], str) and row[item].strip(), "fixture_invalid_row")
    _require(isinstance(key, str) and _ID.fullmatch(key) is not None, "fixture_invalid_row")
    route = _route(row["source_language"], row["target_language"])
    cats = row.get("categories", [])
    _require(isinstance(cats, list) and all(isinstance(c, str) and _ID.fullmatch(c) for c in cats)
             and len(set(cats)) == len(cats), "fixture_invalid_categories")
    checks = row.get("checks", [])
    _require(isinstance(checks, list) and all(isinstance(c, dict) for c in checks),
             "fixture_invalid_checks")
    case = {"id": key, "source": row["source"], "reference": row["reference"],
            "categories": cats, "checks": checks}
    if "context" in row:
        context = row["context"]
        _require(isinstance(context, list)
                 and all(isinstance(t, str) and t.strip() for t in context), "fixture_invalid_context")
        case["context"] = context
    return key, route, case


def load_suite(args):
    """Preflight metadata (and known M0 fingerprints) before opening fixture bytes."""
    is_m0 = args.suite_format == "m0"
    if is_m0:
        _require(args.manifest_sha256 == M0_MANIFEST_SHA256
                 and args.suite_sha256 == M0_SUITE_SHA256, "m0_published_checksums_required")
    manifest = _parse(_raw(args.manifest, args.manifest_sha256, "manifest"), "manifest")
    if is_m0:
        _require(isinstance(manifest, dict)
                 and manifest.get("source_commit") == M0_REVISION
                 and manifest.get("fixture_origin", {}).get("scope") == "development_only"
                 and manifest.get("limitations", {}).get("independent_holdout_opened") is False,
                 "manifest_not_open_development")
    else:
        _dev(manifest, kind="dev_suite_manifest", label="manifest")
        required_fields = {"kind", "schema_version", "scope", "split", "sealed", "fixture_sha256",
                           "fixture_rows", "directions"}
        optional_fields = {"baseline_outputs_sha256_by_direction", "checkpoint_pins",
                           "performance_requests_by_direction"}
        _require(required_fields <= manifest.keys() <= required_fields | optional_fields,
                 "manifest_invalid_metadata")
        declared = manifest.get("directions")
        _require(isinstance(declared, dict) and bool(declared)
                 and all(_wire_route(k) == k and type(n) is int and n > 0
                         for k, n in declared.items()), "manifest_explicit_dev_directions_required")
        _require(type(manifest.get("fixture_rows")) is int
                 and manifest["fixture_rows"] == sum(declared.values()), "manifest_fixture_row_count_mismatch")
    _require(_pin(args.suite_sha256) and manifest.get("fixture_sha256") == args.suite_sha256,
             "suite_checksum_not_in_manifest")
    rows = _records(_raw(args.suite, args.suite_sha256, "suite"), "suite")
    n = manifest.get("fixture_rows")
    _require(type(n) is int and n == len(rows), "fixture_row_count_mismatch")
    cases, route_by_id, old = {}, {}, {}
    for original in rows:
        key, route, case = _checkrow(original, m0=is_m0)
        _require(key not in route_by_id, "duplicate_fixture_case")
        route_by_id[key] = route
        cases.setdefault(route, []).append(case)
        if is_m0:
            output = original["baseline_output"]
            _require(isinstance(output, str) and output.strip(), "m0_invalid_baseline_record")
            old[key] = output
    if is_m0:
        _require(set(cases) == {"en->es", "es->en", "fr->en"}
                 and all(len(c) == 6 for c in cases.values()), "m0_direction_layout_mismatch")
    else:
        _require(manifest["directions"] == {k: len(v) for k, v in cases.items()},
                 "fixture_directions_mismatch")
    return manifest, cases, route_by_id, old


def load_receipt(path, expected, variant, fixture_sha, route_by_id):
    _require(_pin(expected), f"{variant}_checksum_required")
    claimed = _preview_metadata(path, variant)
    _dev(claimed, kind="dev_prediction_receipt", label=variant)
    _require(set(claimed) == {"kind", "schema_version", "scope", "split", "sealed",
                             "variant", "fixture_sha256", "directions"},
             f"{variant}_invalid_metadata")
    _require(claimed.get("variant") == variant and claimed.get("fixture_sha256") == fixture_sha,
             f"{variant}_receipt_binding_mismatch")
    records = _records(_raw(path, expected, variant), variant)
    meta = records[0]
    _require(meta == claimed, f"{variant}_receipt_binding_mismatch")
    _dev(meta, kind="dev_prediction_receipt", label=variant)
    _require(meta.get("variant") == variant and meta.get("fixture_sha256") == fixture_sha,
             f"{variant}_receipt_binding_mismatch")
    dirs_info = meta.get("directions")
    _require(isinstance(dirs_info, dict), f"{variant}_invalid_directions")
    routes = set(route_by_id.values())
    _require(set(dirs_info) == routes, f"{variant}_invalid_directions")
    values = {}
    for row in records[1:]:
        _require(set(row) == {"id", "direction", "output"}, f"{variant}_invalid_output_record")
        key, route, output = row["id"], row["direction"], row["output"]
        _require(isinstance(key, str) and key in route_by_id and key not in values
                 and _wire_route(route) == route_by_id[key]
                 and isinstance(output, str), f"{variant}_invalid_output_record")
        values[key] = output
    _require(values.keys() == route_by_id.keys(), f"{variant}_incomplete_output_coverage")
    for d in dirs_info.values():
        _require(isinstance(d, dict) and d.keys() <= {"checkpoint", "performance", "input_mix_sha256"},
                 f"{variant}_invalid_direction_metadata")
    return values, dirs_info


def _m0_baseline(manifest, rows):
    details = {}
    for d, info in manifest["directions"].items():
        route = _route(*d.split("-"))
        details[route] = {"checkpoint": {"base_revision": info["base_revision"],
                           "files": info["base_file_sha256"],
                           "adapter_sha256": info["active_adapter_sha256"],
                           # Rollback/performance do not follow from the baseline payload.
                           "rollback_sha256": None}}
    return rows, details


def _preview_metadata(path, variant):
    """Check explicit DEV header before reading or parsing any prediction records."""
    try:
        _require(path.is_file() and not path.is_symlink(), f"{variant}_unavailable")
        with path.open("rb") as stream:
            line = stream.readline(1024 * 1024 + 1)
    except OSError:
        raise Rejected(f"{variant}_unavailable") from None
    _require(0 < len(line) <= 1024 * 1024, f"{variant}_invalid_metadata")
    return _parse(line, variant)


def _check_coverage(cases, value, label):
    _require(isinstance(value, dict) and set(value) == set(cases), label)
    return value


def _score_summary(score):
    """Keep aggregate diagnostic output/source hashes, never case rows or text."""
    summary = {name: score[name] for name in (
        "metric_schema_version", "direction", "n", "hashes", "metrics",
        "baseline", "candidate") if name in score}
    # The scorer's top-level row table includes caller-provided example IDs.
    # Categories in the ordinary schema are codes; redact other caller labels.
    expected_categories = {
        "negation", "numerals", "entities", "technical_terms", "context",
        "slang", "noisy_input", "quantity", "units", "correction", "pronoun",
        "time", "schedule", "entity", "slang_idiom", "ordinal", "sequence",
        "left_right", "two_clauses", "all",
    }
    groups = score.get("subgroups")
    if isinstance(groups, dict):
        summary["subgroups"] = {
            (g if g in expected_categories else "group_sha256:" + _hash(g.encode("utf-8"))): stats
            for g, stats in groups.items()
        }
    return summary


def _public_gate(value):
    """Gate rules are locally recomputed; never serialize their raw inputs."""
    _require(isinstance(value, dict), "invalid_gate_result")
    # Gate result subfields contain computed numeric measures and fixed reason
    # codes. Retain the most useful stable summary; reject any unknown verdict.
    status = value.get("status")
    _require(status in {"pass", "fail", "blocked", "not_verified"}, "invalid_gate_result")
    cleaned = {"status": status, "promotion_eligible": False}
    for k in ("checks", "failures", "blockers", "summary", "independent_human_evaluation",
              "external_user_approval_required"):
        if k in value:
            cleaned[k] = _public_gate_value(value[k])
    return cleaned


def _public_gate_value(value):
    if value is None or type(value) in (bool, int, float):
        return value
    if isinstance(value, str):
        # Keep machine reason codes and checksums, never free-form input values.
        if (value == "dev_chrF_drop_exceeds_half_point"
                or re.fullmatch(r"[a-z][a-z_0-9-]{0,95}", value) or _pin(value)):
            return value
        return {"detail_sha256": _hash(value.encode("utf-8"))}
    if isinstance(value, (list, tuple)):
        return [_public_gate_value(x) for x in value]
    if isinstance(value, dict):
        return {(k if isinstance(k, str) and re.fullmatch(r"[a-z][a-z_0-9-]{0,95}", k)
                 else "key_sha256:" + _data_hash(k)):
                _public_gate_value(v) for k, v in value.items()}
    raise Rejected("invalid_gate_result")


def _gate_receipt(direction, suite_hash, arm, outputs):
    # Omit unavailable evidence; do not manufacture performance or checkpoints.
    result = {"direction": direction, "scope": "development",
              "fixture_sha256": suite_hash, "outputs": outputs}
    for key in ("checkpoint", "performance", "input_mix_sha256"):
        if key in arm:
            result[key] = arm[key]
    return result


def compare(args, manifest, cases, route_by_id, old):
    if args.m0_baseline:
        _require(args.suite_format == "m0" and args.baseline is None
                 and args.baseline_sha256 is None, "invalid_baseline_mode")
        baseline, bdetails = _m0_baseline(manifest, old)
        bfile_sha = args.suite_sha256
    else:
        _require(args.baseline is not None and _pin(args.baseline_sha256),
                 "baseline_checksum_required")
        baseline, bdetails = load_receipt(args.baseline, args.baseline_sha256, "baseline",
                                          args.suite_sha256, route_by_id)
        bfile_sha = args.baseline_sha256
    candidate, cdetails = load_receipt(args.candidate, args.candidate_sha256, "candidate",
                                       args.suite_sha256, route_by_id)
    bgroup = {d: {r["id"]: baseline[r["id"]] for r in group} for d, group in cases.items()}
    cgroup = {d: {r["id"]: candidate[r["id"]] for r in group} for d, group in cases.items()}
    observed_baseline_hashes = {d: _data_hash(output) for d, output in bgroup.items()}
    expected_baseline_hashes = None
    if args.m0_baseline:
        expected_baseline_hashes = {d: _data_hash({r["id"]: old[r["id"]] for r in group})
                                    for d, group in cases.items()}
    elif "baseline_outputs_sha256_by_direction" in manifest:
        expected_baseline_hashes = _check_coverage(cases, manifest["baseline_outputs_sha256_by_direction"],
                                                   "manifest_invalid_baseline_pins")
        _require(all(_pin(h) for h in expected_baseline_hashes.values()),
                  "manifest_invalid_baseline_pins")
    if expected_baseline_hashes is not None:
        _require(expected_baseline_hashes == observed_baseline_hashes,
                 "baseline_outputs_do_not_match_frozen_pins")

    trusted = manifest.get("checkpoint_pins")
    if trusted is not None:
        _require(isinstance(trusted, dict) and trusted.keys() == {"baseline", "candidate"},
                 "manifest_invalid_checkpoint_pins")
        for pin in trusted.values():
            _check_coverage(cases, pin, "manifest_invalid_checkpoint_pins")

    lengths = manifest.get("performance_requests_by_direction")
    if lengths is not None:
        _check_coverage(cases, lengths, "manifest_invalid_performance_protocol")
        _require(all(type(v) is int and 40 <= v <= 100_000 for v in lengths.values()),
                 "manifest_invalid_performance_protocol")

    review = None
    if args.development_review is not None or args.development_review_sha256 is not None:
        _require(args.development_review is not None and _pin(args.development_review_sha256),
                 "development_review_checksum_required")
        from .review import load_development_review
        review = load_development_review(
            args.development_review, expected_sha256=args.development_review_sha256,
            fixture_sha256=args.suite_sha256, baseline_outputs_sha256=_data_hash(baseline),
            candidate_outputs_sha256=_data_hash(candidate), case_directions=route_by_id)

    from .metrics import score_direction
    from .gates import evaluate_direction

    report = {}
    for direction in sorted(cases):
        source_language, target_language = direction.split("->")
        score = score_direction(cases[direction], bgroup[direction], cgroup[direction],
                                source_language=source_language, target_language=target_language)
        input_mix = None
        if lengths is not None:
            # A prior turn is part of the workload. Never collapse two different
            # context conditions into the same performance mix identity.
            source_hashes = [
                _data_hash({"source": row["source"], "context": row["context"]})
                if "context" in row else _hash(row["source"].encode("utf-8"))
                for row in cases[direction]]
            input_mix = _data_hash([source_hashes[i % len(source_hashes)] for i in range(lengths[direction])])
        gate = evaluate_direction(
            _gate_receipt(direction, args.suite_sha256, bdetails[direction], bgroup[direction]),
            _gate_receipt(direction, args.suite_sha256, cdetails[direction], cgroup[direction]),
            lane=args.lane, score=score, critical=review.get(direction) if review else None,
            baseline_pins=trusted["baseline"][direction] if trusted else None,
            candidate_pins=trusted["candidate"][direction] if trusted else None,
            expected_fixture_sha256=args.suite_sha256,
            expected_baseline_outputs_sha256=expected_baseline_hashes[direction] if expected_baseline_hashes else None,
            expected_input_mix_sha256=input_mix,
            expected_warm_requests=lengths[direction] if lengths else None)
        report[direction] = {
            "diagnostics": _score_summary(score), "gate": _public_gate(gate),
            "observed_exact_output_matches": sum(bgroup[direction][k] == v
                                                for k, v in cgroup[direction].items()),
            "case_count": len(cases[direction]),
        }

    statuses = {v["gate"]["status"] for v in report.values()}
    outcome = ("fail" if "fail" in statuses else "pass" if statuses == {"pass"} else "not_verified")
    result = {
        "schema_version": 1, "kind": "dev_evaluation_report", "scope": "development",
        "status": outcome, "promotion_eligible": False, "independent_final_evaluation": "unavailable",
        "sealed_holdout_opened": False, "lane": args.lane, "suite_format": args.suite_format,
        "coverage_scope": "only_explicit_fixture_directions",
        "pins": {"manifest_sha256": args.manifest_sha256, "fixture_sha256": args.suite_sha256,
                 "baseline_receipt_sha256": bfile_sha, "candidate_receipt_sha256": args.candidate_sha256,
                 "baseline_outputs_sha256": _data_hash(baseline), "candidate_outputs_sha256": _data_hash(candidate)},
        "directions": report,
    }
    if review is not None:
        result["pins"]["development_review_sha256"] = args.development_review_sha256
    return result, 0 if outcome == "pass" else 3


def parser():
    root = RedactedParser(description="Offline BridgeCaption development diagnostics; never a release gate")
    commands = root.add_subparsers(dest="command", required=True, parser_class=RedactedParser)
    for command in ("preflight", "compare"):
        cli = commands.add_parser(command)
        cli.add_argument("--manifest", required=True, type=Path)
        cli.add_argument("--manifest-sha256", required=True)
        cli.add_argument("--suite", required=True, type=Path)
        cli.add_argument("--suite-sha256", required=True)
        cli.add_argument("--suite-format", choices=("generic", "m0"), default="generic")
        if command == "compare":
            cli.add_argument("--baseline", type=Path)
            cli.add_argument("--baseline-sha256")
            cli.add_argument("--m0-baseline", action="store_true")
            cli.add_argument("--candidate", required=True, type=Path)
            cli.add_argument("--candidate-sha256", required=True)
            cli.add_argument("--lane", choices=("infrastructure", "model_dev"), required=True)
            cli.add_argument("--development-review", type=Path)
            cli.add_argument("--development-review-sha256")
    return root


def main(argv=None):
    try:
        args = parser().parse_args(argv)
        manifest, cases, route_by_id, old = load_suite(args)
        if args.command == "preflight":
            status, result = 0, {
                "schema_version": 1, "kind": "dev_input_preflight", "scope": "development",
                "status": "verified_inputs_only", "promotion_eligible": False,
                "sealed_holdout_opened": False, "fixture_sha256": args.suite_sha256,
                "manifest_sha256": args.manifest_sha256, "suite_format": args.suite_format,
                "coverage_scope": "only_explicit_fixture_directions",
                "directions": {d: len(v) for d, v in sorted(cases.items())},
            }
        else:
            result, status = compare(args, manifest, cases, route_by_id, old)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2))
        return status
    except Rejected as exc:
        code = exc.code
    except ImportError:
        code = "evaluation_dependency_unavailable"
    except (ValueError, TypeError, KeyError, OSError, UnicodeError, RecursionError):
        # Libraries may include raw outputs, local paths or input rows in messages.
        code = "invalid_evaluation_evidence"
    except Exception:
        code = "evaluation_unavailable"
    print(json.dumps({"status": "rejected", "code": code, "promotion_eligible": False}),
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
