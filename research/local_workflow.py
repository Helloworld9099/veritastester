"""Run BridgeCaption's evidence workflow locally, without a service or trainer.

The explicit plan is independently pinned. It names data, checkpoint and
evaluation evidence, not arbitrary commands. The local ledger retains only
aggregate status and digests; it is not a checkpoint or dataset archive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

from . import workflow_contract, workflow_data
from .workflow_store import DuplicatePlanError, WorkflowStore

_PLAN_LIMIT = 512 * 1024
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_PHASES = ("data", "checkpoint", "comparison")


class WorkflowError(ValueError):
    """A local evidence workflow cannot proceed."""


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def _sha(value):
    return isinstance(value, str) and _SHA.fullmatch(value) is not None


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise WorkflowError("Duplicate JSON plan field")
        result[key] = value
    return result


def _reject_constant(_):
    raise WorkflowError("Invalid JSON constant in plan")


def _plan(path, expected):
    if not _sha(expected) or path.is_symlink() or not path.is_file():
        raise WorkflowError("A locally pinned regular plan is required")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise WorkflowError("The plan must be a regular file")
        raw = stream.read(_PLAN_LIMIT + 1)
    if len(raw) > _PLAN_LIMIT or hashlib.sha256(raw).hexdigest() != expected:
        raise WorkflowError("Plan pin validation failed")
    plan = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique, parse_constant=_reject_constant)
    if (type(plan) is not dict or set(plan) != {"schema_version", "kind", *_PHASES}
            or type(plan["schema_version"]) is not int or plan["schema_version"] != 1
            or plan["kind"] != "local_research_workflow"
            or any(plan[k] is not None and type(plan[k]) is not dict for k in _PHASES)):
        raise WorkflowError("Unexpected plan schema")
    comp = plan["comparison"]
    if comp is not None:
        if not {"baseline", "candidate"} <= set(comp) <= {"baseline", "candidate", "lane"}:
            raise WorkflowError("Unexpected comparison schema")
        if comp.get("lane", "infrastructure") not in ("infrastructure", "model_dev"):
            raise WorkflowError("Unknown comparison lane")
        for arm in ("baseline", "candidate"):
            ref = comp[arm]
            if (type(ref) is not dict or ref.keys() != {"path", "sha256"}
                    or not isinstance(ref["path"], str) or not _sha(ref["sha256"])):
                raise WorkflowError("Comparison must use explicit pinned input receipts")
    return plan


def _separate(store, input_root, plan_path):
    if input_root.is_symlink() or not input_root.is_dir():
        raise WorkflowError("Input root must be a local directory")
    root, output = input_root.resolve(), store.resolve()
    repo = Path(__file__).resolve().parents[1]
    if any(p.is_relative_to(output) or output.is_relative_to(p) for p in (root, repo)):
        raise WorkflowError("Put the ledger outside the checkout and input root")
    if plan_path.resolve().is_relative_to(output):
        raise WorkflowError("Put the plan outside the ledger")
    return root


def _empty():
    return {"status": "not_verified", "digest": None}


def _phase_summary(status, evidence):
    if status not in ("pass", "fail", "not_verified"):
        raise WorkflowError("Unexpected phase disposition")
    return {"status": status, "digest": _digest(evidence)}


def _comparison(root, spec):
    arms = [workflow_contract.verify_receipt(
        root, spec[side]["path"], expected_sha256=spec[side]["sha256"],
    ) for side in ("baseline", "candidate")]
    return workflow_contract.compare_receipts(*arms, lane=spec.get("lane", "infrastructure")), arms[1]


def _audit(store, run_id):
    snapshot = store.audit(run_id)
    return {
        "state": snapshot.state,
        "status": snapshot.status or "not_verified",
        "plan_sha256": snapshot.plan_sha256,
        "receipt_sha256": snapshot.receipt_sha256,
        "receipt": snapshot.receipt,
        "training_dispatched": False,
        "promotion_eligible": False,
    }


def execute(args):
    plan = _plan(args.plan, args.plan_sha256)
    root = _separate(args.store, args.input_root, args.plan)
    store = WorkflowStore(args.store)
    envelope = {"schema_version": 1, "kind": "local_research_workflow",
                "plan_sha256": args.plan_sha256}
    reservation = store.reserve(args.run_id, plan=envelope)
    if not reservation.created:
        result = _audit(store, args.run_id)
        return result, 0 if result["status"] == "pass" else 3
    phases = {p: _empty() for p in _PHASES}
    if plan["data"] is not None:
        evidence = workflow_data.prepare(plan["data"], root=root)
        if evidence["status"] != "verified":
            raise WorkflowError("Data preparation did not verify")
        phases["data"] = _phase_summary("pass", evidence)
    if plan["checkpoint"] is not None:
        evidence = workflow_contract.verify_checkpoint(root, plan["checkpoint"])
        if evidence["status"] != "verified_pinned_bytes":
            raise WorkflowError("Checkpoint pin did not verify")
        phases["checkpoint"] = _phase_summary("pass", evidence)
    if plan["comparison"] is not None:
        spec = plan["comparison"]
        evidence, candidate = _comparison(root, spec)
        # Bind a positive comparison to the same candidate whose local bytes
        # were verified, rather than passing because an unrelated file matched.
        status = "fail" if evidence["status"] == "fail" else "not_verified"
        if evidence["status"] == "reported_exact_match":
            checkpoint = plan["checkpoint"]
            facts = candidate.to_public()
            if (checkpoint is not None and facts["checkpoint_sha256"] == checkpoint["checkpoint_sha256"]
                    and facts["direction"] == checkpoint["direction"]):
                status = "pass"
        phases["comparison"] = _phase_summary(status, evidence)
    statuses = {p["status"] for p in phases.values()}
    outcome = "fail" if "fail" in statuses else "pass" if statuses == {"pass"} else "not_verified"
    receipt = {
        "schema_version": 1, "kind": "local_research_workflow_receipt",
        "plan_sha256": args.plan_sha256, "status": outcome, "phases": phases,
    }
    store.complete(reservation, receipt=receipt)
    return _audit(store, args.run_id), 0 if outcome == "pass" else 3


class _Parser(argparse.ArgumentParser):
    def error(self, _message):
        raise WorkflowError("Invalid local workflow arguments")


def main(argv=None):
    root = _Parser(description=__doc__)
    commands = root.add_subparsers(dest="operation", required=True, parser_class=_Parser)
    for op in ("run", "audit"):
        cmd = commands.add_parser(op)
        cmd.add_argument("--store", type=Path, required=True)
        cmd.add_argument("--run-id", required=True)
        if op == "run":
            cmd.add_argument("--plan", type=Path, required=True)
            cmd.add_argument("--plan-sha256", required=True)
            cmd.add_argument("--input-root", type=Path, required=True)
    try:
        args = root.parse_args(argv)
        if args.operation == "run":
            result, code = execute(args)
        else:
            result = _audit(WorkflowStore(args.store), args.run_id)
            code = 0 if result["state"] == "completed" else 3
    except DuplicatePlanError:
        print(json.dumps({"status": "not_verified", "code": "identical_plan_already_reserved",
                          "training_dispatched": False}))
        return 3
    except (TypeError, ValueError, KeyError, OSError, RecursionError):
        print(json.dumps({"status": "rejected", "code": "invalid_or_unverified_local_evidence",
                          "training_dispatched": False}), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
