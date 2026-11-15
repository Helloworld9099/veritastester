"""Non-network foundation checks. Passing is necessary, not sufficient, for training."""
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def inspect(root: Path):
    checks = {}
    checks["python_3_10_plus"] = sys.version_info >= (3, 10)
    checks["ffmpeg"] = bool(shutil.which("ffmpeg"))
    checks["pactl"] = bool(shutil.which("pactl"))
    for name in ("tkinter", "numpy", "faster_whisper", "dotenv"):
        checks["import_" + name] = importlib.util.find_spec(name) is not None
    # Never echo credentials. Model access is NOT established by key presence.
    checks["gemini_key_configured"] = bool(os.getenv("GEMINI_API_KEY"))
    checks["source_tree_exists"] = (root / "src" / "bridgecaption" / "app.py").is_file()
    checks["test_tree_exists"] = (root / "tests" / "test_session.py").is_file()
    subprocess_checks = {}
    if checks["source_tree_exists"] and checks["test_tree_exists"]:
        env = os.environ.copy()
        env.pop("GEMINI_API_KEY", None)  # No accidental paid/network API calls from tests.
        for name, command in {
            "compile": [sys.executable, "-m", "compileall", "-q", "src", "tests"],
            "unit_tests": [sys.executable, "-m", "pytest", "-q"],
        }.items():
            try:
                process = subprocess.run(command, cwd=root, env=env,
                                         capture_output=True, text=True, timeout=120)
                subprocess_checks[name] = {"passed": process.returncode == 0,
                                           "summary": (process.stdout or process.stderr)[-400:]}
            except (OSError, subprocess.TimeoutExpired) as exc:
                subprocess_checks[name] = {"passed": False, "summary": type(exc).__name__}
    offline_pass = (all(subprocess_checks.get(k, {}).get("passed", False) for k in ("compile", "unit_tests"))
                    and checks["source_tree_exists"] and checks["test_tree_exists"])
    return {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "repository_root": str(root), "offline_code_checks_pass": offline_pass,
            "environment": checks, "commands": subprocess_checks,
            "unverified_manual_gates": [
                "Real audio monitor/microphone capture and stop/restart",
                "Downloaded Whisper model, speech quality and sustained latency",
                "Authorized live Gemini translation using the configured account",
                "Overlay controls over an actual video/meeting window",
                "Privacy, consent and error recovery on the user's workstation",
                "Held-out language/accent and context-vs-baseline evaluation",
            ],
            "foundation_fully_verified": False,
            "training_started": False}


def main():
    parser = argparse.ArgumentParser(description="Check local code readiness without contacting providers")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, default=Path("results/foundation_preflight.json"))
    args = parser.parse_args()
    report = inspect(args.root.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["offline_code_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
