"""Safe, synthetic provider connectivity test. Never prints credentials or meeting text."""
from __future__ import annotations

import argparse
import json
import time

from .provider import TranslationRequest
from .providers import PROVIDERS, create_translator, key_available, needs_cloud, translation_text


def check_provider(engine: str, *, factory=create_translator):
    if engine not in PROVIDERS:
        raise ValueError(f"Unsupported engine: {engine}")
    if not key_available(engine):
        raise RuntimeError(f"Missing {engine} credential. Configure the .env file.")
    # Public synthetic sample: no private text or meeting transcript is transmitted.
    request = TranslationRequest(
        text="Hola, buenos días.", source="es", target="en", context_enabled=False)
    start = time.monotonic()
    response = factory(engine).translate(request)
    response = translation_text(response)
    return {"engine": engine, "destination": "cloud" if needs_cloud(engine) else "local",
            "elapsed_seconds": round(time.monotonic() - start, 3),
            "result": response.strip()[:160], "synthetic_only": True}


def main():
    from dotenv import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser(description="Check one translator with a public example")
    parser.add_argument("--engine", required=True, choices=PROVIDERS)
    parser.add_argument("--execute", action="store_true", help="Allow one real provider request")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable report")
    args = parser.parse_args()
    if not args.execute:
        print(f"DRY RUN: {args.engine}; supply --execute to test an actual provider.")
        return
    try:
        report = check_provider(args.engine)
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print(f"PASS · {report['engine'].capitalize()} connection works")
            print(f"Destination: {report['destination']}")
            print(f"Sample: Hola, buenos días. → {report['result']}")
            print(f"Response time: {report['elapsed_seconds']:.2f} seconds")
            print("This checks one public sample, not real-time audio or translation accuracy.")
    except (RuntimeError, ValueError) as error:
        # Error messages from providers must not contain keys or private transcript.
        parser.exit(1, f"Provider check failed: {error}\n")


if __name__ == "__main__":
    main()
