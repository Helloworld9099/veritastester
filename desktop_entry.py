"""Executable entry point for frozen desktop distributions."""
import sys
from pathlib import Path

if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--version-file":
        # Automation can inspect frozen Windows GUI binaries without a stdout console.
        from bridgecaption.updates import current_version
        Path(sys.argv[2]).write_text(current_version() + "\n", encoding="utf-8")
    else:
        from bridgecaption.app import main
        main()
