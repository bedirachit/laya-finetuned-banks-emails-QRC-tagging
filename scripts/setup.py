"""Install the CPU build of PyTorch, then Laya, into the current Python.

Create and activate your own virtualenv first, then from the project folder:

    python scripts/setup.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
PYTHON = sys.executable


def run(cmd: list[str], capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=capture)


def pip(*args: str) -> None:
    result = run([PYTHON, "-m", "pip", *args])
    if result.returncode != 0:
        raise SystemExit(f"pip failed: {' '.join(args)}")


def torch_cuda_version() -> str | None:
    result = run([PYTHON, "-c", "import torch; print(torch.version.cuda)"], capture=True)
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        raise SystemExit(
            "PyTorch did not import. On Windows this usually means this Python is Anaconda, "
            "whose Visual C++ runtime is too old for current PyTorch (WinError 1114 on c10.dll).\n"
            "Use a Python from https://www.python.org/downloads/ and run this script again."
        )
    value = (result.stdout or "").strip()
    if value in ("", "None"):
        return None
    return value


def main() -> None:
    if sys.version_info < (3, 10):
        raise SystemExit("Python 3.10 or newer is required.")

    print(f"Installing into {PYTHON}", flush=True)
    print("Installing the CPU build of PyTorch", flush=True)
    pip("install", "--upgrade", "pip")
    pip("install", "torch", "--index-url", CPU_INDEX)

    print("Installing Laya", flush=True)
    pip("install", "-r", str(ROOT / "requirements.txt"))

    if torch_cuda_version() is not None:
        print("Laya replaced the CPU build of PyTorch. Installing it again.", flush=True)
        pip("install", "torch", "--index-url", CPU_INDEX, "--force-reinstall")
        if torch_cuda_version() is not None:
            raise SystemExit("The CPU build of PyTorch is still not the one that imports.")

    check = run(
        [
            PYTHON,
            "-c",
            "import torch, laya; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('laya', laya.__version__)",
        ]
    )
    if check.returncode != 0:
        raise SystemExit("Laya did not import.")

    print()
    print("Setup finished. Run the sample:")
    print("  python scripts/first_decision.py")


if __name__ == "__main__":
    main()
