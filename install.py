"""
Install — Install correct ROCm 7.x PyTorch wheels for R9700 gfx1201
=====================================================================
Installs PyTorch from the official ROCm 7.x wheel repository, then
installs all Python dependencies from requirements.txt.

Usage: python install.py
"""

import os
import sys
import subprocess


def main():
    print()
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║     INSTALL — AMD Radeon AI PRO R9700 gfx1201 — ROCm 7.x       ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print()
    print("  This script will install:")
    print("    1. PyTorch + torchvision + torchaudio from ROCm 7.x wheels")
    print("    2. All Python dependencies from requirements.txt")
    print()
    print("  ┌───────────────────────────────────────────────────────────────┐")
    print("  │  IMPORTANT: gfx1201 (RDNA 4) requires ROCm 7.x              │")
    print("  │  ROCm 6.x does NOT support gfx1201 at all.                  │")
    print("  │  PyTorch wheels are sourced from repo.radeon.com             │")
    print("  └───────────────────────────────────────────────────────────────┘")
    print()

    # ── PyTorch ROCm 7.x install command ───────────────────────────────────
    # This URL points to the official AMD ROCm 7.x wheel repository
    rocm_index_url = "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0/"

    pytorch_cmd = [
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", rocm_index_url,
    ]

    # On Windows, AMD also provides wheels via this URL pattern
    # If the Linux URL doesn't work, try the Windows-specific one
    pytorch_cmd_win = [
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0/",
        "--extra-index-url", "https://download.pytorch.org/whl/rocm6.2",
    ]

    print("  Step 1: Install PyTorch with ROCm 7.x support")
    print()
    print(f"  Command:")
    print(f"    pip install torch torchvision torchaudio \\")
    print(f"      --index-url {rocm_index_url}")
    print()

    # Alternative manual install instructions
    print("  ─────────────────────────────────────────────────────────────")
    print("  If the automatic install fails, install manually:")
    print()
    print("  Option A — AMD official ROCm 7.x wheels:")
    print(f"    pip install torch torchvision torchaudio --index-url {rocm_index_url}")
    print()
    print("  Option B — PyTorch nightly with ROCm (if stable isn't available yet):")
    print("    pip install --pre torch torchvision torchaudio \\")
    print("      --index-url https://download.pytorch.org/whl/nightly/rocm7.0")
    print()
    print("  Option C — Check AMD's official guide:")
    print("    https://rocm.docs.amd.com/projects/install-on-windows/en/latest/")
    print("  ─────────────────────────────────────────────────────────────")
    print()

    response = input("  Proceed with automatic install? [Y/n]: ").strip().lower()
    if response in ("n", "no"):
        print("  Skipped PyTorch install. Install manually using commands above.")
    else:
        print("\n  Installing PyTorch ROCm 7.x wheels...")
        result = subprocess.run(pytorch_cmd, capture_output=False)
        if result.returncode != 0:
            print()
            print("  [WARNING] Primary install URL may have failed.")
            print("  Trying alternative URL...")
            result2 = subprocess.run(pytorch_cmd_win, capture_output=False)
            if result2.returncode != 0:
                print()
                print("  [ERROR] Automatic PyTorch install failed.")
                print("  Please install manually using the commands shown above.")
                print("  Then re-run: python install.py")
                print()

    # ── Requirements install ───────────────────────────────────────────────
    print()
    print("  Step 2: Install Python dependencies from requirements.txt")
    print()

    req_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    if not os.path.exists(req_file):
        print(f"  [ERROR] requirements.txt not found at: {req_file}")
        sys.exit(1)

    req_cmd = [sys.executable, "-m", "pip", "install", "-r", req_file]
    print(f"  Command: pip install -r requirements.txt")
    print()

    response = input("  Proceed with dependency install? [Y/n]: ").strip().lower()
    if response in ("n", "no"):
        print("  Skipped. Install manually: pip install -r requirements.txt")
    else:
        print("\n  Installing dependencies...")
        subprocess.run(req_cmd, capture_output=False)

    # ── Run setup check ────────────────────────────────────────────────────
    print()
    print("  Step 3: Running setup check...")
    print()

    check_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "setup_check.py")
    subprocess.run([sys.executable, check_script], capture_output=False)


if __name__ == "__main__":
    main()
