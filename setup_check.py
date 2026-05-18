"""
Setup Check — Verify R9700 gfx1201 environment is correctly configured
=======================================================================
Checks Python, PyTorch, ROCm, GPU, env vars, VRAM, and all dependencies.
Prints PASS/FAIL for each check with exact fix instructions.

Usage: python setup_check.py
"""

# ROCm setup first
import rocm_setup
rocm_setup.configure()

import os
import sys
import importlib
import platform


def colorize(text, color):
    """Add color using colorama if available, fallback to plain text."""
    try:
        from colorama import Fore, Style, init
        init(autoreset=True)
        colors = {
            "green": Fore.GREEN,
            "red": Fore.RED,
            "yellow": Fore.YELLOW,
            "cyan": Fore.CYAN,
            "white": Fore.WHITE,
        }
        return colors.get(color, "") + text + Style.RESET_ALL
    except ImportError:
        return text


def check_pass(label):
    return colorize(f"  [PASS] {label}", "green")


def check_fail(label):
    return colorize(f"  [FAIL] {label}", "red")


def check_warn(label):
    return colorize(f"  [WARN] {label}", "yellow")


def check_info(label):
    return colorize(f"  [INFO] {label}", "cyan")


def main():
    print()
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║          SETUP CHECK — AMD Radeon AI PRO R9700 gfx1201          ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print()

    all_passed = True

    # ── 1. Python Version ──────────────────────────────────────────────────
    py_ver = platform.python_version()
    py_major, py_minor = sys.version_info.major, sys.version_info.minor
    if py_major == 3 and py_minor in (11, 12):
        print(check_pass(f"Python {py_ver}"))
    else:
        print(check_fail(f"Python {py_ver} — requires 3.11 or 3.12"))
        print(f"         Fix: Install Python 3.11 or 3.12 from python.org")
        all_passed = False

    # ── 2. PyTorch installed ───────────────────────────────────────────────
    try:
        import torch
        print(check_pass(f"PyTorch {torch.__version__}"))
    except ImportError:
        print(check_fail("PyTorch not installed"))
        print(f"         Fix: python install.py")
        all_passed = False
        # Can't continue without torch
        print()
        print(colorize("  Cannot continue without PyTorch. Run: python install.py", "red"))
        sys.exit(1)

    # ── 3. ROCm/CUDA detection ─────────────────────────────────────────────
    if torch.cuda.is_available():
        print(check_pass("torch.cuda.is_available() = True (ROCm detected)"))
    else:
        print(check_fail("torch.cuda.is_available() = False"))
        print(f"         Fix: Ensure AMD Adrenalin 25.x+ driver is installed")
        print(f"              Ensure PyTorch ROCm 7.x wheels are installed: python install.py")
        print(f"              Restart terminal/VS Code after driver install")
        all_passed = False

    # ── 4. GPU name ────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        if "r9700" in gpu_name.lower() or "radeon" in gpu_name.lower():
            print(check_pass(f"GPU detected: {gpu_name}"))
        else:
            print(check_warn(f"GPU detected: {gpu_name} (expected AMD Radeon AI PRO R9700)"))
    else:
        print(check_fail("GPU not detected — cannot check name"))
        all_passed = False

    # ── 5. GFX architecture ────────────────────────────────────────────────
    if torch.cuda.is_available():
        gfx_arch = "unknown"
        try:
            props = torch.cuda.get_device_properties(0)
            if hasattr(props, "gcnArchName"):
                gfx_arch = props.gcnArchName
        except Exception:
            pass

        if "gfx1201" in gfx_arch.lower():
            print(check_pass(f"GFX architecture: {gfx_arch}"))
        elif gfx_arch == "unknown":
            print(check_warn(f"GFX architecture: could not detect (this may be normal)"))
        else:
            print(check_warn(f"GFX architecture: {gfx_arch} (expected gfx1201)"))

    # ── 6. ROCm version ───────────────────────────────────────────────────
    if torch.cuda.is_available():
        rocm_ver = getattr(torch.version, "hip", None) or "unknown"
        if rocm_ver != "unknown" and rocm_ver.startswith("7"):
            print(check_pass(f"ROCm version: {rocm_ver}"))
        elif rocm_ver != "unknown" and rocm_ver.startswith("6"):
            print(check_fail(f"ROCm version: {rocm_ver} — gfx1201 requires ROCm 7.x"))
            print(f"         Fix: gfx1201 (RDNA 4) is NOT supported in ROCm 6.x at all")
            print(f"              Install ROCm 7.x PyTorch wheels: python install.py")
            all_passed = False
        else:
            print(check_info(f"ROCm/HIP version: {rocm_ver}"))

    # ── 7. HSA_OVERRIDE_GFX_VERSION ────────────────────────────────────────
    hsa = os.environ.get("HSA_OVERRIDE_GFX_VERSION", "")
    if hsa == "12.0.1":
        print(check_pass(f"HSA_OVERRIDE_GFX_VERSION = {hsa}"))
    else:
        print(check_fail(f"HSA_OVERRIDE_GFX_VERSION = '{hsa}' (expected '12.0.1')"))
        print(f"         Fix: This should be set automatically by rocm_setup.py")
        print(f"              Or set system env var: HSA_OVERRIDE_GFX_VERSION=12.0.1")
        all_passed = False

    # ── 8. PYTORCH_ROCM_ARCH ──────────────────────────────────────────────
    rocm_arch = os.environ.get("PYTORCH_ROCM_ARCH", "")
    if rocm_arch == "gfx1201":
        print(check_pass(f"PYTORCH_ROCM_ARCH = {rocm_arch}"))
    else:
        print(check_fail(f"PYTORCH_ROCM_ARCH = '{rocm_arch}' (expected 'gfx1201')"))
        print(f"         Fix: This should be set automatically by rocm_setup.py")
        print(f"              Or set system env var: PYTORCH_ROCM_ARCH=gfx1201")
        all_passed = False

    # ── 9. VRAM ────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        if vram_gb >= 28:
            print(check_pass(f"VRAM: {vram_gb:.1f} GB"))
        else:
            print(check_warn(f"VRAM: {vram_gb:.1f} GB (expected ~32 GB for R9700)"))

    # ── 10. Python packages ────────────────────────────────────────────────
    print()
    print("  Package checks:")

    packages = {
        "transformers": "4.45.0",
        "peft": "0.13.0",
        "trl": "0.11.0",
        "bitsandbytes": "0.44.0",
        "accelerate": "0.34.0",
        "datasets": "3.0.0",
        "pypdf": "4.0.0",
        "dotenv": None,  # python-dotenv imports as dotenv
        "colorama": "0.4.6",
        "huggingface_hub": "0.25.0",
    }

    # Map import names to pip names
    pip_names = {
        "dotenv": "python-dotenv",
        "huggingface_hub": "huggingface-hub",
    }

    for pkg, min_ver in packages.items():
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "unknown")
            pip_name = pip_names.get(pkg, pkg)
            print(check_pass(f"  {pip_name} {ver}"))
        except ImportError:
            pip_name = pip_names.get(pkg, pkg)
            fix_ver = f">={min_ver}" if min_ver else ""
            print(check_fail(f"  {pip_name} not installed"))
            print(f"           Fix: pip install {pip_name}{fix_ver}")
            all_passed = False

    # ── 11. numpy version ──────────────────────────────────────────────────
    try:
        import numpy as np
        if np.__version__ == "1.26.4":
            print(check_pass(f"  numpy {np.__version__} (pinned)"))
        else:
            print(check_warn(f"  numpy {np.__version__} (expected 1.26.4)"))
            print(f"           Fix: pip install numpy==1.26.4")
    except ImportError:
        print(check_fail("  numpy not installed"))
        print(f"           Fix: pip install numpy==1.26.4")
        all_passed = False

    # ── 12. /train folder ──────────────────────────────────────────────────
    print()
    train_dir = os.path.abspath("./train")
    if os.path.isdir(train_dir):
        files = [f for f in os.listdir(train_dir)
                 if f.lower().endswith((".txt", ".pdf"))]
        if files:
            print(check_pass(f"/train folder exists with {len(files)} training file(s)"))
        else:
            print(check_warn(f"/train folder exists but has no .txt or .pdf files"))
            print(f"         Add your training files to: {train_dir}")
    else:
        print(check_warn(f"/train folder not found"))
        print(f"         It will be auto-created when you run train.py")

    # ── 13. No HF token required ──────────────────────────────────────
    print(check_pass("HF token: not required (all models are Apache 2.0 / MIT)"))

    # ── Summary ────────────────────────────────────────────────────────────
    print()
    print("─" * 65)
    if all_passed:
        print(colorize("  ALL CHECKS PASSED — Ready to train!", "green"))
        print(colorize("  Run: python train.py", "green"))
    else:
        print(colorize("  SOME CHECKS FAILED — Fix the issues above before training.", "red"))
    print("─" * 65)
    print()


if __name__ == "__main__":
    main()
