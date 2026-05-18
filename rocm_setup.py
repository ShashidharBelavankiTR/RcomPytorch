"""
ROCm Environment Setup for AMD Radeon AI PRO R9700 (gfx1201)
=============================================================
THIS FILE MUST BE IMPORTED BEFORE ANY OTHER IMPORT IN EVERY SCRIPT.
It sets all required environment variables for gfx1201 on ROCm 7.x.

Usage:
    import rocm_setup
    rocm_setup.configure()

    # Now safe to import torch, transformers, etc.
"""

import os
import sys


def configure():
    """
    Set all environment variables required for AMD Radeon AI PRO R9700 (gfx1201)
    on ROCm 7.x BEFORE torch is imported. Call this at the top of every script.
    """

    # ── GFX Architecture Override ──────────────────────────────────────────────
    # gfx1201 (RDNA 4) requires this exact override for PyTorch ROCm 7.x
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "12.0.1"

    # ── PyTorch ROCm Architecture Target ───────────────────────────────────────
    os.environ["PYTORCH_ROCM_ARCH"] = "gfx1201"

    # ── GPU Device Visibility ──────────────────────────────────────────────────
    # Use GPU 0 (single-GPU setup)
    os.environ["HIP_VISIBLE_DEVICES"] = "0"
    os.environ["ROCR_VISIBLE_DEVICES"] = "0"

    # ── Performance Tuning ─────────────────────────────────────────────────────
    # Maximum hardware queues for best throughput on R9700
    os.environ["GPU_MAX_HW_QUEUES"] = "8"

    # Enable PyTorch tunable operations for ROCm kernel auto-tuning
    os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "1"

    # ── Suppress Warnings ──────────────────────────────────────────────────────
    # Prevent tokenizer parallelism deadlocks
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Suppress non-critical HuggingFace warnings
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"


def verify():
    """
    Verify that the ROCm environment is correctly configured and the R9700 is detected.
    Call this AFTER configure() and AFTER importing torch.
    """
    try:
        import torch
    except ImportError:
        print("\n[ERROR] PyTorch is not installed.")
        print("  Fix: python install.py")
        sys.exit(1)

    print("=" * 65)
    print("  AMD Radeon AI PRO R9700 — ROCm 7.x Environment Check")
    print("=" * 65)

    # Check torch.cuda (ROCm uses the CUDA API surface in PyTorch)
    if not torch.cuda.is_available():
        print("\n[FAILED] torch.cuda.is_available() returned False")
        print()
        print("  This means PyTorch cannot see your R9700 GPU.")
        print()
        print("  Fixes for gfx1201 on Windows 11:")
        print("  1. Ensure AMD Adrenalin driver 25.x or later is installed")
        print("  2. Ensure PyTorch was installed from ROCm 7.x wheels:")
        print("     python install.py")
        print("  3. Verify environment variables are set:")
        print("     HSA_OVERRIDE_GFX_VERSION=12.0.1")
        print("     PYTORCH_ROCM_ARCH=gfx1201")
        print("  4. Restart your terminal/VS Code after driver install")
        print("  5. gfx1201 does NOT work on ROCm 6.x — ROCm 7.x is required")
        print()
        sys.exit(1)

    # GPU info
    gpu_name = torch.cuda.get_device_name(0)
    vram_bytes = torch.cuda.get_device_properties(0).total_mem
    vram_gb = vram_bytes / (1024 ** 3)

    # Try to get gfx architecture
    gfx_arch = "unknown"
    try:
        props = torch.cuda.get_device_properties(0)
        if hasattr(props, "gcnArchName"):
            gfx_arch = props.gcnArchName
        elif hasattr(props, "name"):
            gfx_arch = f"(detected via name: {props.name})"
    except Exception:
        pass

    # ROCm version
    rocm_version = "unknown"
    if hasattr(torch.version, "hip"):
        rocm_version = torch.version.hip or "unknown"

    print(f"\n  GPU Name:      {gpu_name}")
    print(f"  VRAM:          {vram_gb:.1f} GB")
    print(f"  GFX Arch:      {gfx_arch}")
    print(f"  ROCm/HIP:      {rocm_version}")
    print(f"  PyTorch:       {torch.__version__}")
    print(f"  HSA_OVERRIDE:  {os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'NOT SET')}")
    print(f"  ROCM_ARCH:     {os.environ.get('PYTORCH_ROCM_ARCH', 'NOT SET')}")

    print()
    print("  [OK] R9700 gfx1201 ROCm 7.x environment configured successfully")
    print("=" * 65)
    print()


# Auto-configure on import so every file that does `import rocm_setup`
# gets the env vars set immediately, before they import torch.
configure()


if __name__ == "__main__":
    verify()
