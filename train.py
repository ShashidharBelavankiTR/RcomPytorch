"""
Train — Main entry point: python train.py
==========================================
This is the ONLY command you need to run. It handles everything:
1. Configures ROCm environment for gfx1201
2. Chunks your training data
3. Loads the model with LoRA
4. Trains and saves the adapter

Usage: python train.py
"""

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  ROCm setup MUST come before ANY torch import                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
import rocm_setup
rocm_setup.configure()

import os
import gc
import sys
import time
import json

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

import config
from model_registry import get_target_modules, get_chat_template, print_model_info
from chunk import run_chunking


def print_banner():
    """Print startup banner with GPU and config info."""
    print()
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║           MY AI TRAINER — LoRA Fine-Tuning Framework            ║")
    print("║         AMD Radeon AI PRO R9700 32GB · gfx1201 · ROCm 7.x      ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print()


def print_gpu_info():
    """Print GPU details."""
    if not torch.cuda.is_available():
        print("[FATAL] GPU not detected. Run: python setup_check.py")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
    rocm_ver = getattr(torch.version, "hip", "unknown") or "unknown"

    print(f"  GPU:           {gpu_name}")
    print(f"  VRAM:          {vram_gb:.1f} GB")
    print(f"  ROCm/HIP:      {rocm_ver}")
    print(f"  PyTorch:       {torch.__version__}")
    print(f"  HSA Override:  {os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'NOT SET')}")
    print(f"  ROCM Arch:     {os.environ.get('PYTORCH_ROCM_ARCH', 'NOT SET')}")
    print()


def print_config_summary():
    """Print training configuration summary."""
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print(f"  │  Model:             {config.MODEL_NAME:<40}│")
    print(f"  │  LoRA Rank:         {config.LORA_RANK:<40}│")
    print(f"  │  LoRA Alpha:        {config.LORA_ALPHA:<40}│")
    print(f"  │  Epochs:            {config.EPOCHS:<40}│")
    print(f"  │  Batch Size:        {config.BATCH_SIZE:<40}│")
    print(f"  │  Grad Accum Steps:  {config.GRAD_ACCUM_STEPS:<40}│")
    eff_batch = config.BATCH_SIZE * config.GRAD_ACCUM_STEPS
    print(f"  │  Effective Batch:   {eff_batch:<40}│")
    print(f"  │  Learning Rate:     {config.LEARNING_RATE:<40}│")
    print(f"  │  Max Seq Length:    {config.MAX_SEQ_LENGTH:<40}│")
    print(f"  │  4-bit Quant:       {str(config.USE_4BIT_QUANTIZATION):<40}│")
    print(f"  │  Grad Checkpoint:   {str(config.USE_GRADIENT_CHECKPOINTING):<40}│")
    print(f"  │  Output Dir:        {config.OUTPUT_DIR:<40}│")
    print("  └─────────────────────────────────────────────────────────────┘")
    print()

    # Model family info
    print_model_info(config.MODEL_NAME)
    print()


def prepare_data():
    """Run chunking or reuse existing JSONL."""
    data_path = os.path.join(config.DATA_DIR, "train.jsonl")

    if os.path.exists(data_path):
        # Count existing examples
        with open(data_path, "r", encoding="utf-8") as f:
            count = sum(1 for _ in f)
        print(f"  Found existing dataset: {data_path} ({count} examples)")
        response = input("  Reuse existing data? [Y/n]: ").strip().lower()
        if response in ("n", "no"):
            print("  Re-chunking training data...")
            return run_chunking()
        else:
            print("  Using existing dataset.")
            return data_path
    else:
        return run_chunking()


def main():
    print_banner()
    print_gpu_info()
    print_config_summary()

    # ── Create directories ─────────────────────────────────────────────────
    os.makedirs(config.TRAIN_DIR, exist_ok=True)
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    # ── Step 1: Prepare training data ──────────────────────────────────────
    print("━" * 65)
    print("  Step 1: Preparing Training Data")
    print("━" * 65)
    data_path = prepare_data()
    print()

    # ── Step 2: Load tokenizer ─────────────────────────────────────────────
    print("━" * 65)
    print("  Step 2: Loading Tokenizer")
    print("━" * 65)
    print(f"  Model: {config.MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME,
        trust_remote_code=True,
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Pad token: {tokenizer.pad_token}")
    print()

    # ── Step 3: Configure quantization ─────────────────────────────────────
    print("━" * 65)
    print("  Step 3: Loading Model")
    print("━" * 65)

    quant_config = None
    if config.USE_4BIT_QUANTIZATION:
        try:
            # Test that bitsandbytes actually works before using it
            import bitsandbytes as bnb
            print("  4-bit quantization: ENABLED (NF4 + double quantization)")
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        except (ImportError, RuntimeError, Exception) as e:
            print(f"  [WARNING] bitsandbytes failed to load: {e}")
            print(f"  Falling back to full precision (no 4-bit quantization).")
            print(f"  This is fine for models ≤14B on your 32GB R9700.")
            print(f"  To fix: install ROCm HIP SDK and rebuild bitsandbytes from source.")
            print()
            config.USE_4BIT_QUANTIZATION = False
    else:
        print("  4-bit quantization: DISABLED (full precision)")

    # ── Step 4: Load model ─────────────────────────────────────────────────
    print(f"  Loading {config.MODEL_NAME}...")

    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    print(f"  Model loaded successfully.")

    # Prepare for k-bit training if quantized
    if config.USE_4BIT_QUANTIZATION:
        model = prepare_model_for_kbit_training(model)
        print("  Model prepared for k-bit training.")

    # Enable gradient checkpointing
    if config.USE_GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing: ENABLED")

    print()

    # ── Step 5: Apply LoRA ─────────────────────────────────────────────────
    print("━" * 65)
    print("  Step 4: Applying LoRA Adapter")
    print("━" * 65)

    target_modules = get_target_modules(config.MODEL_NAME)
    print(f"  Target modules: {target_modules}")

    lora_config = LoraConfig(
        r=config.LORA_RANK,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    # Print parameter counts
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable_params / total_params if total_params > 0 else 0

    print(f"  Trainable parameters: {trainable_params:,} / {total_params:,} ({pct:.2f}%)")
    print()

    # ── Step 6: Load and tokenize dataset ──────────────────────────────────
    print("━" * 65)
    print("  Step 5: Loading Dataset")
    print("━" * 65)

    dataset = load_dataset("json", data_files=data_path, split="train")
    print(f"  Dataset loaded: {len(dataset)} examples")
    print()

    # ── Step 7: Training ───────────────────────────────────────────────────
    print("━" * 65)
    print("  Step 6: Training")
    print("━" * 65)

    # Pre-training memory cleanup
    gc.collect()
    torch.cuda.empty_cache()

    # Disable cache for training (incompatible with gradient checkpointing)
    model.config.use_cache = False

    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM_STEPS,
        learning_rate=config.LEARNING_RATE,
        max_grad_norm=0.3,
        warmup_ratio=config.WARMUP_RATIO,
        lr_scheduler_type="cosine",
        logging_dir=config.LOGS_DIR,
        logging_steps=config.LOGGING_STEPS,
        save_steps=config.SAVE_STEPS,
        save_total_limit=2,
        fp16=True,
        bf16=False,
        optim="paged_adamw_8bit",
        report_to="none",
        gradient_checkpointing=config.USE_GRADIENT_CHECKPOINTING,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        group_by_length=True,
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        max_seq_length=config.MAX_SEQ_LENGTH,
    )

    print(f"\n  Starting training...")
    print(f"  Epochs: {config.EPOCHS}")
    print(f"  Total steps: {trainer.state.max_steps if hasattr(trainer.state, 'max_steps') else '(calculating...)'}")
    print()

    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time

    # Format time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    time_str = f"{hours}h {minutes}m {seconds}s" if hours > 0 else f"{minutes}m {seconds}s"

    # ── Step 8: Save ───────────────────────────────────────────────────────
    print()
    print("━" * 65)
    print("  Step 7: Saving Adapter")
    print("━" * 65)

    # Save LoRA adapter
    model.save_pretrained(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)

    output_abs = os.path.abspath(config.OUTPUT_DIR)
    print(f"  LoRA adapter saved to: {output_abs}")

    # ── Done ───────────────────────────────────────────────────────────────
    print()
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║                     TRAINING COMPLETE                            ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")
    print(f"║  Training time:  {time_str:<48}║")
    print(f"║  Output:         {output_abs:<48}║")
    print("║                                                                   ║")
    print("║  Next: python inference.py  (to test your model)                  ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print()

    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
