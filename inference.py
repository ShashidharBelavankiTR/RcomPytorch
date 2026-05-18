"""
Inference — Test your fine-tuned model interactively
=====================================================
Loads the base model + LoRA adapter from /output, merges them,
and starts an interactive chat loop in the terminal.

Usage: python inference.py
"""

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  ROCm setup MUST come before ANY torch import                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
import rocm_setup
rocm_setup.configure()

import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import config
from model_registry import get_chat_template, format_prompt


def main():
    print()
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║              MY AI TRAINER — Interactive Inference               ║")
    print("║         AMD Radeon AI PRO R9700 32GB · gfx1201 · ROCm 7.x      ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print()

    if not torch.cuda.is_available():
        print("[FATAL] GPU not detected. Run: python setup_check.py")
        sys.exit(1)

    adapter_path = os.path.abspath(config.OUTPUT_DIR)

    # Check adapter exists
    adapter_config = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(adapter_config):
        print(f"  [ERROR] No trained adapter found at: {adapter_path}")
        print(f"  Run 'python train.py' first to train a model.")
        sys.exit(1)

    print(f"  Base model:  {config.MODEL_NAME}")
    print(f"  Adapter:     {adapter_path}")
    print()

    # ── Load base model ────────────────────────────────────────────────────
    print("  Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    # ── Load tokenizer ─────────────────────────────────────────────────────
    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load and merge LoRA adapter ────────────────────────────────────────
    print("  Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("  Merging adapter into base model...")
    model = model.merge_and_unload()

    model.eval()

    print()
    print("  Model loaded and merged successfully!")
    print()
    print("  ─────────────────────────────────────────────────────────────")
    print("  Interactive mode. Type your message and press Enter.")
    print("  Type 'exit' or 'quit' to stop.")
    print("  ─────────────────────────────────────────────────────────────")
    print()

    # Get chat template for this model
    template = get_chat_template(config.MODEL_NAME)

    while True:
        try:
            user_input = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("  Goodbye!")
            break

        # Format prompt using model's chat template
        prompt = format_prompt(template, user_input, system_text=config.SYSTEM_PROMPT)

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the generated part (skip the prompt tokens)
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        print(f"\n  AI: {response}\n")


if __name__ == "__main__":
    main()
