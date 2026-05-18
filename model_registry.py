"""
Model Registry — Auto-detect LoRA targets, chat templates, and model info
=========================================================================
Supports all major model families. No user editing needed.
Add new families by adding entries to the dictionaries below.
"""


def get_target_modules(model_name: str) -> list[str]:
    """
    Return the correct LoRA target module names for the given model.
    Uses keyword matching on the model name / HuggingFace repo ID.
    """
    name = model_name.lower()

    # Phi models have a different architecture
    if "phi" in name:
        return ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]

    # Falcon uses fused QKV projection
    if "falcon" in name:
        return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]

    # Mixtral has extra expert gate layers
    if "mixtral" in name:
        return [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "w1", "w2", "w3",
        ]

    # Standard transformer architecture used by most modern models:
    # Qwen, Llama, Mistral, Gemma, DeepSeek, OLMo, Yi, InternLM
    standard_families = [
        "qwen", "llama", "mistral", "gemma", "deepseek",
        "olmo", "yi", "internlm",
    ]
    for family in standard_families:
        if family in name:
            return [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]

    # Default fallback — works for most AutoModelForCausalLM models
    return ["q_proj", "v_proj"]


def get_chat_template(model_name: str) -> dict:
    """
    Return the chat template format for the given model family.
    Returns a dict with 'system_start', 'system_end', 'user_start',
    'user_end', 'assistant_start', 'assistant_end' strings.
    """
    name = model_name.lower()

    # ── Qwen / DeepSeek-R1-Distill-Qwen — ChatML format ──────────────────────
    if "qwen" in name or ("deepseek" in name and "qwen" in name):
        return {
            "system_start": "<|im_start|>system\n",
            "system_end": "<|im_end|>\n",
            "user_start": "<|im_start|>user\n",
            "user_end": "<|im_end|>\n",
            "assistant_start": "<|im_start|>assistant\n",
            "assistant_end": "<|im_end|>\n",
        }

    # ── Llama 3 / 3.1 / 3.2 / 3.3 / DeepSeek-R1-Distill-Llama ──────────────
    if "llama" in name or ("deepseek" in name and "llama" in name):
        return {
            "system_start": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
            "system_end": "<|eot_id|>",
            "user_start": "<|start_header_id|>user<|end_header_id|>\n\n",
            "user_end": "<|eot_id|>",
            "assistant_start": "<|start_header_id|>assistant<|end_header_id|>\n\n",
            "assistant_end": "<|eot_id|>",
        }

    # ── Mistral / Mixtral — [INST] format ─────────────────────────────────────
    if "mistral" in name or "mixtral" in name:
        return {
            "system_start": "",
            "system_end": "",
            "user_start": "[INST] ",
            "user_end": " [/INST]",
            "assistant_start": "",
            "assistant_end": "</s>",
        }

    # ── Gemma 2 & 3 — turn-based format ──────────────────────────────────────
    if "gemma" in name:
        return {
            "system_start": "",
            "system_end": "",
            "user_start": "<start_of_turn>user\n",
            "user_end": "<end_of_turn>\n",
            "assistant_start": "<start_of_turn>model\n",
            "assistant_end": "<end_of_turn>\n",
        }

    # ── Phi — Instruct/Output format ─────────────────────────────────────────
    if "phi" in name:
        return {
            "system_start": "<|system|>\n",
            "system_end": "<|end|>\n",
            "user_start": "<|user|>\n",
            "user_end": "<|end|>\n",
            "assistant_start": "<|assistant|>\n",
            "assistant_end": "<|end|>\n",
        }

    # ── InternLM ──────────────────────────────────────────────────────────────
    if "internlm" in name:
        return {
            "system_start": "<|im_start|>system\n",
            "system_end": "<|im_end|>\n",
            "user_start": "<|im_start|>user\n",
            "user_end": "<|im_end|>\n",
            "assistant_start": "<|im_start|>assistant\n",
            "assistant_end": "<|im_end|>\n",
        }

    # ── Falcon ────────────────────────────────────────────────────────────────
    if "falcon" in name:
        return {
            "system_start": "System: ",
            "system_end": "\n",
            "user_start": "User: ",
            "user_end": "\n",
            "assistant_start": "Falcon: ",
            "assistant_end": "\n",
        }

    # ── Yi ────────────────────────────────────────────────────────────────────
    if "yi" in name:
        return {
            "system_start": "<|im_start|>system\n",
            "system_end": "<|im_end|>\n",
            "user_start": "<|im_start|>user\n",
            "user_end": "<|im_end|>\n",
            "assistant_start": "<|im_start|>assistant\n",
            "assistant_end": "<|im_end|>\n",
        }

    # ── OLMo ──────────────────────────────────────────────────────────────────
    if "olmo" in name:
        return {
            "system_start": "<|system|>\n",
            "system_end": "\n",
            "user_start": "<|user|>\n",
            "user_end": "\n",
            "assistant_start": "<|assistant|>\n",
            "assistant_end": "\n",
        }

    # ── DeepSeek (generic, non-distill) ───────────────────────────────────────
    if "deepseek" in name:
        return {
            "system_start": "<|im_start|>system\n",
            "system_end": "<|im_end|>\n",
            "user_start": "<|im_start|>user\n",
            "user_end": "<|im_end|>\n",
            "assistant_start": "<|im_start|>assistant\n",
            "assistant_end": "<|im_end|>\n",
        }

    # ── Default — Alpaca format (works with most unknown models) ──────────────
    return {
        "system_start": "### System:\n",
        "system_end": "\n\n",
        "user_start": "### Instruction:\n",
        "user_end": "\n\n",
        "assistant_start": "### Response:\n",
        "assistant_end": "\n\n",
    }


def format_prompt(template: dict, user_text: str,
                  system_text: str = "You are a helpful assistant.") -> str:
    """
    Format a single user message into the model's expected prompt structure.
    """
    prompt = ""
    if template["system_start"]:
        prompt += template["system_start"] + system_text + template["system_end"]
    prompt += template["user_start"] + user_text + template["user_end"]
    prompt += template["assistant_start"]
    return prompt


def format_training_example(template: dict, user_text: str, assistant_text: str,
                            system_text: str = "You are a helpful assistant.") -> str:
    """
    Format a complete training example with system, user, and assistant turns.
    """
    prompt = ""
    if template["system_start"]:
        prompt += template["system_start"] + system_text + template["system_end"]
    prompt += template["user_start"] + user_text + template["user_end"]
    prompt += template["assistant_start"] + assistant_text + template["assistant_end"]
    return prompt


def get_model_info(model_name: str) -> dict:
    """
    Return human-readable info about the model: family, approx params,
    VRAM estimate, and whether 4-bit quantization is needed on 32GB.
    """
    name = model_name.lower()

    # Detect family
    family = "Unknown"
    families = {
        "qwen2.5": "Qwen 2.5", "qwen3": "Qwen 3", "qwen": "Qwen",
        "llama-3.3": "Llama 3.3", "llama-3.2": "Llama 3.2",
        "llama-3.1": "Llama 3.1", "llama-3": "Llama 3", "llama": "Llama",
        "mixtral": "Mixtral", "mistral-nemo": "Mistral NeMo", "mistral": "Mistral",
        "gemma-3": "Gemma 3", "gemma-2": "Gemma 2", "gemma": "Gemma",
        "phi-4-mini": "Phi-4-mini", "phi-4": "Phi-4",
        "phi-3.5": "Phi-3.5", "phi": "Phi",
        "deepseek": "DeepSeek R1 Distill",
        "falcon": "Falcon",
        "yi": "Yi 1.5",
        "internlm": "InternLM 2.5",
        "olmo": "OLMo 2",
    }
    for key, val in families.items():
        if key in name:
            family = val
            break

    # Detect param size from model name
    param_size = "Unknown"
    import re
    size_match = re.search(r"(\d+\.?\d*)[bB]", name)
    if size_match:
        param_size = size_match.group(1) + "B"

    # Estimate VRAM and 4-bit requirement
    size_num = 0.0
    if size_match:
        size_num = float(size_match.group(1))

    if "mixtral" in name:
        needs_4bit = True
        vram_full = "~90 GB"
        vram_4bit = "~24 GB"
    elif size_num > 20:
        needs_4bit = True
        vram_full = f"~{size_num * 2:.0f} GB"
        vram_4bit = f"~{size_num * 0.6:.0f} GB"
    elif size_num > 14:
        needs_4bit = False
        vram_full = f"~{size_num * 2:.0f} GB"
        vram_4bit = f"~{size_num * 0.6:.0f} GB"
    elif size_num > 0:
        needs_4bit = False
        vram_full = f"~{size_num * 2:.0f} GB"
        vram_4bit = f"~{size_num * 0.6:.0f} GB"
    else:
        needs_4bit = False
        vram_full = "Unknown"
        vram_4bit = "Unknown"

    info = {
        "family": family,
        "param_size": param_size,
        "vram_full_precision": vram_full,
        "vram_4bit": vram_4bit,
        "needs_4bit_on_32gb": needs_4bit,
    }

    return info


def print_model_info(model_name: str):
    """Pretty-print model info for the user."""
    info = get_model_info(model_name)
    print(f"  Model Family:       {info['family']}")
    print(f"  Parameters:         {info['param_size']}")
    print(f"  VRAM (full prec.):  {info['vram_full_precision']}")
    print(f"  VRAM (4-bit):       {info['vram_4bit']}")
    if info["needs_4bit_on_32gb"]:
        print(f"  4-bit required:     YES (model too large for 32GB full precision)")
    else:
        print(f"  4-bit required:     NO  (fits in 32GB at full precision)")


if __name__ == "__main__":
    # Quick test
    test_models = [
        "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "microsoft/phi-4",
        "tiiuae/falcon-7b-instruct",
        "google/gemma-2-9b-it",
    ]
    for m in test_models:
        print(f"\n{'─' * 50}")
        print(f"  Model: {m}")
        print(f"  LoRA targets: {get_target_modules(m)}")
        print_model_info(m)
