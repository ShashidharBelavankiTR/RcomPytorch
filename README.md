# MY AI TRAINER

**Local LLM Fine-Tuning Framework for AMD Radeon AI PRO R9700 32GB**

Fine-tune any Hugging Face language model on your own data using LoRA — optimized specifically for the AMD Radeon AI PRO R9700 (gfx1201, RDNA 4, 32GB VRAM) on Windows 11 with ROCm 7.x.

---

## 1. Your GPU — AMD Radeon AI PRO R9700

| Spec | Value |
|------|-------|
| GPU | AMD Radeon AI PRO R9700 |
| Architecture | RDNA 4 |
| GFX Code | gfx1201 |
| VRAM | 32 GB GDDR6 |
| ROCm Support | **ROCm 7.x ONLY** |
| Driver | AMD Adrenalin 25.x or later |

### Why ROCm 7.x?

The R9700's `gfx1201` architecture is **RDNA 4** — it is **not supported in ROCm 6.x at all**. You must use ROCm 7.x PyTorch wheels. The framework automatically sets all required environment variables:

- `HSA_OVERRIDE_GFX_VERSION=12.0.1` — maps gfx1201 to the ROCm runtime
- `PYTORCH_ROCM_ARCH=gfx1201` — targets the correct GPU architecture
- `HIP_VISIBLE_DEVICES=0` / `ROCR_VISIBLE_DEVICES=0` — single GPU selection
- `GPU_MAX_HW_QUEUES=8` — maximum hardware queue throughput
- `PYTORCH_TUNABLEOP_ENABLED=1` — ROCm kernel auto-tuning

---

## 2. Quick Start

### One-Line Do Everything

```bash
python install.py && python setup_check.py && python train.py
```

Installs deps → verifies GPU → trains. Stops if any step fails.

---

### Step 1: Install Dependencies

```bash
python install.py
```

This installs:
- PyTorch from ROCm 7.x wheels (repo.radeon.com)
- All Python dependencies from requirements.txt

### Step 2: Verify Setup

```bash
python setup_check.py
```

Runs PASS/FAIL checks for Python, PyTorch, ROCm, GPU detection, env vars, VRAM, and all packages.

### Step 3: Add Training Data

Drop your files into the `train/` folder. Supported formats:

| Format | Extension |
|--------|-----------|
| Plain text | `.txt` |
| PDF (text-based) | `.pdf` |
| Microsoft Word | `.docx` |
| Excel spreadsheet | `.xlsx`, `.xls` |
| Markdown | `.md` |

### Step 4: Configure (Optional)

Edit `config.py` to change the model, training parameters, or LoRA settings. The defaults work great for 7B models on 32GB VRAM.

### Step 5: Chunk Data Only (Optional)

To run **only** the data pipeline (extract → chunk → deduplicate → validate) without training:

```bash
python chunk.py
```

This scans `train/`, processes all supported file types, removes near-duplicate chunks, prints a quality report, and saves the output to `data/train.jsonl`. Useful for inspecting your data before committing to a full training run.

### Step 6: Train

```bash
python train.py
```

That's it. The script handles chunking, tokenization, model downloading, LoRA, training, and saving. All supported models (Qwen, Mistral, Phi-4, DeepSeek, Falcon, Yi, InternLM, OLMo, Mixtral) are Apache 2.0 / MIT licensed — **no HuggingFace token or API key needed**.

### Step 7: Test

```bash
python inference.py
```

Interactive chat with your fine-tuned model.

---

## 3. Model Compatibility — 32GB VRAM

### Full Precision (USE_4BIT_QUANTIZATION = False)

These models fit in 32GB VRAM at full FP16 precision:

| Model | Params | VRAM Est. |
|-------|--------|-----------|
| Qwen2.5-0.5B to 7B-Instruct | 0.5–7B | 1–14 GB |
| Qwen2.5-14B-Instruct | 14B | ~28 GB |
| Qwen3-4B, 8B, 14B | 4–14B | 8–28 GB |
| Llama-3.2-3B, Llama-3-8B, Llama-3.1-8B | 3–8B | 6–16 GB |
| Mistral-7B-Instruct | 7B | ~14 GB |
| Mistral-NeMo-12B | 12B | ~24 GB |
| Gemma-2-2b, 9b / Gemma-3-4b, 12b | 2–12B | 4–24 GB |
| Phi-3.5-mini, Phi-4-mini, Phi-4 | 3.8–14B | 8–28 GB |
| DeepSeek-R1-Distill 7B, 8B, 14B | 7–14B | 14–28 GB |
| Falcon 7B, 11B | 7–11B | 14–22 GB |
| Yi-1.5-9B | 9B | ~18 GB |
| InternLM2.5-7B, 20B | 7–20B | 14–30 GB |
| OLMo-2-7B, 13B | 7–13B | 14–26 GB |

### 4-Bit Quantization Required (USE_4BIT_QUANTIZATION = True)

These models need quantization to fit in 32GB:

| Model | Params | VRAM Est. (4-bit) |
|-------|--------|-------------------|
| Qwen3-30B-A3B, Qwen3-32B | 30–32B | ~18–20 GB |
| Qwen2.5-32B-Instruct | 32B | ~20 GB |
| DeepSeek-R1-Distill-32B | 32B | ~20 GB |
| Yi-1.5-34B | 34B | ~22 GB |
| Gemma-2-27b, Gemma-3-27b | 27B | ~16 GB |
| Llama-3.3-70B | 70B | ~28 GB (tight) |
| Mixtral-8x7B | 46.7B (MoE) | ~24 GB |

---

## 4. R9700-Specific Troubleshooting

### gfx1201 Not Detected

**Symptom:** `torch.cuda.is_available()` returns `False`

**Fixes:**
1. Ensure AMD Adrenalin **25.x or later** driver is installed
2. Install PyTorch from ROCm 7.x wheels: `python install.py`
3. Verify env vars are set (automatic via `rocm_setup.py`):
   ```
   HSA_OVERRIDE_GFX_VERSION=12.0.1
   PYTORCH_ROCM_ARCH=gfx1201
   ```
4. Restart your terminal and VS Code after driver installation
5. **gfx1201 does NOT work on ROCm 6.x** — you must use ROCm 7.x

### HSA_OVERRIDE_GFX_VERSION Not Working

If the env var doesn't take effect:
1. Set it as a **system** environment variable in Windows:
   - Settings → System → About → Advanced System Settings → Environment Variables
   - Add: `HSA_OVERRIDE_GFX_VERSION` = `12.0.1`
2. Restart your PC
3. Verify: `echo %HSA_OVERRIDE_GFX_VERSION%` should print `12.0.1`

### bitsandbytes Issues on Windows + ROCm

**Symptom:** `bitsandbytes` fails with `libbitsandbytes_rocm72.dll` not found or ROCm GPU architecture detection errors.

**Why:** The `bitsandbytes` pip package does not include the ROCm DLL for Windows. This is a known upstream issue.

**Impact:** 4-bit quantization (`USE_4BIT_QUANTIZATION = True`) won't work without the fix below.

**Workaround (recommended):** Keep `USE_4BIT_QUANTIZATION = False` in `config.py`. With 32GB VRAM on the R9700, models ≤14B fit in full precision — no quantization needed. The framework automatically falls back to full precision if bitsandbytes fails.

**If you need 4-bit for 27B+ models:**
1. Install AMD ROCm HIP SDK for Windows
2. Build from source:
   ```bash
   pip install bitsandbytes --no-binary bitsandbytes
   ```
3. Verify the DLL exists at the path shown in the error message

### Out of Memory Errors

1. Reduce `BATCH_SIZE` to 1 in `config.py`
2. Reduce `MAX_SEQ_LENGTH` to 1024
3. Enable `USE_4BIT_QUANTIZATION = True`
4. Enable `USE_GRADIENT_CHECKPOINTING = True`
5. Reduce `LORA_RANK` to 16

### Training is Slow

1. Ensure `PYTORCH_TUNABLEOP_ENABLED=1` is set (automatic via `rocm_setup.py`)
2. Ensure `GPU_MAX_HW_QUEUES=8` is set (automatic)
3. First epoch may be slow due to ROCm kernel tuning — subsequent epochs will be faster
4. Increase `BATCH_SIZE` to 4 if VRAM allows (monitor with `rocm-smi`)

---

## 5. Driver Requirements

### AMD Adrenalin Driver

**Minimum:** Adrenalin Edition 25.x or later

The gfx1201 architecture (RDNA 4) requires a recent Adrenalin driver that includes ROCm 7.x HIP runtime support for Windows 11. Older drivers (24.x and below) do not include gfx1201 support.

**To install/update:**
1. Download from: https://www.amd.com/en/support
2. Select: Radeon AI PRO → R9700
3. Install the recommended driver
4. Restart your PC
5. Run `python setup_check.py` to verify

### ROCm on Windows

ROCm on Windows works through the HIP runtime bundled with the Adrenalin driver. You do **not** need to install ROCm separately on Windows — the driver includes everything PyTorch needs.

---

## Project Structure

```
my-ai-trainer/
├── train/              ← Drop your .txt and .pdf files here
├── output/             ← Trained LoRA adapter saved here
├── data/               ← Auto-created: stores train.jsonl
├── logs/               ← Auto-created: training logs
├── rocm_setup.py       ← Sets env vars for gfx1201 (auto-imported)
├── config.py           ← YOUR SETTINGS — edit only this file
├── model_registry.py   ← Auto-detects LoRA targets + chat templates
├── chunk.py            ← Reads .txt/.pdf, chunks, saves JSONL
├── train.py            ← Main entry: python train.py
├── inference.py        ← Test your model: python inference.py
├── setup_check.py      ← Verify everything: python setup_check.py
├── install.py          ← Install dependencies: python install.py
├── requirements.txt    ← Python package list
├── .env                ← Your HF_TOKEN (not committed to git)
├── .env.example        ← Template for .env
└── README.md           ← This file
```

---

## Files You Edit

| File | What to change |
|------|----------------|
| `config.py` | Model name, training parameters, LoRA settings |
| `.env` | Your Hugging Face token |

**Everything else is automatic.** Just run `python train.py`.
