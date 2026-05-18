"""
Configuration — AMD Radeon AI PRO R9700 32GB (gfx1201)
======================================================
THIS IS THE ONLY FILE YOU NEED TO EDIT.
Change MODEL_NAME to switch models. Adjust training params as needed.
Everything else is automatic.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL SELECTION
# ═══════════════════════════════════════════════════════════════════════════════
# Change this to any supported model. Just swap the name and run train.py.
#
# ── Qwen 2.5 Series ──────────────────────────────────────────────────────────
#   "Qwen/Qwen2.5-0.5B-Instruct"
#   "Qwen/Qwen2.5-1.5B-Instruct"
#   "Qwen/Qwen2.5-3B-Instruct"
#   "Qwen/Qwen2.5-7B-Instruct"          ← full precision OK on 32GB
#   "Qwen/Qwen2.5-14B-Instruct"         ← full precision OK on 32GB
#   "Qwen/Qwen2.5-32B-Instruct"         ← requires 4-bit quantization
#
# ── Qwen 3 Series ────────────────────────────────────────────────────────────
#   "Qwen/Qwen3-4B"
#   "Qwen/Qwen3-8B"                     ← full precision OK on 32GB
#   "Qwen/Qwen3-14B"                    ← full precision OK on 32GB
#   "Qwen/Qwen3-30B-A3B"               ← requires 4-bit quantization
#   "Qwen/Qwen3-32B"                    ← requires 4-bit quantization
#
# ── Llama 3 / 3.1 / 3.2 / 3.3 Series ────────────────────────────────────────
#   "meta-llama/Llama-3.2-3B-Instruct"
#   "meta-llama/Meta-Llama-3-8B-Instruct"         ← full precision OK on 32GB
#   "meta-llama/Meta-Llama-3.1-8B-Instruct"       ← full precision OK on 32GB
#   "meta-llama/Llama-3.3-70B-Instruct"           ← requires 4-bit quantization
#
# ── Mistral Series ────────────────────────────────────────────────────────────
#   "mistralai/Mistral-7B-Instruct-v0.3"          ← full precision OK on 32GB
#   "mistralai/Mistral-Nemo-Instruct-2407"        ← full precision OK on 32GB (12B)
#   "mistralai/Mixtral-8x7B-Instruct-v0.1"        ← requires 4-bit quantization
#
# ── Gemma 2 & 3 Series ───────────────────────────────────────────────────────
#   "google/gemma-2-2b-it"
#   "google/gemma-2-9b-it"               ← full precision OK on 32GB
#   "google/gemma-2-27b-it"              ← requires 4-bit quantization
#   "google/gemma-3-4b-it"
#   "google/gemma-3-12b-it"              ← full precision OK on 32GB
#   "google/gemma-3-27b-it"              ← requires 4-bit quantization
#
# ── Phi Series ────────────────────────────────────────────────────────────────
#   "microsoft/Phi-3.5-mini-instruct"    ← full precision OK on 32GB (3.8B)
#   "microsoft/phi-4-mini-instruct"      ← full precision OK on 32GB
#   "microsoft/phi-4"                    ← full precision OK on 32GB (14B)
#
# ── DeepSeek R1 Distill Series ───────────────────────────────────────────────
#   "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"    ← full precision OK on 32GB
#   "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"   ← full precision OK on 32GB
#   "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"   ← full precision OK on 32GB
#   "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"   ← requires 4-bit quantization
#
# ── Falcon Series ─────────────────────────────────────────────────────────────
#   "tiiuae/falcon-7b-instruct"          ← full precision OK on 32GB
#   "tiiuae/falcon-11B"                  ← full precision OK on 32GB
#
# ── Yi 1.5 Series ────────────────────────────────────────────────────────────
#   "01-ai/Yi-1.5-9B-Chat"              ← full precision OK on 32GB
#   "01-ai/Yi-1.5-34B-Chat"             ← requires 4-bit quantization
#
# ── InternLM 2.5 Series ──────────────────────────────────────────────────────
#   "internlm/internlm2_5-7b-chat"      ← full precision OK on 32GB
#   "internlm/internlm2_5-20b-chat"     ← full precision OK on 32GB
#
# ── OLMo 2 Series ────────────────────────────────────────────────────────────
#   "allenai/OLMo-2-7B-Instruct"        ← full precision OK on 32GB
#   "allenai/OLMo-2-13B-Instruct"       ← full precision OK on 32GB
#
# ── Any HuggingFace AutoModelForCausalLM model ───────────────────────────────
#   Any model that works with AutoModelForCausalLM.from_pretrained() is supported.
#   The framework auto-detects LoRA target modules and chat templates.

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# ═══════════════════════════════════════════════════════════════════════════════
# VRAM USAGE GUIDE — AMD Radeon AI PRO R9700 (32GB)
# ═══════════════════════════════════════════════════════════════════════════════
#
# ┌────────────────────────────────────────────────────────────────────────────┐
# │  FULL PRECISION (USE_4BIT_QUANTIZATION = False) — fits in 32GB:          │
# │                                                                          │
# │    All ≤7B models:   ~14 GB VRAM during training                         │
# │    8B–9B models:     ~18 GB VRAM during training                         │
# │    12B–14B models:   ~24–28 GB VRAM during training                      │
# │    20B models:       ~30 GB VRAM during training (tight fit)             │
# │                                                                          │
# │  4-BIT QUANTIZATION (USE_4BIT_QUANTIZATION = True) — required for:       │
# │                                                                          │
# │    27B–32B models:   ~18–22 GB VRAM during training                      │
# │    34B models:       ~22 GB VRAM during training                         │
# │    70B models:       ~28–32 GB VRAM during training (tight fit)          │
# │    Mixtral 8x7B:     ~24 GB VRAM during training                         │
# └────────────────────────────────────────────────────────────────────────────┘

# ═══════════════════════════════════════════════════════════════════════════════
# DIRECTORIES
# ═══════════════════════════════════════════════════════════════════════════════

# Where you put your .txt and .pdf training files
TRAIN_DIR = "./train"

# Auto-created: chunked training data (train.jsonl) goes here
DATA_DIR = "./data"

# Auto-created: trained LoRA adapter saved here after training
OUTPUT_DIR = "./output"

# Auto-created: training logs saved here
LOGS_DIR = "./logs"

# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

# This is the system prompt used during BOTH training and inference.
# It tells the model what role to play and how to behave.
# Change this to match your use case — the model learns to follow it during training.
#
# Examples:
#   "You are a legal assistant specializing in contract law."
#   "You are a Python expert who writes clean, efficient code."
#   "You are a medical knowledge assistant. Always recommend consulting a doctor."
#   "You are a customer support agent for Acme Corp. Be polite and helpful."
SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. Use the following information "
    "to answer questions accurately and helpfully."
)

# ═══════════════════════════════════════════════════════════════════════════════
# CHUNKING SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

# Number of words per chunk — R9700 32GB can handle large chunks
CHUNK_SIZE = 1024

# Overlap between chunks in words — ensures context continuity
CHUNK_OVERLAP = 100

# ── Supported file types (all scanned recursively from /train) ────────────────
# .txt  — plain text
# .pdf  — PDF documents (text-based; scanned PDFs need OCR)
# .docx — Microsoft Word documents (text + tables)
# .xlsx — Microsoft Excel spreadsheets (all sheets, all cells)
# .xls  — Legacy Excel format
# .md   — Markdown files (markdown syntax is stripped before chunking)

# ═══════════════════════════════════════════════════════════════════════════════
# DEDUPLICATION SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

# Remove near-duplicate chunks before saving to JSONL
# Recommended: True — duplicate chunks waste training compute
DEDUP_ENABLED = True

# Jaccard similarity threshold (0.0–1.0) above which two chunks are considered
# near-duplicates. The second chunk is discarded.
#   0.95 = only near-identical chunks removed (strict)
#   0.85 = chunks with 85%+ word trigram overlap removed (recommended)
#   0.70 = more aggressive dedup (use if dataset has lots of repetition)
DEDUP_THRESHOLD = 0.85

# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

# Validate chunk quality and print a distribution report after chunking
VALIDATE_CHUNKS = True

# Chunks with fewer words than this are flagged as too short (low information)
MIN_CHUNK_WORDS = 50

# Chunks with more words than this are flagged as too long
# Defaults to CHUNK_SIZE * 1.5 at runtime if set to None
MAX_CHUNK_WORDS = None

# ═══════════════════════════════════════════════════════════════════════════════
# LoRA SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

# LoRA rank — higher = more expressive but more VRAM; 32 is great for 32GB
LORA_RANK = 32

# LoRA alpha — scaling factor, typically 2x the rank
LORA_ALPHA = 64

# LoRA dropout — regularization to prevent overfitting (0.05 = 5%)
LORA_DROPOUT = 0.05

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

# Number of full passes through the training data
EPOCHS = 3

# Samples per GPU step — R9700 32GB can handle 2 for most 7B–14B models
BATCH_SIZE = 2

# Accumulate gradients over this many steps before updating weights
# Effective batch size = BATCH_SIZE × GRAD_ACCUM_STEPS = 2 × 8 = 16
GRAD_ACCUM_STEPS = 8

# Learning rate — 2e-4 is a good default for LoRA fine-tuning
LEARNING_RATE = 2e-4

# Maximum sequence length in tokens — R9700 32GB supports longer sequences
MAX_SEQ_LENGTH = 2048

# Fraction of training steps for learning rate warmup
WARMUP_RATIO = 0.05

# Print training metrics every N steps
LOGGING_STEPS = 10

# Save a checkpoint every N steps
SAVE_STEPS = 100

# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════

# Saves VRAM by recomputing activations instead of storing them
# Recommended: True for models > 7B, optional for smaller models
USE_GRADIENT_CHECKPOINTING = True

# Load model in 4-bit precision to save VRAM
# Required for 27B+ models on 32GB, optional for smaller models
# Set to False for ≤14B models if you want full precision training
USE_4BIT_QUANTIZATION = True

# ═══════════════════════════════════════════════════════════════════════════════
# HUGGING FACE TOKEN — NOT REQUIRED
# ═══════════════════════════════════════════════════════════════════════════════
# All supported models use Apache 2.0 or MIT licenses and are freely accessible
# without authentication. No token is needed.
HF_TOKEN = None
