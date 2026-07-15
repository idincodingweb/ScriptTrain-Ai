# ============================================================
# KIKAI CASUAL-VIBE — TRAINING v3 (Qwen2.5-3B-Instruct)
# ============================================================
import os, gc, json, math, time, random, warnings, inspect, traceback
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
os.environ["TOKENIZERS_PARALLELISM"]  = "false"
os.environ["BITSANDBYTES_NOWELCOME"]  = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import trl
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
from huggingface_hub import HfApi, create_repo, login

torch.cuda.empty_cache(); gc.collect()

print(f"🔍 TRL version: {trl.__version__}")

# ============================================================
# 🎯 CONFIG (di-tune untuk Qwen2.5-3B-Instruct)
# ============================================================
BASE_MODEL       = "Qwen/Qwen2.5-3B-Instruct"  # 🆕 GANTI
TRAIN_JSONL      = "/content/kikai_train.jsonl"
RESPONSE_IDS_FN  = "/content/response_ids.json"

OUTPUT_DIR       = "/content/KiKai-casual-adapter"

MAX_SEQ_LEN      = 1024
EPOCHS           = 2         # data besar, cepat converge
LR               = 2e-4      # 🟡 3B model butuh LR sedikit lebih tinggi dari 7B
MICRO_BATCH      = 2         # 🆕 3B lebih ringan, bisa naik dari 1
GRAD_ACCUM       = 8         # 🆕 turunin dari 16 (effective batch tetep 16)
EVAL_SUBSAMPLE   = 300
LOGGING_STEPS    = 20
WARMUP_RATIO     = 0.05

IDENTITY_KEYWORDS = ["KiKai", "Idin Iskandar", "Idin"]

# HuggingFace target
HF_REPO_ID       = "IDINN/KiKai"
HF_PATH_IN_REPO  = "adapter/casual-train"
HF_PRIVATE       = True
HF_PUSH_EVERY_SAVE = True

# ============================================================
# HF LOGIN CHECK
# ============================================================
print("\n🔐 Checking HuggingFace login...")
try:
    hf_api = HfApi()
    user_info = hf_api.whoami()
    print(f"  ✅ Logged in as: {user_info['name']}")
except Exception:
    print("  ⚠️  Belum login, minta token...")
    login()
    hf_api = HfApi()
    user_info = hf_api.whoami()
    print(f"  ✅ Logged in as: {user_info['name']}")

print(f"\n🏗️  Ensuring repo: {HF_REPO_ID}")
create_repo(repo_id=HF_REPO_ID, repo_type="model", private=HF_PRIVATE, exist_ok=True)
print(f"  ✅ Repo ready")
print(f"  📁 Target path: /{HF_PATH_IN_REPO}/")

# ============================================================
# COLLATOR AUTO-DETECT
# ============================================================
COLLATOR_CLS = None
USE_COMPLETION_ONLY_CONFIG = False

try:
    from trl import DataCollatorForCompletionOnlyLM as COLLATOR_CLS
    print("  ✅ Collator: trl.DataCollatorForCompletionOnlyLM")
except ImportError:
    try:
        from trl.trainer.utils import DataCollatorForCompletionOnlyLM as COLLATOR_CLS
        print("  ✅ Collator: trl.trainer.utils.DataCollatorForCompletionOnlyLM")
    except ImportError:
        sig = inspect.signature(SFTConfig.__init__)
        if 'completion_only_loss' in sig.parameters:
            USE_COMPLETION_ONLY_CONFIG = True
            print("  ✅ Pake: SFTConfig(completion_only_loss=True)")

assert torch.cuda.is_available()
gpu_cap = torch.cuda.get_device_capability(0)
print(f"\n🖥️  GPU: {torch.cuda.get_device_name(0)} | SM: {gpu_cap[0]}.{gpu_cap[1]}")

# Qwen2.5-3B kecil, kita bisa coba bf16 kalau SM >= 8, fallback fp16
if gpu_cap[0] >= 8:
    compute_dtype = torch.bfloat16
    print(f"  ⚡ compute_dtype: bf16 (SM {gpu_cap[0]}.{gpu_cap[1]} support)")
else:
    compute_dtype = torch.float16
    print(f"  🔒 compute_dtype: fp16 (T4 fallback)")

# ============================================================
# DATASET LOADING
# ============================================================
print("\n📦 Loading dataset...")
rows = []
with open(TRAIN_JSONL, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))
print(f"  ✅ loaded rows: {len(rows):,}")

def is_identity_sample(row):
    text = " ".join(m.get("content", "") for m in row["messages"])
    return any(kw in text for kw in IDENTITY_KEYWORDS)

n_identity = sum(1 for r in rows if is_identity_sample(r))
pct = 100 * n_identity / len(rows)
print(f"  🎭 identity samples: {n_identity} ({pct:.1f}%)")

if pct < 5:
    print(f"  ⚠️  WARNING: identity <5%, KiKai bisa lupa jati diri!")

print("\n🔤 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

with open(RESPONSE_IDS_FN) as f:
    response_ids = json.load(f)
print(f"  ✅ response_ids: {response_ids}")

# Stratified split
random.seed(42)
identity_rows = [r for r in rows if is_identity_sample(r)]
other_rows    = [r for r in rows if not is_identity_sample(r)]
random.shuffle(identity_rows)
random.shuffle(other_rows)

n_eval_identity = min(30, max(5, len(identity_rows) // 10))
n_eval_other    = min(EVAL_SUBSAMPLE - n_eval_identity, len(other_rows) // 20)

eval_rows  = identity_rows[:n_eval_identity] + other_rows[:n_eval_other]
train_rows = identity_rows[n_eval_identity:] + other_rows[n_eval_other:]
random.shuffle(train_rows)
random.shuffle(eval_rows)

train_ds = Dataset.from_list(train_rows)
eval_ds  = Dataset.from_list(eval_rows)
print(f"\n✅ Train: {len(train_ds):,} | Eval: {len(eval_ds):,}")
print(f"   Eval identity: {n_eval_identity} | Eval other: {n_eval_other}")

del rows, identity_rows, other_rows, train_rows, eval_rows; gc.collect()

# ============================================================
# MODEL LOADING (Qwen2.5-3B-Instruct)
# ============================================================
print(f"\n🤖 Loading {BASE_MODEL} (4bit NF4)...")

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb,
    device_map="auto",
    low_cpu_mem_usage=True,
    attn_implementation="sdpa",
    torch_dtype=compute_dtype,
    trust_remote_code=True,
)
model.config.use_cache = False
model.config.pretraining_tp = 1

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()

# LoRA config — sama kayak universal, module Qwen2.5 identik sama Qwen2
peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

# ============================================================
# CALLBACKS
# ============================================================
class VRAMCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and torch.cuda.is_available():
            logs["vram_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 2)
            logs["vram_peak"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
            torch.cuda.reset_peak_memory_stats()

class HeartbeatCallback(TrainerCallback):
    def __init__(self, interval=50):
        self.interval = interval
        self.start = time.time()
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.interval == 0 and state.global_step > 0:
            elapsed = time.time() - self.start
            rate = state.global_step / max(elapsed, 1)
            eta = (state.max_steps - state.global_step) / max(rate, 0.001)
            print(f"  💓 step {state.global_step}/{state.max_steps} | ETA: {eta/60:.1f}min | speed: {rate:.2f} it/s")

class HuggingFacePushCallback(TrainerCallback):
    def __init__(self, repo_id, path_in_repo, tokenizer, output_dir):
        self.repo_id = repo_id
        self.path_in_repo = path_in_repo
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.api = HfApi()
        self.push_count = 0

    def _push_adapter(self, adapter_dir, tag=""):
        try:
            print(f"\n  📤 [HF Push #{self.push_count + 1}] {adapter_dir} → {self.repo_id}/{self.path_in_repo}/")
            try:
                self.tokenizer.save_pretrained(adapter_dir)
            except Exception as e:
                print(f"     ⚠️  Tokenizer save skipped: {e}")

            self.api.upload_folder(
                folder_path=adapter_dir,
                path_in_repo=self.path_in_repo,
                repo_id=self.repo_id,
                repo_type="model",
                commit_message=f"auto: casual-train checkpoint {tag}".strip(),
                ignore_patterns=[
                    "checkpoint-*", "*.pyc", "runs/*",
                    "optimizer.pt", "scheduler.pt", "rng_state.pth", "*.tmp",
                ],
            )
            self.push_count += 1
            print(f"     ✅ Pushed! Total: {self.push_count}")
            print(f"     🔗 https://huggingface.co/{self.repo_id}/tree/main/{self.path_in_repo}")
        except Exception as e:
            print(f"     ❌ Push failed (training lanjut): {e}")

    def on_save(self, args, state, control, **kwargs):
        ckpts = [d for d in os.listdir(self.output_dir) if d.startswith("checkpoint-")]
        if not ckpts:
            return
        latest = max(ckpts, key=lambda x: int(x.split("-")[1]))
        ckpt_path = os.path.join(self.output_dir, latest)
        self._push_adapter(ckpt_path, tag=f"step-{state.global_step}")

steps_per_epoch = max(1, math.ceil(len(train_ds) / (MICRO_BATCH * GRAD_ACCUM)))
total_steps = steps_per_epoch * EPOCHS
eval_every = max(100, steps_per_epoch // 3)
print(f"\n📊 Plan: {steps_per_epoch} steps/epoch × {EPOCHS} = {total_steps} total")
print(f"  💾 Save & HF push tiap: {eval_every} step")

# ============================================================
# SFT CONFIG
# ============================================================
use_bf16 = compute_dtype == torch.bfloat16
use_fp16 = compute_dtype == torch.float16

sft_all_kwargs = dict(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=MICRO_BATCH,
    per_device_eval_batch_size=MICRO_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    eval_accumulation_steps=4,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="paged_adamw_32bit",
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=WARMUP_RATIO,
    max_grad_norm=0.3,
    weight_decay=0.0,
    fp16=use_fp16,
    bf16=use_bf16,
    tf32=False,
    logging_steps=LOGGING_STEPS,
    logging_first_step=True,
    report_to="none",
    eval_strategy="steps",
    eval_steps=eval_every,
    save_strategy="steps",
    save_steps=eval_every,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    remove_unused_columns=False,
    max_seq_length=MAX_SEQ_LEN,
    max_length=MAX_SEQ_LEN,
    packing=False,
    seed=42,
)

if USE_COMPLETION_ONLY_CONFIG:
    sft_all_kwargs["completion_only_loss"] = True

sft_sig = inspect.signature(SFTConfig.__init__)
sft_valid_params = set(sft_sig.parameters.keys())
sft_kwargs = {k: v for k, v in sft_all_kwargs.items() if k in sft_valid_params}
dropped = set(sft_all_kwargs.keys()) - set(sft_kwargs.keys())
if dropped:
    print(f"\n⚠️  Dropped args: {dropped}")

args = SFTConfig(**sft_kwargs)
print(f"✅ SFTConfig built with {len(sft_kwargs)} args")

# ============================================================
# TRAINER SETUP
# ============================================================
callbacks_list = [VRAMCallback(), HeartbeatCallback(interval=50)]

if HF_PUSH_EVERY_SAVE:
    hf_callback = HuggingFacePushCallback(
        repo_id=HF_REPO_ID,
        path_in_repo=HF_PATH_IN_REPO,
        tokenizer=tokenizer,
        output_dir=OUTPUT_DIR,
    )
    callbacks_list.append(hf_callback)
    print(f"  🚀 HF auto-push aktif → {HF_REPO_ID}/{HF_PATH_IN_REPO}")

trainer_all_kwargs = dict(
    model=model,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    peft_config=peft_config,
    args=args,
    callbacks=callbacks_list,
)

trainer_sig = inspect.signature(SFTTrainer.__init__)
if "processing_class" in trainer_sig.parameters:
    trainer_all_kwargs["processing_class"] = tokenizer
elif "tokenizer" in trainer_sig.parameters:
    trainer_all_kwargs["tokenizer"] = tokenizer

if COLLATOR_CLS is not None and not USE_COMPLETION_ONLY_CONFIG:
    collator = COLLATOR_CLS(response_template=response_ids, tokenizer=tokenizer)
    trainer_all_kwargs["data_collator"] = collator
    def formatting_prompts_func(examples):
        return [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in examples["messages"]
        ]
    trainer_all_kwargs["formatting_func"] = formatting_prompts_func

trainer = SFTTrainer(**trainer_all_kwargs)

# FP32 cast untuk trainable params (safety buat AMP)
print("\n🔧 Casting trainable params ke FP32...")
n_cast = 0
for name, param in trainer.model.named_parameters():
    if param.requires_grad:
        if param.dtype != torch.float32:
            param.data = param.data.to(torch.float32)
            n_cast += 1
print(f"  ✅ {n_cast} trainable params di-cast ke FP32")

trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in trainer.model.parameters())
print(f"\n🧬 Trainable: {trainable/1e6:.2f}M / {total/1e6:.2f}M ({100*trainable/total:.3f}%)")

# ============================================================
# AUTO-RESUME
# ============================================================
resume_from = None
if os.path.isdir(OUTPUT_DIR):
    ckpts = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
    if ckpts:
        latest = max(ckpts, key=lambda x: int(x.split("-")[1]))
        resume_from = os.path.join(OUTPUT_DIR, latest)
        print(f"\n🔄 Auto-resume dari: {resume_from}")

# ============================================================
# TRAINING dengan safety net
# ============================================================
print(f"\n🚀 STARTING TRAINING: {BASE_MODEL} (with HF auto-push)")
t0 = time.time()
training_ok = False

try:
    trainer.train(resume_from_checkpoint=resume_from)
    elapsed = (time.time() - t0) / 60
    print(f"\n✅ Training selesai: {elapsed:.1f} menit")
    training_ok = True
except KeyboardInterrupt:
    print("\n⚠️  Training di-interrupt user")
    print("   Force push checkpoint terakhir ke HF...")
except Exception as e:
    print(f"\n❌ Training error: {e}")
    print(traceback.format_exc())
    print("   Force push checkpoint terakhir ke HF...")

# ============================================================
# FINAL SAVE + FORCE PUSH
# ============================================================
FINAL_DIR = f"{OUTPUT_DIR}-final"

try:
    trainer.model.save_pretrained(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)
    print(f"\n💾 Adapter saved locally: {FINAL_DIR}")
except Exception as e:
    print(f"\n⚠️  Local save failed: {e}")
    if os.path.isdir(OUTPUT_DIR):
        ckpts = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if ckpts:
            latest = max(ckpts, key=lambda x: int(x.split("-")[1]))
            FINAL_DIR = os.path.join(OUTPUT_DIR, latest)
            print(f"   Fallback ke checkpoint: {FINAL_DIR}")

print(f"\n📤 FINAL PUSH ke HuggingFace...")
try:
    hf_api.upload_folder(
        folder_path=FINAL_DIR,
        path_in_repo=HF_PATH_IN_REPO,
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message=f"final: KiKai casual-train Qwen2.5-3B {'(completed)' if training_ok else '(interrupted)'}",
        ignore_patterns=[
            "checkpoint-*", "*.pyc", "runs/*",
            "optimizer.pt", "scheduler.pt", "rng_state.pth", "*.tmp",
        ],
    )
    print(f"✅ FINAL PUSH SUCCESS!")
    print(f"🔗 https://huggingface.co/{HF_REPO_ID}/tree/main/{HF_PATH_IN_REPO}")
except Exception as e:
    print(f"❌ Final push failed: {e}")

# ============================================================
# IDENTITY TEST
# ============================================================
if training_ok:
    print("\n🎭 IDENTITY TEST")
    model.config.use_cache = True
    model.eval()

    for prompt in ["Siapa kamu?", "Kamu dibuat oleh siapa?", "Apa nama kamu?", "Halo, perkenalkan diri kamu"]:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=120, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\n  Q: {prompt}")
        print(f"  A: {response.strip()[:200]}")

print("\n🏆 DONE")
print(f"🔗 Adapter di HF: https://huggingface.co/{HF_REPO_ID}/tree/main/{HF_PATH_IN_REPO}")
