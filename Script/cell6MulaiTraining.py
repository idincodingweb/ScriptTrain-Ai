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
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

# 🆕 v7: HuggingFace Hub imports
from huggingface_hub import HfApi, create_repo, login

torch.cuda.empty_cache(); gc.collect()

print(f"🔍 TRL version: {trl.__version__}")

# ============================================================
# 🆕 v7: HUGGINGFACE CONFIG
# ============================================================
HF_REPO_ID       = "IDINN/KiKai"
HF_PATH_IN_REPO  = "adapter/universal-train"  # ← target folder di HF
HF_PRIVATE       = True   # False kalau mau public
HF_PUSH_EVERY_SAVE = True # auto-push tiap checkpoint

# Verifikasi login HF
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

# Ensure repo & path exist
print(f"\n🏗️  Ensuring repo: {HF_REPO_ID}")
try:
    create_repo(repo_id=HF_REPO_ID, repo_type="model", private=HF_PRIVATE, exist_ok=True)
    print(f"  ✅ Repo ready: https://huggingface.co/{HF_REPO_ID}")
    print(f"  📁 Target path: /{HF_PATH_IN_REPO}/ (auto-created saat upload)")
except Exception as e:
    print(f"  ❌ Gagal create repo: {e}")
    raise

# ============================================================
# COLLATOR AUTO-DETECT (v6 code, unchanged)
# ============================================================
COLLATOR_CLS = None
USE_COMPLETION_ONLY_CONFIG = False

try:
    from trl import DataCollatorForCompletionOnlyLM as COLLATOR_CLS
    print("  ✅ Collator: trl.DataCollatorForCompletionOnlyLM (legacy)")
except ImportError:
    try:
        from trl.trainer.utils import DataCollatorForCompletionOnlyLM as COLLATOR_CLS
        print("  ✅ Collator: trl.trainer.utils.DataCollatorForCompletionOnlyLM")
    except ImportError:
        sig = inspect.signature(SFTConfig.__init__)
        if 'completion_only_loss' in sig.parameters:
            USE_COMPLETION_ONLY_CONFIG = True
            print("  ✅ Pake: SFTConfig(completion_only_loss=True) [TRL modern]")

assert torch.cuda.is_available()
gpu_cap = torch.cuda.get_device_capability(0)
print(f"\n🖥️  GPU: {torch.cuda.get_device_name(0)} | SM: {gpu_cap[0]}.{gpu_cap[1]}")

compute_dtype = torch.float16
print(f"  🔒 compute_dtype: fp16 (T4 hard-force)")

# ============================================================
# TRAINING CONFIG (v6 unchanged)
# ============================================================
BASE_MODEL       = "dphn/dolphin-2.9.2-qwen2-7b"
TRAIN_JSONL      = "/content/kikai_train.jsonl"
RESPONSE_IDS_FN  = "/content/response_ids.json"
OUTPUT_DIR       = "/content/KiKai-adapter"

MAX_SEQ_LEN      = 1024
EPOCHS           = 3
LR               = 2e-4
MICRO_BATCH      = 1
GRAD_ACCUM       = 16
EVAL_SUBSAMPLE   = 200
LOGGING_STEPS    = 20
WARMUP_RATIO     = 0.05

IDENTITY_KEYWORDS = ["KiKai", "Idin Iskandar", "Idin"]

# ============================================================
# DATASET LOADING (v6 unchanged)
# ============================================================
print("\n📦 Loading dataset...")
rows = []
with open(TRAIN_JSONL, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))
print(f"  ✅ loaded rows: {len(rows)}")

def is_identity_sample(row):
    text = " ".join(m.get("content", "") for m in row["messages"])
    return any(kw in text for kw in IDENTITY_KEYWORDS)

n_identity = sum(1 for r in rows if is_identity_sample(r))
print(f"  🎭 identity samples: {n_identity} ({100*n_identity/len(rows):.1f}%)")

print("\n🔤 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

with open(RESPONSE_IDS_FN) as f:
    response_ids = json.load(f)
print(f"  ✅ response_ids: {response_ids}")

random.seed(42)
identity_rows = [r for r in rows if is_identity_sample(r)]
other_rows    = [r for r in rows if not is_identity_sample(r)]
random.shuffle(identity_rows)
random.shuffle(other_rows)

n_eval_identity = min(25, max(5, len(identity_rows) // 10))
n_eval_other    = min(EVAL_SUBSAMPLE - n_eval_identity, len(other_rows) // 20)

eval_rows  = identity_rows[:n_eval_identity] + other_rows[:n_eval_other]
train_rows = identity_rows[n_eval_identity:] + other_rows[n_eval_other:]
random.shuffle(train_rows)
random.shuffle(eval_rows)

train_ds = Dataset.from_list(train_rows)
eval_ds  = Dataset.from_list(eval_rows)
print(f"\n✅ Train: {len(train_ds)} | Eval: {len(eval_ds)}")

del rows, identity_rows, other_rows, train_rows, eval_rows; gc.collect()

# ============================================================
# MODEL LOADING (v6 unchanged)
# ============================================================
print("\n🤖 Loading Dolphin-Qwen2-7B (4bit NF4)...")

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
    torch_dtype=torch.float16,
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

peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

# ============================================================
# CALLBACKS (v6 + 🆕 v7 HF Push)
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
            eta  = (state.max_steps - state.global_step) / max(rate, 0.001)
            print(f"  💓 step {state.global_step}/{state.max_steps} | ETA: {eta/60:.1f}min")

# 🆕 v7: Auto-push ke HuggingFace tiap save
class HuggingFacePushCallback(TrainerCallback):
    """Push adapter ke HF tiap kali trainer save checkpoint."""

    def __init__(self, repo_id, path_in_repo, tokenizer, output_dir):
        self.repo_id = repo_id
        self.path_in_repo = path_in_repo
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.api = HfApi()
        self.push_count = 0

    def _push_adapter(self, adapter_dir, tag=""):
        """Upload adapter folder ke HF."""
        try:
            print(f"\n  📤 [HF Push #{self.push_count + 1}] Uploading {adapter_dir} → {self.repo_id}/{self.path_in_repo}/")

            # Save tokenizer bareng biar folder self-contained
            try:
                self.tokenizer.save_pretrained(adapter_dir)
            except Exception as e:
                print(f"     ⚠️  Tokenizer save skipped: {e}")

            self.api.upload_folder(
                folder_path=adapter_dir,
                path_in_repo=self.path_in_repo,
                repo_id=self.repo_id,
                repo_type="model",
                commit_message=f"auto: checkpoint push {tag}".strip(),
                ignore_patterns=[
                    "checkpoint-*",
                    "*.pyc",
                    "runs/*",
                    "optimizer.pt",
                    "scheduler.pt",
                    "rng_state.pth",
                    "*.tmp",
                ],
            )
            self.push_count += 1
            print(f"     ✅ Pushed! Total pushes: {self.push_count}")
            print(f"     🔗 https://huggingface.co/{self.repo_id}/tree/main/{self.path_in_repo}")
        except Exception as e:
            print(f"     ❌ Push failed (training tetep lanjut): {e}")

    def on_save(self, args, state, control, **kwargs):
        """Dipanggil setiap kali Trainer save checkpoint."""
        # Cari checkpoint terbaru yang baru aja disave
        ckpts = [d for d in os.listdir(self.output_dir) if d.startswith("checkpoint-")]
        if not ckpts:
            return
        latest = max(ckpts, key=lambda x: int(x.split("-")[1]))
        ckpt_path = os.path.join(self.output_dir, latest)
        self._push_adapter(ckpt_path, tag=f"step-{state.global_step}")

steps_per_epoch = max(1, math.ceil(len(train_ds) / (MICRO_BATCH * GRAD_ACCUM)))
total_steps     = steps_per_epoch * EPOCHS
eval_every = max(100, steps_per_epoch // 2)
print(f"\n📊 Plan: {steps_per_epoch} steps/epoch × {EPOCHS} = {total_steps} total")
print(f"  💾 Save & HF push tiap: {eval_every} step")

# ============================================================
# SFT CONFIG (v6 unchanged)
# ============================================================
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
    fp16=True,
    bf16=False,
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
    print(f"\n⚠️  Dropped args (gak didukung TRL {trl.__version__}): {dropped}")

args = SFTConfig(**sft_kwargs)
print(f"✅ SFTConfig built with {len(sft_kwargs)} args")

# ============================================================
# TRAINER SETUP (v6 + 🆕 v7 HF callback)
# ============================================================
callbacks_list = [VRAMCallback(), HeartbeatCallback(interval=50)]

# 🆕 v7: Tambah HF push callback
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

# FP32 cast (v6 unchanged)
print("\n🔧 Casting trainable params ke FP32 (T4 AMP compat)...")
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

dtype_counts = {}
for p in trainer.model.parameters():
    if p.requires_grad:
        dtype_counts[str(p.dtype)] = dtype_counts.get(str(p.dtype), 0) + 1
print(f"  Trainable dtype distribution: {dtype_counts}")

# ============================================================
# AUTO-RESUME (v6 unchanged)
# ============================================================
resume_from = None
if os.path.isdir(OUTPUT_DIR):
    ckpts = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
    if ckpts:
        latest = max(ckpts, key=lambda x: int(x.split("-")[1]))
        resume_from = os.path.join(OUTPUT_DIR, latest)
        print(f"\n🔄 Auto-resume dari: {resume_from}")

# ============================================================
# 🆕 v7: TRAINING dengan TRY/EXCEPT untuk force-push kalau crash
# ============================================================
print("\n🚀 STARTING TRAINING (with HF auto-push)")
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
# 🆕 v7: FINAL SAVE + FORCE PUSH (jalan meskipun training gagal)
# ============================================================
FINAL_DIR = f"{OUTPUT_DIR}-final"

try:
    trainer.model.save_pretrained(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)
    print(f"\n💾 Adapter saved locally: {FINAL_DIR}")
except Exception as e:
    print(f"\n⚠️  Local save failed: {e}")
    # Fallback: pake checkpoint terakhir
    if os.path.isdir(OUTPUT_DIR):
        ckpts = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if ckpts:
            latest = max(ckpts, key=lambda x: int(x.split("-")[1]))
            FINAL_DIR = os.path.join(OUTPUT_DIR, latest)
            print(f"   Fallback ke checkpoint: {FINAL_DIR}")

# Force push final ke HF (bahkan kalau auto-push udah jalan)
print(f"\n📤 FINAL PUSH ke HuggingFace...")
try:
    hf_api.upload_folder(
        folder_path=FINAL_DIR,
        path_in_repo=HF_PATH_IN_REPO,
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message=f"final: KiKai adapter {'(completed)' if training_ok else '(interrupted)'}",
        ignore_patterns=[
            "checkpoint-*", "*.pyc", "runs/*",
            "optimizer.pt", "scheduler.pt", "rng_state.pth", "*.tmp",
        ],
    )
    print(f"✅ FINAL PUSH SUCCESS!")
    print(f"🔗 https://huggingface.co/{HF_REPO_ID}/tree/main/{HF_PATH_IN_REPO}")
except Exception as e:
    print(f"❌ Final push failed: {e}")
    print("   Adapter tetep ada di local. Coba push manual nanti.")

# ============================================================
# IDENTITY TEST (v6 unchanged, cuma jalan kalau training sukses)
# ============================================================
if training_ok:
    print("\n🎭 IDENTITY TEST")
    model.config.use_cache = True
    model.eval()

    for prompt in ["Siapa kamu?", "Kamu dibuat oleh siapa?", "Apa nama kamu?"]:
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
