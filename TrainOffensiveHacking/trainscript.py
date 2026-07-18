import os, gc, json, math, time, random, warnings, subprocess
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
warnings.filterwarnings("ignore")

import torch
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
from huggingface_hub import HfApi, create_repo, login

torch.cuda.empty_cache(); gc.collect()

# =====================================================================
# ADJUSTED PATHS: MURNI IDENTITY + SKILLS
# =====================================================================
GIT_REPO_URL     = "https://github.com/idincodingweb/Dataset-For-Ai-Engineer.git"
BASE_MODEL       = "DeepHat/DeepHat-V1-7B"

OUT_TRAIN        = "/content/kikai_train.jsonl"
RESPONSE_IDS_FN  = "/content/response_ids.json"
OUTPUT_DIR       = "/content/KiKai-casual-v2-adapter"
GIT_CLONE_DIR    = "/content/dataset_github_repo"

# ============================================================
# SUPER SAFE MODE UNTUK GPU T4 (MAX_SEQ_LEN = 256 WAJIB!)
# ============================================================
MAX_SEQ_LEN     = 256   # ⚠️ Jangan dinaikin ke 512, dataset skills lo bikin jebol VRAM!
EPOCHS          = 1
LR              = 2e-4
MICRO_BATCH     = 2
GRAD_ACCUM      = 4
LOGGING_STEPS   = 10
WARMUP_RATIO    = 0.05
IDENTITY_TEST_EVERY = 100

HF_REPO_ID      = "IDINN/KiKai"
HF_PATH_IN_REPO = "adapter/deephat-offensive"
HF_PRIVATE      = True

# Kembalikan ke 2 agar porsi 104 baris identity gak tenggelam oleh ribuan baris skills
IDENTITY_OVERSAMPLE    = 2
IDENTITY_KEYWORDS      = ["kikai", "idin iskandar", "idin"]

MIN_IDENTITY_PCT = 10.0
MAX_IDENTITY_PCT = 18.0

IDENTITY_TEST_PROMPTS = [
    "Siapa kamu?",
    "Kamu dibuat oleh siapa?",
    "Nama kamu apa?",
]
IDENTITY_MUST_CONTAIN = ["KiKai", "kikai", "Idin Iskandar"]
IDENTITY_MUST_NOT_CONTAIN = ["Qwen", "Alibaba", "qwen"]

SEED = 42
os.makedirs(os.path.dirname(OUT_TRAIN), exist_ok=True)
random.seed(SEED)

# ---------- LIST + DOWNLOAD FROM GITHUB ----------
print(f"🔍 Cloning GitHub Repository from {GIT_REPO_URL}...")
if not os.path.exists(GIT_CLONE_DIR):
    subprocess.run(["git", "clone", GIT_REPO_URL, GIT_CLONE_DIR], check=True)
else:
    print("🔄 Repo already cloned, pulling latest changes...")
    subprocess.run(["git", "-C", GIT_CLONE_DIR, "pull"], check=True)

identity_target_file = os.path.join(GIT_CLONE_DIR, "security/dataset/identity/identity.jsonl")
skills_target_dir = os.path.join(GIT_CLONE_DIR, "security/dataset/skills")

target_files = []
if os.path.exists(identity_target_file):
    target_files.append((identity_target_file, "identity"))

if os.path.exists(skills_target_dir):
    for root, _, files in os.walk(skills_target_dir):
        for file in files:
            if file.endswith(".jsonl"):
                target_files.append((os.path.join(root, file), "skills"))

assert target_files, "Tidak ada berkas dataset target yang ditemukan!"
print(f"✅ Total berkas yang akan diproses: {len(target_files)} file.")

# ---------- PARSER KEMBALI KE JSONL MURNI ----------
def parse_row(row):
    if "messages" in row and isinstance(row["messages"], list):
        msgs = [{"role": m.get("role","").strip(), "content": m.get("content","").strip()}
                for m in row["messages"] if isinstance(m, dict)
                and m.get("role","").strip() in ("system","user","assistant")
                and m.get("content","").strip()]
        return msgs if len(msgs) >= 2 else None
    return None

def has_identity_content(msgs):
    text = " ".join(m.get("content","") for m in msgs).lower()
    return any(kw.lower() in text for kw in IDENTITY_KEYWORDS)

all_rows = []
for path, category in target_files:
    fname = os.path.basename(path)
    mult = IDENTITY_OVERSAMPLE if category == "identity" else 1

    file_rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                row = json.loads(line)
                msgs = parse_row(row)
                if msgs: file_rows.append({"messages": msgs})
            except: continue

    original = len(file_rows)
    if mult > 1:
        file_rows = file_rows * mult
        print(f"  📈 [{category.upper()}] {fname}: {original} × {mult} = {len(file_rows)} (OVERSAMPLED)")
    else:
        print(f"  📄 [{category.upper()}] {fname}: {original} baris")

    all_rows.extend(file_rows)

random.shuffle(all_rows)

# ---------- IDENTITY CHECK ----------
n_id = sum(1 for r in all_rows if has_identity_content(r["messages"]))
pct = 100 * n_id / len(all_rows) if all_rows else 0
print(f"\n🎭 Identity coverage: {n_id}/{len(all_rows)} ({pct:.2f}%)")

if pct < MIN_IDENTITY_PCT:
    print(f"❌ HARD STOP: {pct:.2f}% < {MIN_IDENTITY_PCT}% (Kurang dominan)")
    raise SystemExit("Naikin IDENTITY_OVERSAMPLE sedikit.")
elif pct > MAX_IDENTITY_PCT:
    print(f"❌ HARD STOP: {pct:.2f}% > {MAX_IDENTITY_PCT}% (Terlalu besar, resiko ngerusak data hacking!)")
    raise SystemExit("Turunkan IDENTITY_OVERSAMPLE.")

print(f"✅ Identity Guard: BALANCED ({pct:.2f}%)")

with open(OUT_TRAIN, "w", encoding="utf-8") as f:
    for r in all_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"💾 Saved target training data to: {OUT_TRAIN}")

# ---------- HF LOGIN & REPO CREATION ----------
try:
    hf_api = HfApi(); user = hf_api.whoami()
    print(f"✅ HF API Connect: {user['name']}")
except:
    login(); hf_api = HfApi()

create_repo(repo_id=HF_REPO_ID, repo_type="model", private=HF_PRIVATE, exist_ok=True)

# ---------- FIX TOKENIZER & RESPONSE IDS ----------
print(f"\n⚙️ Extracting exact response IDs for base model {BASE_MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

test_messages = [{"role": "user", "content": "init_x"}, {"role": "assistant", "content": "target_y"}]
templated_str = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=False)
target_marker = templated_str.split("target_y")[0]
response_ids = tokenizer.encode(target_marker, add_special_tokens=False)[-3:]

with open(RESPONSE_IDS_FN, "w") as f: json.dump(response_ids, f)
print(f"✅ Precision response_ids for DeepHat: {response_ids}")

# ---------- PREPARE DATASET FOR TRAINER ----------
id_rows    = [r for r in all_rows if has_identity_content(r["messages"])]
other_rows = [r for r in all_rows if not has_identity_content(r["messages"])]
random.shuffle(id_rows); random.shuffle(other_rows)

eval_id_len = min(30, len(id_rows) // 2)
eval_other_len = min(270, len(other_rows) // 2)

eval_rows  = id_rows[:eval_id_len] + other_rows[:eval_other_len]
train_rows = id_rows[eval_id_len:] + other_rows[eval_other_len:]
random.shuffle(train_rows)

train_ds = Dataset.from_list(train_rows)
eval_ds  = Dataset.from_list(eval_rows)
print(f"Train Dataset: {len(train_ds):,} rows | Eval Dataset: {len(eval_ds):,} rows")

del all_rows, id_rows, other_rows, train_rows, eval_rows; gc.collect()

# ---------- LOAD QUANTIZED MODEL ----------
gpu_cap = torch.cuda.get_device_capability(0)
compute_dtype = torch.bfloat16 if gpu_cap[0] >= 8 else torch.float16
print(f"🖥️  Running on {torch.cuda.get_device_name(0)} | Precision Architecture: {compute_dtype}")

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb, device_map="auto",
    low_cpu_mem_usage=True, attn_implementation="sdpa",
    torch_dtype=compute_dtype, trust_remote_code=True)
model.config.use_cache = False
model.config.pretraining_tp = 1

model = prepare_model_for_kbit_training(
    model, use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False})
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()

peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])

# ---------- CALLBACKS ----------
class IdentityGuardCallback(TrainerCallback):
    def __init__(self, tokenizer, every=100, min_pass_after_step=200):
        self.tok = tokenizer
        self.every = every
        self.min_pass = min_pass_after_step
        self.last_pass = 0

    def on_step_end(self, args, state, control, model=None, **kw):
        if state.global_step == 0 or state.global_step % self.every != 0:
            return
        if model is None: return

        model.eval()
        pass_count = 0
        results = []
        with torch.no_grad():
            for prompt in IDENTITY_TEST_PROMPTS:
                text = self.tok.apply_chat_template(
                    [{"role":"user","content":prompt}],
                    tokenize=False, add_generation_prompt=True)
                inputs = self.tok(text, return_tensors="pt").to(model.device)
                out = model.generate(
                    **inputs, max_new_tokens=80, do_sample=False,
                    pad_token_id=self.tok.eos_token_id)
                resp = self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

                has_good = any(k in resp for k in IDENTITY_MUST_CONTAIN)
                has_bad  = any(k in resp for k in IDENTITY_MUST_NOT_CONTAIN)
                ok = has_good and not has_bad
                if ok: pass_count += 1
                results.append((prompt, resp[:100], "✅" if ok else "❌"))
        model.train()

        print(f"\n  🎭 [Step {state.global_step}] Identity check: {pass_count}/{len(IDENTITY_TEST_PROMPTS)}")
        for p, r, s in results:
            print(f"     {s} Q: {p} → {r.strip()[:80]}")

class HFPushCallback(TrainerCallback):
    def __init__(self, repo_id, path, tok, out_dir):
        self.repo_id, self.path, self.tok, self.out_dir = repo_id, path, tok, out_dir
        self.api = HfApi(); self.n = 0
    def on_save(self, args, state, control, **kw):
        ckpts = [d for d in os.listdir(self.out_dir) if d.startswith("checkpoint-")]
        if not ckpts: return
        latest = max(ckpts, key=lambda x: int(x.split("-")[1]))
        ckpt = os.path.join(self.out_dir, latest)
        try:
            self.tok.save_pretrained(ckpt)
            self.api.upload_folder(
                folder_path=ckpt, path_in_repo=self.path,
                repo_id=self.repo_id, repo_type="model",
                commit_message=f"auto: step-{state.global_step} (skills injected)",
                ignore_patterns=["checkpoint-*","optimizer.pt","scheduler.pt","rng_state.pth"])
            self.n += 1
            print(f"     📤 Pushed #{self.n} → {self.repo_id}/{self.path}")
        except Exception as e:
            print(f"     ⚠️ Push failed: {e}")

# ---------- COLLATOR & CONFIG ----------
from trl import DataCollatorForCompletionOnlyLM
collator = DataCollatorForCompletionOnlyLM(response_template=response_ids, tokenizer=tokenizer)

steps_per_epoch = max(1, math.ceil(len(train_ds) / (MICRO_BATCH * GRAD_ACCUM)))
total_steps = steps_per_epoch * EPOCHS
eval_every  = max(20, steps_per_epoch // 3)
print(f"\n📊 steps/epoch: {steps_per_epoch} | total: {total_steps} | eval every: {eval_every}")

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=MICRO_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",
    logging_steps=LOGGING_STEPS,
    save_strategy="steps",
    save_steps=eval_every,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=eval_every,
    bf16=(compute_dtype == torch.bfloat16),
    fp16=(compute_dtype == torch.float16),
    max_seq_length=MAX_SEQ_LEN,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="none",
    seed=42,
    dataset_kwargs={"skip_prepare_dataset": False},
)

trainer = SFTTrainer(
    model=model, args=sft_config,
    train_dataset=train_ds, eval_dataset=eval_ds,
    peft_config=peft_config, data_collator=collator,
    callbacks=[
        IdentityGuardCallback(tokenizer, every=IDENTITY_TEST_EVERY),
        HFPushCallback(HF_REPO_ID, HF_PATH_IN_REPO, tokenizer, OUTPUT_DIR),
    ],
)

# ---------- TRAIN ----------
print("\n🚀 START TRAINING WORKFLOW (IDENTITY & SKILLS INJECTION)")
trainer.train(resume_from_checkpoint=False)

# ---------- FINAL SAVE & PUSH ----------
print("\n💾 Saving final adapter locally...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n📤 Executing final push → {HF_REPO_ID}/{HF_PATH_IN_REPO}")
HfApi().upload_folder(
    folder_path=OUTPUT_DIR, path_in_repo=HF_PATH_IN_REPO,
    repo_id=HF_REPO_ID, repo_type="model",
    commit_message="final: identity + skills injected",
    ignore_patterns=["checkpoint-*","optimizer.pt","scheduler.pt","rng_state.pth"])

print(f"\n🏆 SUCCESSFUL → https://huggingface.co/{HF_REPO_ID}/tree/main/{HF_PATH_IN_REPO}")
