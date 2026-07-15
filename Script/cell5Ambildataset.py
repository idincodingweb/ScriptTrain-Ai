# ============================================================
# KIKAI CASUAL-VIBE — DATA PREP v3 (Qwen2.5-3B-Instruct)
# ============================================================
import os, json, random
from collections import Counter
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

# ============================================================
# CONFIG
# ============================================================
HF_REPO_ID       = "IDINN/KiKai"
HF_DATA_FOLDER   = "casual-vibe"
BASE_MODEL       = "Qwen/Qwen2.5-3B-Instruct"  # 🆕 GANTI

OUT_TRAIN        = "/content/kikai_train.jsonl"
OUT_RESP_IDS     = "/content/response_ids.json"
LOCAL_CACHE_DIR  = "/content/kikai_casual_raw"

OVERSAMPLE_MAP = {
    "kikai_developer": 4,
    "kikai_identity":  4,
}

IDENTITY_KEYWORDS = ["KiKai", "Idin Iskandar", "Idin"]
SEED = 42

os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)
random.seed(SEED)

# ============================================================
# STEP 1: LIST FILES DI HF FOLDER
# ============================================================
print(f"🔍 Listing files di {HF_REPO_ID}/{HF_DATA_FOLDER}/ ...")
api = HfApi()
all_files = api.list_repo_files(repo_id=HF_REPO_ID, repo_type="model")
jsonl_files = [
    f for f in all_files
    if f.startswith(f"{HF_DATA_FOLDER}/") and f.endswith(".jsonl")
]

if not jsonl_files:
    raise FileNotFoundError(f"Gak ada file .jsonl di {HF_REPO_ID}/{HF_DATA_FOLDER}/")

print(f"  ✅ Ketemu {len(jsonl_files)} file JSONL:")
for f in jsonl_files:
    print(f"    - {f}")

# ============================================================
# STEP 2: DOWNLOAD
# ============================================================
print(f"\n📥 Downloading ke {LOCAL_CACHE_DIR}/ ...")
local_paths = []
for remote_path in jsonl_files:
    local = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=remote_path,
        repo_type="model",
        local_dir=LOCAL_CACHE_DIR,
    )
    local_paths.append(local)
    print(f"  ✅ {os.path.basename(local)}")

# ============================================================
# STEP 3: SMART PARSER (support 3 format)
# ============================================================
def parse_row(row):
    """Support: messages / conversation / instruction+response."""
    # Format 1: messages
    if "messages" in row and isinstance(row["messages"], list):
        msgs = []
        for m in row["messages"]:
            if not isinstance(m, dict): continue
            role = m.get("role", "").strip()
            content = m.get("content", "").strip()
            if role in ("system", "user", "assistant") and content:
                msgs.append({"role": role, "content": content})
        return msgs if len(msgs) >= 2 else None

    # Format 2: conversation
    if "conversation" in row and isinstance(row["conversation"], list):
        msgs = []
        for m in row["conversation"]:
            if not isinstance(m, dict): continue
            role = m.get("role") or m.get("from", "")
            content = m.get("content") or m.get("value", "")
            role = role.strip().lower()
            if role in ("human", "user"): role = "user"
            elif role in ("gpt", "assistant", "bot"): role = "assistant"
            elif role == "system": role = "system"
            else: continue
            content = content.strip()
            if content:
                msgs.append({"role": role, "content": content})
        return msgs if len(msgs) >= 2 else None

    # Format 3: instruction + response
    if "instruction" in row and "response" in row:
        instruction = str(row.get("instruction", "")).strip()
        inp = str(row.get("input", "")).strip()
        response = str(row.get("response", "")).strip()
        if not instruction or not response:
            return None
        user_content = f"{instruction}\n\n{inp}" if inp else instruction
        msgs = []
        sys_prompt = str(row.get("system", "")).strip()
        if sys_prompt:
            msgs.append({"role": "system", "content": sys_prompt})
        msgs.append({"role": "user", "content": user_content})
        msgs.append({"role": "assistant", "content": response})
        return msgs

    return None

# ============================================================
# STEP 4: LOAD + PARSE + OVERSAMPLE
# ============================================================
print(f"\n📦 Parsing & oversampling...")
all_rows = []
stats = Counter()

for local_path in local_paths:
    fname = os.path.basename(local_path).replace(".jsonl", "")

    multiplier = 1
    for key, mult in OVERSAMPLE_MAP.items():
        if key.lower() in fname.lower():
            multiplier = mult
            break

    file_rows = []
    with open(local_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line: continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"  ⚠️  {fname} line {line_no}: JSON error, skip")
                continue

            msgs = parse_row(row)
            if msgs is None:
                stats[f"{fname}:invalid"] += 1
                continue
            file_rows.append({"messages": msgs})

    original_count = len(file_rows)
    if multiplier > 1:
        file_rows = file_rows * multiplier
        print(f"  📈 {fname}: {original_count} × {multiplier} = {len(file_rows)}")
    else:
        print(f"  📄 {fname}: {original_count}")

    stats[f"{fname}:final"] = len(file_rows)
    all_rows.extend(file_rows)

random.shuffle(all_rows)
print(f"\n✅ Total training rows: {len(all_rows):,}")

# ============================================================
# STEP 5: IDENTITY CHECK
# ============================================================
def has_identity(row):
    text = " ".join(m.get("content", "") for m in row["messages"])
    return any(kw in text for kw in IDENTITY_KEYWORDS)

n_id = sum(1 for r in all_rows if has_identity(r))
pct = 100 * n_id / len(all_rows)
print(f"\n🎭 Identity samples: {n_id}/{len(all_rows)} ({pct:.1f}%)")

if pct < 10:
    print(f"  ⚠️  WARNING: identity samples <10%!")
    print(f"     Rekomendasi: tambah oversample kikai_developer.")
else:
    print(f"  ✅ Identity coverage cukup ({pct:.1f}%)")

# ============================================================
# STEP 6: WRITE TRAIN JSONL
# ============================================================
print(f"\n💾 Writing {OUT_TRAIN} ...")
with open(OUT_TRAIN, "w", encoding="utf-8") as f:
    for row in all_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"  ✅ {len(all_rows):,} rows written")

# ============================================================
# STEP 7: RESPONSE IDS (Qwen2.5 native ChatML)
# ============================================================
print(f"\n🔤 Computing response_ids for {BASE_MODEL}...")
tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, trust_remote_code=True)

# Qwen2.5 pake ChatML: <|im_start|>assistant\n
ASSISTANT_MARKER = "<|im_start|>assistant\n"
response_ids = tok.encode(ASSISTANT_MARKER, add_special_tokens=False)
print(f"  Marker: {repr(ASSISTANT_MARKER)}")
print(f"  Encoded IDs: {response_ids}")
print(f"  Decoded back: {repr(tok.decode(response_ids))}")

# Validasi
sample_msgs = [
    {"role": "user", "content": "test"},
    {"role": "assistant", "content": "test response"},
]
rendered = tok.apply_chat_template(sample_msgs, tokenize=False, add_generation_prompt=False)
print(f"\n  📝 Sample chat template:")
print(f"  {repr(rendered[:300])}")

if ASSISTANT_MARKER not in rendered:
    print(f"  ❌ ERROR: marker gak muncul!")
    raise ValueError("Chat template Qwen2.5 gak sesuai ekspektasi")

full_ids = tok.encode(rendered, add_special_tokens=False)

def find_subseq(haystack, needle):
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i+len(needle)] == needle:
            return i
    return -1

pos = find_subseq(full_ids, response_ids)
if pos == -1:
    print(f"  ⚠️  WARNING: subsequence not found, using fallback...")
    pre_ids = tok.encode(rendered.split("test response")[0], add_special_tokens=False)
    response_ids = pre_ids[-5:]
    print(f"  Fallback IDs: {response_ids}")
else:
    print(f"  ✅ Marker validated at position {pos}")

with open(OUT_RESP_IDS, "w") as f:
    json.dump(response_ids, f)
print(f"  ✅ Saved: {OUT_RESP_IDS}")

# ============================================================
# STEP 8: SUMMARY
# ============================================================
print(f"\n{'='*60}")
print(f"📊 DATA PREP SUMMARY (Qwen2.5-3B-Instruct)")
print(f"{'='*60}")
print(f"  Base model       : {BASE_MODEL}")
print(f"  Total rows       : {len(all_rows):,}")
print(f"  Identity coverage: {pct:.1f}%")
print(f"  Steps/epoch (bs=16): ~{len(all_rows)//16}")
print(f"  Total steps × 2  : ~{2 * len(all_rows)//16}")
print(f"  Estimasi (0.15 it/s): ~{(2 * len(all_rows)//16) / 0.15 / 60:.1f} menit")
print(f"{'='*60}")
print(f"\n🏆 READY! Lanjut ke Cell 2 (training).")
