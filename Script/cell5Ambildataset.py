# =============================================================================
# CELL 5 : Data Prep FIXED — Enhanced Parser + Verbose Logging
# Target: ~6500 rows dari 64 JSONL + 19 MD
# =============================================================================
import os, json, re, gc
from collections import Counter
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer
from tqdm.auto import tqdm

REPO_ID     = "IDINN/KiKai"
REPO_TYPE   = "model"
BASE_MODEL  = "dphn/dolphin-2.9.2-qwen2-7b"
MAX_SEQ_LEN = 1024

# Path filter — ambil SEMUA dari dataset-universal
JSONL_PREFIX = "dataset-universal/"
MD_PREFIX    = "dataset-universal/knowledge/"

OVERSAMPLE = {
    "universal_persona": 3,
    "kikai_developer":   4,
    "casual_chat":       2,
    "batas_pengetahuan": 3,
}

# ============================================================
# ENHANCED MD PARSER
# ============================================================
def md_to_chat_samples(md_text: str, source_file: str):
    md_text = md_text.strip()
    if not md_text:
        return []

    samples = []
    fname_clean = os.path.basename(source_file).replace(".md", "").replace("_", " ")

    # Split by H1/H2 dulu
    sections = re.split(r'\n(?=#{1,2}\s)', md_text)

    # Handle first chunk kalau gak diawali #
    if sections and not sections[0].strip().startswith("#"):
        first = sections[0].strip()
        if len(first) >= 30:
            samples.append({
                "messages": [
                    {"role": "user", "content": f"Jelaskan tentang {fname_clean}."},
                    {"role": "assistant", "content": first[:2500]},
                ]
            })
        sections = sections[1:]

    for sec in sections:
        sec = sec.strip()
        if len(sec) < 30:
            continue

        m = re.match(r'^(#{1,2})\s+(.+?)(?:\n|$)', sec)
        if not m:
            continue

        heading = m.group(2).strip()
        body_full = sec[m.end():].strip()

        if len(body_full) < 20:
            continue

        # Cek sub-heading H3
        subsections = re.split(r'\n(?=#{3,4}\s)', body_full)

        if len(subsections) > 1:
            # Main body
            main_body = subsections[0].strip()
            if len(main_body) >= 20:
                body_trunc = main_body[:2500].rsplit(".", 1)[0] + "." if len(main_body) > 2500 else main_body
                samples.append({
                    "messages": [
                        {"role": "user", "content": f"Jelaskan tentang {heading}."},
                        {"role": "assistant", "content": body_trunc},
                    ]
                })

            # Subsections
            for sub in subsections[1:]:
                sub = sub.strip()
                sm = re.match(r'^(#{3,4})\s+(.+?)(?:\n|$)', sub)
                if not sm:
                    continue
                sub_heading = sm.group(2).strip()
                sub_body = sub[sm.end():].strip()
                if len(sub_body) < 20:
                    continue
                sub_body_trunc = sub_body[:2000].rsplit(".", 1)[0] + "." if len(sub_body) > 2000 else sub_body
                samples.append({
                    "messages": [
                        {"role": "user", "content": f"Jelaskan tentang {sub_heading} dalam konteks {heading}."},
                        {"role": "assistant", "content": sub_body_trunc},
                    ]
                })
        else:
            body_trunc = body_full[:2500].rsplit(".", 1)[0] + "." if len(body_full) > 2500 else body_full
            samples.append({
                "messages": [
                    {"role": "user", "content": f"Jelaskan tentang {heading}."},
                    {"role": "assistant", "content": body_trunc},
                ]
            })

    # Fallback
    if not samples and len(md_text) >= 100:
        body_trunc = md_text[:2500].rsplit(".", 1)[0] + "." if len(md_text) > 2500 else md_text
        samples.append({
            "messages": [
                {"role": "user", "content": f"Jelaskan tentang {fname_clean}."},
                {"role": "assistant", "content": body_trunc},
            ]
        })

    return samples

# ============================================================
# LIST FILES — ambil SEMUA dari dataset-universal
# ============================================================
print("📡 Fetching file list dari HF Hub...")
api = HfApi()
all_files = api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)

jsonl_files = [f for f in all_files
               if f.endswith(".jsonl") and f.startswith(JSONL_PREFIX)]
md_files    = [f for f in all_files
               if f.endswith(".md") and f.startswith(MD_PREFIX)
               and not f.endswith("INDEX.md") and not f.endswith("README.md")]

print(f"📂 JSONL files : {len(jsonl_files)}")
print(f"📂 MD files    : {len(md_files)}")

if not jsonl_files and not md_files:
    raise ValueError("STOP: gak ada file ketemu.")

# ============================================================
# PARSE JSONL
# ============================================================
print("\n📥 Parsing JSONL files...")
rows = []
per_source = Counter()
skipped = Counter()
failed_downloads = []

for f in tqdm(jsonl_files, desc="JSONL", ncols=80):
    try:
        local = hf_hub_download(repo_id=REPO_ID, filename=f, repo_type=REPO_TYPE)
    except Exception as e:
        failed_downloads.append((f, str(e)))
        continue

    fname = os.path.basename(f).replace(".jsonl", "")
    mult = OVERSAMPLE.get(fname, 1)

    try:
        with open(local, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip().rstrip(",")
                if not line or line in ("[", "]"):
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    skipped[f"{fname}:json_error"] += 1
                    continue
                if "messages" not in obj or not isinstance(obj["messages"], list):
                    skipped[f"{fname}:no_messages"] += 1
                    continue

                msgs = obj["messages"]
                roles = [m.get("role") for m in msgs if isinstance(m, dict)]
                if "user" not in roles or "assistant" not in roles:
                    skipped[f"{fname}:missing_role"] += 1
                    continue

                valid = all(
                    isinstance(m, dict) and "role" in m and "content" in m
                    and m["role"] in {"system", "user", "assistant"}
                    and isinstance(m["content"], str) and m["content"].strip()
                    for m in msgs
                )
                if not valid:
                    skipped[f"{fname}:malformed"] += 1
                    continue

                for _ in range(mult):
                    rows.append({"messages": msgs})
                per_source[f"jsonl/{fname}"] += mult
    except Exception as e:
        skipped[f"{fname}:read_error"] += 1

print(f"  ✅ JSONL parsed: {len(rows)} rows (dengan oversample)")

# ============================================================
# PARSE MD dengan ENHANCED parser
# ============================================================
print("\n📥 Parsing MD files (ENHANCED)...")
for f in tqdm(md_files, desc="MD", ncols=80):
    try:
        local = hf_hub_download(repo_id=REPO_ID, filename=f, repo_type=REPO_TYPE)
    except Exception as e:
        failed_downloads.append((f, str(e)))
        continue

    try:
        with open(local, "r", encoding="utf-8") as fh:
            md_text = fh.read()
        samples = md_to_chat_samples(md_text, f)
        rows.extend(samples)
        folder = f.split("/")[2] if len(f.split("/")) > 2 else "md"
        per_source[f"md/{folder}"] += len(samples)
    except Exception as e:
        skipped[f"md/{os.path.basename(f)}:read_error"] += 1

print(f"  ✅ Total rows: {len(rows)}")

# ============================================================
# REPORT
# ============================================================
print("\n" + "=" * 60)
print(f"TOTAL SAMPLE: {len(rows)}")
print("=" * 60)
for src, n in per_source.most_common(25):
    print(f"  {src:<50} {n:>5}  ({100*n/len(rows):5.1f}%)")

if failed_downloads:
    print(f"\n⚠️  FAILED DOWNLOADS ({len(failed_downloads)}):")
    for fname, err in failed_downloads[:5]:
        print(f"  {fname}: {err[:80]}")

if skipped:
    print(f"\n⚠️  SKIPPED SAMPLES ({sum(skipped.values())}):")
    for k, v in skipped.most_common(10):
        print(f"  {k:<40} {v}")

if not rows:
    raise ValueError("STOP: 0 sample valid.")

# ============================================================
# TOKENIZER + RESPONSE_IDS
# ============================================================
print("\n🔤 Loading tokenizer & extracting response_ids...")
tok = AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

if not tok.chat_template:
    raise ValueError("❌ Tokenizer gak punya chat_template.")

rendered = tok.apply_chat_template(rows[0]["messages"], tokenize=False,
                                   add_generation_prompt=False)
print("\n[ SAMPLE #0 RENDERED (first 500 chars) ]")
print("-" * 60)
print(rendered[:500])
print("-" * 60)

marker_text = "<|im_start|>assistant\n"
marker_pos = rendered.find(marker_text)
if marker_pos == -1:
    raise ValueError("❌ marker '<|im_start|>assistant' gak ada.")

prefix_ids = tok.encode(rendered[:marker_pos + len(marker_text)],
                        add_special_tokens=False)
resp_ids = prefix_ids[-3:]

full_ids = tok.encode(rendered, add_special_tokens=False)

def find_sub(hay, needle):
    for i in range(len(hay) - len(needle) + 1):
        if hay[i:i+len(needle)] == needle:
            return i
    return -1

if find_sub(full_ids, resp_ids) == -1:
    raise ValueError("❌ response_ids gak match — STOP.")

print(f"\n✅ response_ids : {resp_ids}")
print(f"   decoded     : {repr(tok.decode(resp_ids))}")

# ============================================================
# TOKEN LENGTH — sampling
# ============================================================
print("\n📏 Analyzing token lengths...")
import random
random.seed(42)
sample_for_stats = rows if len(rows) <= 3000 else random.sample(rows, 3000)
print(f"   Analyzing {len(sample_for_stats)} samples...")

lengths = []
for r in tqdm(sample_for_stats, desc="Length", ncols=80):
    r_text = tok.apply_chat_template(r["messages"], tokenize=False,
                                     add_generation_prompt=False)
    lengths.append(len(tok.encode(r_text, add_special_tokens=False)))
lengths.sort()

def pct(q): return lengths[min(int(len(lengths)*q), len(lengths)-1)]
over = sum(1 for l in lengths if l > MAX_SEQ_LEN)
print(f"\n[ TOKEN LEN ] min={lengths[0]} p50={pct(0.5)} p90={pct(0.9)} "
      f"p99={pct(0.99)} max={lengths[-1]}")
print(f"MAX_SEQ_LEN={MAX_SEQ_LEN} → truncated: {over}/{len(lengths)} "
      f"({100*over/len(lengths):.1f}%)")

# ============================================================
# SAVE
# ============================================================
print("\n💾 Saving...")
TRAIN_JSONL = "/content/kikai_train.jsonl"
with open(TRAIN_JSONL, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")

with open("/content/response_ids.json", "w") as f:
    json.dump(resp_ids, f)

train_size_mb = os.path.getsize(TRAIN_JSONL) / 1024**2
print(f"\n✅ Saved: {TRAIN_JSONL} ({len(rows)} rows, {train_size_mb:.1f} MB)")
print(f"✅ Saved: /content/response_ids.json")

# ============================================================
# ESTIMASI TRAINING
# ============================================================
MICRO_BATCH = 1
GRAD_ACCUM = 16
EPOCHS = 3
steps_per_epoch = (len(rows) + MICRO_BATCH * GRAD_ACCUM - 1) // (MICRO_BATCH * GRAD_ACCUM)
total_steps = steps_per_epoch * EPOCHS
est_hours_low = total_steps * 3 / 3600
est_hours_high = total_steps * 5 / 3600

print("\n" + "=" * 60)
print("📊 ESTIMASI TRAINING DI T4")
print("=" * 60)
print(f"  Total sample     : {len(rows)}")
print(f"  Steps per epoch  : {steps_per_epoch}")
print(f"  Total steps      : {total_steps} ({EPOCHS} epochs)")
print(f"  Estimasi waktu   : {est_hours_low:.1f} - {est_hours_high:.1f} jam")
print("=" * 60)

del rows, lengths, sample_for_stats
gc.collect()
print("\n🚀 Ready untuk Cell 4")
