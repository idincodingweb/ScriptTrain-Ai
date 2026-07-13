# =============================================================================
# CELL 2: HuggingFace Login + Verifikasi Access
# =============================================================================
from huggingface_hub import login, whoami, HfApi

login()  # paste HF token dengan WRITE access

try:
    user_info = whoami()
    print(f"\n✅ Login sebagai: {user_info['name']}")

    api = HfApi()
    files = api.list_repo_files(repo_id="IDINN/KiKai", repo_type="model")
    print(f"✅ Akses ke IDINN/KiKai: {len(files)} files terdeteksi")

    jsonl_count = sum(1 for f in files if f.endswith(".jsonl"))
    md_count = sum(1 for f in files if f.endswith(".md"))
    print(f"   → {jsonl_count} JSONL files")
    print(f"   → {md_count} MD files")

except Exception as e:
    print(f"\n❌ Login/Access error: {e}")
    raise

print("\n🚀 Ready untuk Cell 3 (data prep)")
