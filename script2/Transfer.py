import os
import shutil
from huggingface_hub import HfApi, snapshot_download, create_repo

# 1. Definisikan repo asal dan repo tujuan baru
REPO_ASAL = "IDINN/KiKai"
SUBFOLDER_ASAL = "adapter/universal-train"

REPO_TUJUAN = "IDINN/KiKai-Universal-7B"
LOKAL_DIR = "/content/kikai_universal_dir"

# Bersihkan sisa direktori lokal lama jika ada
if os.path.exists(LOKAL_DIR):
    shutil.rmtree(LOKAL_DIR)

print("⏳ Menghubungkan ke Hugging Face API...")
api = HfApi()

# 2. Bikin repo universal baru di HF (set private=True agar aman)
print(f"📁 Memastikan repo tujuan tersedia: {REPO_TUJUAN}")
create_repo(repo_id=REPO_TUJUAN, repo_type="model", private=True, exist_ok=True)

# 3. Download file adapter universal dari subfolder lama
print(f"📦 Mendownload file adapter dari {REPO_ASAL}/{SUBFOLDER_ASAL}...")
snapshot_download(
    repo_id=REPO_ASAL,
    allow_patterns=f"{SUBFOLDER_ASAL}/*",
    local_dir=LOKAL_DIR
)

# 4. Angkat semua file dari subfolder ke root folder lokal
path_lama = os.path.join(LOKAL_DIR, SUBFOLDER_ASAL)
print("🔄 Merapikan struktur file ke root directory...")
for file_name in os.listdir(path_lama):
    src = os.path.join(path_lama, file_name)
    dst = os.path.join(LOKAL_DIR, file_name)
    os.rename(src, dst)

# Hapus subfolder kosongnya
shutil.rmtree(path_lama)

# 5. Upload bersih ke root repo baru
print(f"🚀 Mengupload file ke repo baru: {REPO_TUJUAN}...")
api.upload_folder(
    folder_path=LOKAL_DIR,
    repo_id=REPO_TUJUAN,
    repo_type="model",
    commit_message="Migration: KiKai Universal Adapter moved to official universal repo"
)

print(f"\n🏆 TRANSFER BERHASIL, NYET! Lapak universal lo udah rapi di sini: https://huggingface.co/{REPO_TUJUAN}")
