import os
from huggingface_hub import HfApi, snapshot_download, create_repo

# 1. Definisikan repo asal dan repo tujuan
REPO_ASAL = "IDINN/KiKai"
SUBFOLDER_ASAL = "adapter/deephat-offensive"

REPO_TUJUAN = "IDINN/KiKai-For-Hacking-7.6B"
LOKAL_DIR = "/content/kikai_transfer_dir"

print("⏳ Menghubungkan ke Hugging Face API...")
api = HfApi()

# 2. Bikin repo baru di HF (kalau belum ada) set sebagai Private biar aman
print(f"📁 Memastikan repo tujuan tersedia: {REPO_TUJUAN}")
create_repo(repo_id=REPO_TUJUAN, repo_type="model", private=True, exist_ok=True)

# 3. Download semua file dari subfolder repo lama ke lokal Colab
print(f"📦 Mendownload file adapter dari {REPO_ASAL}/{SUBFOLDER_ASAL}...")
snapshot_download(
    repo_id=REPO_ASAL,
    allow_patterns=f"{SUBFOLDER_ASAL}/*",
    local_dir=LOKAL_DIR
)

# 4. Pindahkan file dari dalam subfolder ke root folder lokal biar struktur reponya bersih
path_lama = os.path.join(LOKAL_DIR, SUBFOLDER_ASAL)
print("🔄 Merapikan struktur file...")
for file_name in os.listdir(path_lama):
    src = os.path.join(path_lama, file_name)
    dst = os.path.join(LOKAL_DIR, file_name)
    os.rename(src, dst)

# Hapus sisa subfolder kosong biar gak ikut keupload
os.removedirs(path_lama)

# 5. Upload bersih langsung ke root repo tujuan
print(f"🚀 Mengupload file ke repo baru: {REPO_TUJUAN}...")
api.upload_folder(
    folder_path=LOKAL_DIR,
    repo_id=REPO_TUJUAN,
    repo_type="model",
    commit_message="Migration: KiKai Offensive Adapter moved to official hacking repo"
)

print(f"\n🏆 TRANSFER BERHASIL, DIN! Cek reponya disini bro: https://huggingface.co/{REPO_TUJUAN}")
