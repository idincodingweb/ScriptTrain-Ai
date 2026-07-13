# 🚀 ScriptTrain-Ai: Modular LLM Fine-Tuning Pipeline

Kumpulan *pipeline* skrip untuk *Supervised Fine-Tuning* (SFT) model AI menggunakan teknik LoRA (*Low-Rank Adaptation*) di atas infrastruktur GPU gratis (Google Colab/T4). Proyek ini difokuskan pada efisiensi *resource* untuk membangun persona model AI kustom (seperti KiKai v3) dengan biaya **nol rupiah**.

## 🛠️ Stack Teknologi
- **Framework:** PyTorch, Transformers, TRL (Transformer Reinforcement Learning)
- **Quantization:** BitsAndBytes (4-bit/NF4) untuk efisiensi VRAM
- **Adapter:** PEFT (Parameter-Efficient Fine-Tuning)
- **Deployment:** Hugging Face Hub (untuk *auto-checkpointing*)

## 📂 Struktur Pipeline
Pipeline ini dibagi menjadi beberapa modul yang dirancang untuk otomasi penuh:

* `cell1-2.txt`: Konfigurasi *environment* & instalasi *dependency*.
* `cell3-4.py`: Otentikasi & Verifikasi *repo* Hugging Face.
* `cell5.py`: *Data Preparation* (Parser otomatis untuk dataset `.jsonl` dan `.md`).
* `cell6.py`: *Main Training Engine* (SFT Trainer dengan *Auto-Push* ke Hugging Face).

## ⚡ Fitur Unggulan
1.  **Memory Optimization:** Integrasi konfigurasi `PYTORCH_CUDA_ALLOC_CONF` untuk mencegah *out-of-memory* pada GPU terbatas.
2.  **Auto-Checkpointing:** Skrip otomatis melakukan *push* ke Hugging Face tiap kali *checkpoint* selesai, mencegah kehilangan progress jika *runtime* terputus.
3.  **Identity Injection:** Mampu menyuntikkan persona kustom ke dalam model agar model memiliki *self-awareness* terhadap kreatornya.
4.  **Hardware Efficient:** Menggunakan teknik *Gradient Accumulation* dan *LoRA config* yang dioptimalkan untuk GPU T4.

## 🚀 Cara Penggunaan
1.  Pastikan lo punya *Hugging Face Token* dengan akses **WRITE**.
2.  Buka Notebook (Google Colab) dan jalankan skrip sesuai urutan modul (`cell1` s/d `cell6`).
3.  Pastikan parameter `HF_REPO_ID` di dalam `cell6.py` sudah mengarah ke repo model lo.
4.  Pantau *training progress* dan *auto-push* melalui terminal.

## ⚠️ Lisensi & Catatan
* Proyek ini ditujukan untuk edukasi dan pengembangan *custom persona AI*.
* Harap selalu perhatikan *terms of service* dari *base model* yang digunakan.
* *Built with Hustle & Efficiency by Idin Iskandar.*

---
*Stay modular, stay efficient.* 🤖🔥
