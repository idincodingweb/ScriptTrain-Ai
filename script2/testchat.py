from threading import Thread
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
from peft import PeftModel  # <-- TAMBAHAN WAJIB buat ngeload adapter LoRA

# ============================================================
# AUTO-LOADER & ADAPTER INJECTOR
# ============================================================
if 'tokenizer' not in globals() or 'model' not in globals():
    print("⏳ Setup memori belum ke-detect. Loading otomatis...")

    # 1. Sesuaikan Base Model ke DeepHat
    BASE_MODEL = "DeepHat/DeepHat-V1-7B"
    # 2. Repo Adapter LoRA lo di Hugging Face
    ADAPTER_REPO = "IDINN/KiKai"
    SUBFOLDER = "adapter/deephat-offensive"

    # Load tokenizer dari Base Model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("⏳ Loading Base Model DeepHat 7B (4-bit mode)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    print(f"💉 Menyuntikkan jiwa KiKai (Adapter LoRA) dari {ADAPTER_REPO}/{SUBFOLDER}...")
    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_REPO,
        subfolder=SUBFOLDER
    )
    print("✅ KiKai Model & Tokenizer siap digunakan!")

# ============================================================
# KIKAI CHAT — STREAMING MODE
# ============================================================

GEN_CONFIG_STREAM = {
    "max_new_tokens": 512,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "pad_token_id": tokenizer.pad_token_id,
}

# System prompt disesuaikan dengan aura AI racikan lo
SYSTEM_PROMPT = "Gue KiKai, AI asisten cybersecurity buatan Idin Iskandar. Gaya ngomong gue santai, casual, pake gue-lo, dan selalu siap bantu bedah celah keamanan atau ngoding."

def chat_with_kikai_streaming():
    """Interactive chat dengan streaming response."""
    history = []

    print("=" * 60)
    print("💬 KIKAI CHAT — Streaming Mode 🌊")
    print("=" * 60)
    print("Commands: /clear, /history, /exit")
    print("=" * 60)
    print()

    model.eval()

    while True:
        try:
            user_input = input("🧑 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Bye bro!")
            break

        if not user_input:
            continue

        if user_input.lower() == "/exit":
            print("\n👋 Bye bro!")
            break

        if user_input.lower() == "/clear":
            history = []
            print("\n🧹 History cleared\n")
            continue

        if user_input.lower() == "/history":
            print(f"\n📜 {len(history)} messages in history\n")
            continue

        # Build messages
        messages = []
        if SYSTEM_PROMPT:
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
        messages.extend(history)
        messages.append({"role": "user", "content": user_input})

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        # Setup streamer
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=60.0,
        )

        # Generate di thread terpisah biar streaming jalan
        gen_kwargs = {**inputs, **GEN_CONFIG_STREAM, "streamer": streamer}
        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        # Stream output
        print("🎭 KiKai: ", end="", flush=True)
        response_parts = []
        for token in streamer:
            print(token, end="", flush=True)
            response_parts.append(token)
        print("\n")

        thread.join()

        response = "".join(response_parts).strip()
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})

        if len(history) > 20:
            history = history[-20:]

# START STREAMING CHAT
chat_with_kikai_streaming()
