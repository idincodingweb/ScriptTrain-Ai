import torch, gc
torch.cuda.empty_cache()
gc.collect()
try:
    del model, trainer
except:
    pass
torch.cuda.empty_cache()
gc.collect()
print(f"VRAM free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated())/1024**3:.2f} GB")
