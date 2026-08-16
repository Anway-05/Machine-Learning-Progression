import torch
from pathlib import Path

def save_model(model, target_dir, model_name):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), target_dir / model_name)

    print(f"Model saved to {target_dir / model_name}")