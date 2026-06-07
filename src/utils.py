import torch
import config

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model( model : torch.nn.Module ):
    config.MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH )
    print(f"Model saved to {config.MODEL_SAVE_PATH}")

def load_model( model : nn.Module ) -> nn.Module:
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, weights_only=True))
    print(f"Loaded model from {config.MODEL_SAVE_PATH}")
    return model