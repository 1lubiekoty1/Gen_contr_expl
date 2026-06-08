import torch
from matplotlib import pyplot as plt
import config

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model( model : torch.nn.Module ):
    config.MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH )
    print(f"Model saved to {config.MODEL_SAVE_PATH}")

def load_model( model : torch.nn.Module ) -> torch.nn.Module:
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, weights_only=True))          
    print(f"Loaded model from {config.MODEL_SAVE_PATH}")
    return model

def save_image( fig: plt.Figure , filename: str ):
    config.IMAGE_SAVE_DIR.mkdir( parents=True , exist_ok=True)
    path = config.IMAGE_SAVE_DIR / filename
    fig.savefig( path , dpi=150 , bbox_inches="tight")
    print(f"Image saved to {path}")