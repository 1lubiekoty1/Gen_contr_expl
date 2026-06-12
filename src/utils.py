import torch
from matplotlib import pyplot as plt
import config

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model( model : torch.nn.Module , filename: str = "model" ):
    config.MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.MODEL_SAVE_DIR / (filename + ".pt")
    torch.save(model.state_dict(), path )
    print(f"Model saved to {path}")

def load_model( model : torch.nn.Module , filename: str = "model" ) -> torch.nn.Module:
    path = config.MODEL_SAVE_DIR / (filename + ".pt")
    model.load_state_dict(torch.load(path, weights_only=True))          
    print(f"Loaded model from {path}")
    return model

def save_image( fig: plt.Figure , filename: str ):
    config.IMAGE_SAVE_DIR.mkdir( parents=True , exist_ok=True)
    path = config.IMAGE_SAVE_DIR / filename
    fig.savefig( path , dpi=150 , bbox_inches="tight")
    print(f"Image saved to {path}")

def merge_image_saliency( image: torch.Tensor , saliency: torch.Tensor ) -> torch.Tensor:
    saliency = saliency - saliency.mean()
    saliency = saliency/saliency.max()
    #squeezed = False
    #if( image.dim() == 4 ):
    #    image = image.squeeze()
    #    squeezed = True
    image = image.squeeze()
    image = image + saliency
    #if( squeezed ):
    #    image = image.unsqueeze(0)
    image = image.unsqueeze(0)
    return image
