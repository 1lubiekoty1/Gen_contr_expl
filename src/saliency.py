import torch
import matplotlib.pyplot as plt
from src.utils import save_image

def one_image_saliency(model: torch.nn.Module, image: torch.Tensor, target_class=None, device="cpu"):
    # COMPUTES SALIENCY MAP FOR A SINGLE IMAGE
    # ARGS:
    # model: - trained MnistCNN
    # image: - tensor (1,1,28,28)
    # target_class: - class to explain "what pixels would increase possibility of choosing this class"

    model.eval()
    image = image.clone().to(device).requires_grad_(True) # gradients is how saliency map is made so grad(True)
    print("before taking image")
    output = model(image)
    print("after taking image")

    if target_class is None: # if no target chosen - take the prediction for class
        target_class = output.argmax(dim=1).item()

    # backpropagate only the score for the target class
    score = output[0, target_class]
    print("before backwards")
    model.zero_grad()
    score.backward()
    print("after backwards")

    saliency = image.grad.data.squeeze()  # (28, 28)

    return saliency

# SALIENCY VISUALISATION
def save_saliency_result(image: torch.Tensor , saliency: torch.Tensor , filename: str):
    fig, axes = plt.subplots(1, 3, figsize=(6, 3))
    axes[0].imshow(image.squeeze().cpu(), cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    saliency = (saliency - saliency.mean()) * 1/((saliency - saliency.mean())).max()

    axes[1].imshow(saliency.cpu(), cmap="PiYG")
    axes[1].set_title("Saliency")
    axes[1].axis("off")

    axes[2].imshow((image.squeeze() + saliency ).cpu(), cmap="gray")
    axes[2].set_title("Merged")
    axes[2].axis("off")
    plt.tight_layout()
    #plt.show()
    save_image( plt , filename )