import torch
import matplotlib.pyplot as plt
import config
import src.CNN
import src.utils
import src.autoencoder
from src.counterfact import generate_counterfactual_latent
from src.utils import normalize, denormalize
from torchvision import datasets, transforms

device = src.utils.get_device()

# --- load black box + plausibility autoencoders ---
model = src.CNN.MnistCNN().to(device)
src.utils.load_model(model)
model.eval()

plausibility_model = src.autoencoder.ClassConditionalPlausibility()
plausibility_model.load_saved_autoencoders()

# --- load same example as before ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((config.MNIST_MEAN,), (config.MNIST_STD,))
])
dataset = datasets.MNIST(config.DATA_DIR, train=False, download=True, transform=transform)

number = 2137
target_class = 8
image, label = dataset[number]
image = image.unsqueeze(0).to(device)
print(f"Original label: {label}, target class: {target_class}")

# --- sanity check: what does z0 decode to, BEFORE optimization? ---
ae = plausibility_model.autoencoders[target_class].to(device)
ae.eval()
with torch.no_grad():
    z0 = ae.encoder(image)
    initial_decode_raw = ae.decoder(z0)

fig, axes = plt.subplots(1, 2, figsize=(4, 2))
axes[0].imshow(denormalize(image.squeeze().cpu()), cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Original")
axes[0].axis("off")
axes[1].imshow(initial_decode_raw.squeeze().cpu(), cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"z0 decoded\n(class-{target_class} AE)")
axes[1].axis("off")
plt.tight_layout()
plt.savefig("images/latent_sanity_check.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved images/latent_sanity_check.png")

# --- now run the actual optimization ---
decoded_norm, history = generate_counterfactual_latent(
    model, plausibility_model, image, target_class,
    num_steps=300, lr=0.05, device=device, verbose=True
)

fig, axes = plt.subplots(1, 2, figsize=(4, 2))
axes[0].imshow(denormalize(image.squeeze().cpu()), cmap="gray", vmin=0, vmax=1)
axes[0].set_title(f"Original (pred {label})")
axes[0].axis("off")
axes[1].imshow(denormalize(decoded_norm.squeeze().cpu()), cmap="gray", vmin=0, vmax=1)
with torch.no_grad():
    final_pred = model(decoded_norm).argmax(dim=1).item()
axes[1].set_title(f"Latent CF (pred {final_pred})")
axes[1].axis("off")
plt.tight_layout()
plt.savefig(f"images/latent_counterfactual_{number}.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved images/latent_counterfactual_{number}.png")