import torch
import config
import src.CNN
import src.utils
import src.autoencoder
from src.visualization import plot_counterfactual_grid
from torchvision import datasets, transforms


def main():
    device = src.utils.get_device()
    print(f"Using device: {device}")

    # --- Load the trained black-box classifier ---
    model = src.CNN.MnistCNN().to(device)
    src.utils.load_model(model)
    model.eval()

    # --- Load the trained class-conditional plausibility autoencoders ---
    plausibility_model = src.autoencoder.ClassConditionalPlausibility()
    plausibility_model.load_saved_autoencoders()

    # --- Load test dataset ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((config.MNIST_MEAN,), (config.MNIST_STD,))
    ])
    dataset = datasets.MNIST(config.DATA_DIR, train=False, download=True, transform=transform)

    # --- Pick an example and a target class ---
    number = 3000
    image, label = dataset[number]
    image = image.unsqueeze(0).to(device)  # (1,1,28,28)
    target_class = 8

    print(f"Original label: {label}, target class: {target_class}")

    # --- Generate and visualize the counterfactual ---
    examples = [(image, target_class)]
    all_metrics, summary = plot_counterfactual_grid(
        model, plausibility_model, examples, device=device,
        generate_kwargs={"num_steps": 300, "lr": 0.01},
        save_path=str(config.IMAGE_SAVE_DIR / f"counterfactual_{number}.png")
    )

    print(summary)


if __name__ == "__main__":
    main()