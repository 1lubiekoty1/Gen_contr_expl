import torch
import numpy as np
import matplotlib.pyplot as plt
import config
from src.utils import denormalize
from src.counterfact import l1_loss, l2_loss, generate_counterfactual


def compute_metrics(model, plausibility_model, original, counterfactual, target_class, device="cpu"):
    """Computes the evaluation metrics for a single (original, counterfactual) pair."""
    model.eval()
    with torch.no_grad():
        orig_pred = model(original.to(device)).argmax(dim=1).item()
        cf_output = model(counterfactual.to(device))
        cf_pred = cf_output.argmax(dim=1).item()

        plau = plausibility_model.plausibility_loss(counterfactual.to(device), target_class).item()

    return {
        "original_class": orig_pred,
        "counterfactual_class": cf_pred,
        "target_class": target_class,
        "success": cf_pred == target_class,
        "l1": l1_loss(original, counterfactual).item(),
        "l2": l2_loss(original, counterfactual).item(),
        "plausibility": plau,
    }


def plot_counterfactual_grid(model, plausibility_model, examples, device="cpu",
                              generate_kwargs=None, save_path=None):
    """
    Runs the counterfactual generator on a list of examples and plots
    Original -> Change Map -> Counterfactual for each, with per-row metrics.

    Args:
        model: trained MnistCNN
        plausibility_model: trained ClassConditionalPlausibility
        examples: list of (image_tensor, target_class) pairs.
                  image_tensor should be (1,1,28,28), normalized.
        device: "cpu" or "cuda"
        generate_kwargs: extra kwargs forwarded to generate_counterfactual
                         (e.g. num_steps, lr, lambda_prox, lambda_plau)
        save_path: if given, saves the figure here (via plt.savefig)

    Returns:
        all_metrics: list of per-example metric dicts
        summary: dict with aggregated stats (success_rate, mean_l1, mean_l2, mean_plausibility)
    """
    generate_kwargs = generate_kwargs or {}
    n = len(examples)

    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes.reshape(1, 3)  # keep indexing consistent for a single row

    all_metrics = []

    for row, (image, target_class) in enumerate(examples):
        image = image.to(device)

        counterfactual, history = generate_counterfactual(
            model, plausibility_model, image, target_class,
            device=device, verbose=False, **generate_kwargs
        )

        metrics = compute_metrics(model, plausibility_model, image, counterfactual, target_class, device)
        all_metrics.append(metrics)

        # --- denormalize for display ---
        orig_disp = denormalize(image.squeeze().cpu()).numpy()
        cf_disp = denormalize(counterfactual.squeeze().cpu()).numpy()
        diff = cf_disp - orig_disp  # change map, range roughly [-1, 1]

        # --- Original ---
        axes[row, 0].imshow(orig_disp, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"Original (pred: {metrics['original_class']})")
        axes[row, 0].axis("off")

        # --- Change map ---
        axes[row, 1].imshow(diff, cmap="bwr", vmin=-1, vmax=1)
        axes[row, 1].set_title("Change Map")
        axes[row, 1].axis("off")

        # --- Counterfactual ---
        success_str = "✓" if metrics["success"] else "✗"
        axes[row, 2].imshow(cf_disp, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_title(f"Counterfactual (pred: {metrics['counterfactual_class']}) {success_str}")
        axes[row, 2].axis("off")

        # --- per-row metrics as a subtitle under the whole row ---
        subtitle = (f"target={target_class} | L1={metrics['l1']:.2f} | "
                    f"L2={metrics['l2']:.2f} | plausibility={metrics['plausibility']:.4f}")
        axes[row, 1].text(0.5, -0.18, subtitle, transform=axes[row, 1].transAxes,
                           ha="center", va="top", fontsize=9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    # --- aggregate stats across all examples ---
    summary = {
        "success_rate": float(np.mean([m["success"] for m in all_metrics])),
        "mean_l1": float(np.mean([m["l1"] for m in all_metrics])),
        "mean_l2": float(np.mean([m["l2"] for m in all_metrics])),
        "mean_plausibility": float(np.mean([m["plausibility"] for m in all_metrics])),
    }

    return all_metrics, summary