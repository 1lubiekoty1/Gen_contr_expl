import torch
import torch.nn.functional as F


# here I make some metrics that will act as loss functions for counterfact generator

# PEANTLY FOR GENERAL CHANGES
def l1_loss(original, counterfactual):
    return torch.abs(counterfactual - original).sum()


# PEANTLY FOR SATURATED CHANGES
def l2_loss(original, counterfactual):
    return ((counterfactual - original) ** 2).sum()


# CONNECTS TWO ABOVE
def l1_l2_loss(original, counterfactual, alpha=0.5):
    return alpha * l1_loss(original, counterfactual) + (1 - alpha) * l2_loss(original, counterfactual)


# CHECKS HOW MANY PIXELS CHANGED THEIR COLOR FORM BLACK TO WHITE
def perceptual_loss(original, counterfactual):
    sign_change = ((counterfactual * original) < 0).float()
    magnitude = torch.abs(counterfactual - original)
    return (sign_change * magnitude).sum()


def counterfactual_loss(original, counterfactual, model_output, target_class, plausibility_model, lambda_prox=1.0,
                        lambda_plau=0.1, alpha=0.5):
    # LOSS FUNCTION THAT COMBINES ALL METHODS - 
    # 
    # original: - original image
    # counterfactual: - image of potential counterfact
    # model_output: - how does model classify the counterfact
    # target_class: - class that counterfact is supposed to be
    # plausibility_model: - the ClassConditionalPlausibility that runs plausibility loss
    # lambda_prox: - importance of proximity
    # lambda_plau: - importance of plausibility
    # alpha: l1/L2 balance

    # 1. Classification: push model toward target class
    target = torch.tensor([target_class], device=model_output.device)
    # model_output is already a probability distribution (Softmax in MnistCNN),
    # so we use nll_loss on its log instead of cross_entropy (which would
    # apply softmax again).
    cls_loss = F.nll_loss(torch.log(model_output + 1e-12), target)

    # 2. Proximity: minimize how much the image changed
    prox_loss = l1_l2_loss(original, counterfactual, alpha)

    # 3. Plausibility: penaltise unmeaningful contrafacts
    plau_loss = plausibility_model.plausibility_loss(counterfactual, target_class)

    total = cls_loss + lambda_prox * prox_loss + lambda_plau * plau_loss

    return total, {
        "classification": cls_loss.item(),
        "proximity": prox_loss.item(),
        "plausibility": plau_loss.item(),
        "total": total.item()
    }

# Valid pixel range after MNIST normalization (mean=0.1307, std=0.3081),
# corresponding to raw pixel values in [0, 1].
PIXEL_MIN = (0.0 - config.MNIST_MEAN) / config.MNIST_STD
PIXEL_MAX = (1.0 - config.MNIST_MEAN) / config.MNIST_STD


def generate_counterfactual(model, plausibility_model, original, target_class,
                             num_steps=300, lr=0.01,
                             lambda_prox=1.0, lambda_plau=0.1, alpha=0.5,
                             clamp_to_valid_range=True, verbose=True, device="cpu"):
    """
    Runs gradient descent directly on the image pixels to find a counterfactual.

    Args:
        model: trained MnistCNN (frozen, eval mode)
        plausibility_model: trained ClassConditionalPlausibility
        original: tensor (1,1,28,28), the starting image
        target_class: int, desired output class
        num_steps: number of optimization steps
        lr: Adam learning rate for the pixel updates
        lambda_prox, lambda_plau, alpha: see counterfactual_loss
        clamp_to_valid_range: if True, clamp pixels back to the valid
            normalized range after each step (keeps the image "realistic"
            in terms of intensity, doesn't blow up to extreme values)
        device: "cpu" or "cuda"

    Returns:
        best_counterfactual: tensor (1,1,28,28), detached
        history: list of per-step loss component dicts
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    original = original.to(device).detach()
    counterfactual = original.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([counterfactual], lr=lr)

    history = []
    best_cf = counterfactual.detach().clone()
    best_prox = None

    for step in range(num_steps):
        optimizer.zero_grad()

        output = model(counterfactual)
        loss, components = counterfactual_loss(
            original, counterfactual, output, target_class,
            plausibility_model, lambda_prox, lambda_plau, alpha
        )

        loss.backward()
        optimizer.step()

        if clamp_to_valid_range:
            with torch.no_grad():
                counterfactual.clamp_(PIXEL_MIN, PIXEL_MAX)

        pred = output.argmax(dim=1).item()
        components["pred"] = pred
        history.append(components)

        # keep the cheapest (smallest proximity) counterfactual that
        # already fools the model into the target class
        if pred == target_class:
            if best_prox is None or components["proximity"] < best_prox:
                best_prox = components["proximity"]
                best_cf = counterfactual.detach().clone()

        if verbose and (step % 20 == 0 or step == num_steps - 1):
            print(f"Step {step:03d} | total={components['total']:.4f} "
                  f"| cls={components['classification']:.4f} "
                  f"| prox={components['proximity']:.4f} "
                  f"| plau={components['plausibility']:.4f} "
                  f"| pred={pred}")

    if best_prox is None:
        # never reached target class; return the final iterate anyway
        best_cf = counterfactual.detach().clone()

    return best_cf, history