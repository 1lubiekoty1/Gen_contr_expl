import torch
import torch.nn as nn
import config
import src.utils
import src.CNN

# here I make some metrics that will act as loss functions for counterfact generator

# PEANTLY FOR GENERAL CHANGES
def l1_loss( original , counterfactual ):
    return torch.abs(counterfactual - original).sum()

# PEANTLY FOR SATURATED CHANGES
def l2_loss( original , counterfactual ):
    return ((counterfactual-original)**2).sum()

# CONNECTS TWO ABOVE
def l1_l2_loss( original , counterfactual , alpha = 0.5):
    return alpha * l1_loss( original , counterfactual ) + (1-alpha) * l2_loss(original , counterfactual )

#CHECKS HOW MANY PIXELS CHANGED THEIR COLOR FORM BLACK TO WHITE
def perceptual_loss( original , counterfactual ):
    sign_change = ((counterfactual*original)<0).float()
    magnitude = torch.abs(counterfactual-original)
    return (sign_change*magnitude).sum()


def counterfactual_loss( original, counterfactual, model_output, target_class, plausibility_model , lambda_prox=1.0, lambda_plau=0.1, alpha=0.5 ):
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
    cls_loss = F.cross_entropy(model_output, target)

    # 2. Proximity: minimize how much the image changed
    prox_loss = l1_l2_loss(original, counterfactual, alpha)

    # 3. Plausibility: penaltise unmeaningful contrafacts
    plau_loss  = plausibility_model.plausibility_loss( counterfactual , target_class )



    total = cls_loss + lambda_prox * prox_loss + lambda_plau * palu_loss;

    return total, {
        "classification": cls_loss.item(),
        "proximity":      prox_loss.item(),
        "plausibility":     plau_loss.item(),
        "total":          total.item()
    }

