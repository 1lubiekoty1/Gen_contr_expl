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

