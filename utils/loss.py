import torch
import torch.nn.functional as F


def mask_loss(y_pred, y_true):

    y_pred = torch.flatten(y_pred, 0, 1)
    y_true = torch.flatten(y_true, 0, 1)

    mask_true = (y_true != 0).float()
    loss = F.cross_entropy(y_pred, y_true, reduction="none")
    loss *= mask_true
    loss = torch.sum(loss) / torch.sum(mask_true)

    return loss
