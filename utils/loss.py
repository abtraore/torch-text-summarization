import torch
import torch.nn.functional as F


def mask_loss(y_pred, y_true):

    # Vectorize predictions and ground truth.
    y_pred = torch.flatten(y_pred, 0, 1)  # (B, S, vocab_size) -> (B*S, vocab_size)
    y_true = torch.flatten(y_true, 0, 1)  # (B, S) -> (B*S, )

    # Create a mask to avoid adding 0 (padding to the loss computation).
    mask_true = (y_true != 0).float()

    loss = F.cross_entropy(y_pred, y_true, reduction="none")  # Compute the loss
    loss *= mask_true  # Turn padding values to 0.
    loss = torch.sum(loss) / torch.sum(
        mask_true
    )  # Compute the loss on values that are padding.

    return loss
