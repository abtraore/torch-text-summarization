import torch

from tqdm import tqdm

from .loss import mask_loss
from .models import create_look_ahead_mask, create_padding_mask


def loops(model, epochs, train_loader, val_loader, optimizer, scheduler, device):

    model = model.to(device)

    for epoch in tqdm(range(epochs), desc="Epochs", leave=False):

        train_total_loss = 0.0
        model.train(True)

        for step, data in enumerate(tqdm(train_loader, desc="Training", leave=False)):

            lr = scheduler(step * epoch)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr.item()  # Update the learning rate

            context, target = data

            context = context.to(device)
            target = target.to(device)

            target_in = target[:, :-1]
            target_out = target[:, 1:]

            enc_padding_mask = create_padding_mask(context)
            dec_padding_mask = create_padding_mask(target_in).to(device)
            dec_look_ahead_mask = create_look_ahead_mask(target_in.shape[1]).to(device)

            optimizer.zero_grad()

            out = model(
                context,
                target_in,
                enc_padding_mask,
                dec_padding_mask,
                dec_look_ahead_mask,
            )

            loss = mask_loss(out, target_out)

            loss.backward()
            optimizer.step()

            train_total_loss += loss.item()

        train_total_loss = train_total_loss / len(train_loader)

        with torch.no_grad():
            model.train(False)
            val_total_loss = 0.0
            for _, data in enumerate(tqdm(val_loader, desc="Validating", leave=False)):

                context, target = data

                context = context.to(device)
                target = target.to(device)

                target_in = target[:, :-1]
                target_out = target[:, 1:]

                enc_padding_mask = create_padding_mask(context)
                dec_padding_mask = create_padding_mask(target_in).to(device)
                dec_look_ahead_mask = create_look_ahead_mask(target_in.shape[1]).to(
                    device
                )

                optimizer.zero_grad()

                out = model(
                    context,
                    target_in,
                    enc_padding_mask,
                    dec_padding_mask,
                    dec_look_ahead_mask,
                )

                loss = mask_loss(out, target_out)

                val_total_loss += loss.item()

        val_total_loss = val_total_loss / len(val_loader)

        print(f" Epoch: {epoch+1} | Train Loss: {train_total_loss:.4}")
        print(f" Epoch: {epoch+1} | Val Loss: {val_total_loss:.4}\n")
