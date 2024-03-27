import torch
from .models import create_padding_mask, create_look_ahead_mask
from .data import preproccess_util


def proccess_inputs(inputs, encoder):
    proccessed_input = preproccess_util(input_data=inputs)
    encoded_input = list(map(encoder, proccessed_input.split()))
    return torch.tensor(encoded_input).unsqueeze(0)


def next_word(model, enc_input, output, encoder, device="cpu"):
    with torch.no_grad():
        model = model.to(device)

        enc_input = proccess_inputs(enc_input, encoder)

        enc_input = enc_input.to(device)
        output = output.to(device)

        enc_padding_mask = create_padding_mask(enc_input)
        dec_padding_mask = create_padding_mask(output)
        dec_look_ahead_mask = create_look_ahead_mask(output.shape[1]).to(device)

        predictions = model(
            enc_input,
            output,
            enc_padding_mask,
            dec_padding_mask,
            dec_look_ahead_mask,
        )

        predictions = predictions[0, -1:, :]
        # print(predictions.sum())
        predicted_id = torch.argmax(predictions, dim=-1)

    return predicted_id


def summarize(model, enc_input, output, max_dec_len, encoder, decoder, device="cpu"):

    for _ in range(max_dec_len):

        predicted_id = next_word(model, enc_input, output, encoder, device=device)
        output = torch.concat([output[0], predicted_id]).unsqueeze(0)

        if predicted_id[0] == 2:
            break

    return " ".join(list(map(decoder, output[0].cpu().numpy())))
