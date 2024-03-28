import torch
from torch.utils.data import Dataset

from .data import load_dataset, make_maps, get_encoder_decoder, preproccess

import numpy as np


class SummarizationDataset(Dataset):
    def __init__(
        self, dataset_path, train=True, max_enc_seq_length=150, max_dec_seq_length=50
    ):
        super().__init__()
        self.train_data, self.test_data = load_dataset(dataset_path)

        self.max_enc_seq_length = max_enc_seq_length
        self.max_dec_seq_length = max_dec_seq_length

        if train == True:
            self.data = preproccess(self.train_data)
        else:
            self.data = preproccess(self.test_data)

        self.stoi, self.itos = make_maps(self.data)

        self.vocab_size = len(self.stoi.keys())

        self.encoder, self.decoder = get_encoder_decoder(self.stoi, self.itos)

        self.trunc_pad()

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):

        dialogue = self.tensor_data_dialog[index]
        summary_in = self.tensor_data_summary_in[index]
        summary_out = self.tensor_data_summary_out[index]

        return dialogue, summary_in, summary_out

    def trunc_pad(self):
        # TODO: Find a better way to rewrite that.

        self.tensor_data_dialog = torch.zeros(
            (self.data[0].shape[0], self.max_enc_seq_length), dtype=torch.long
        )

        # Encode dialogue.
        for i in range(self.__len__()):

            encoded_data = torch.tensor(
                np.array(
                    list(map(self.encoder, self.data[0][i].split()))[
                        : self.max_enc_seq_length
                    ]
                )
            )

            padding_amount = self.max_enc_seq_length - encoded_data.shape[0]

            encoded_data = torch.constant_pad_nd(
                encoded_data,
                (0, padding_amount),
            )

            self.tensor_data_dialog[i] = encoded_data

        self.tensor_data_summary_in = torch.zeros(
            (self.data[0].shape[0], self.max_dec_seq_length), dtype=torch.long
        )

        self.tensor_data_summary_out = torch.zeros(
            (self.data[0].shape[0], self.max_dec_seq_length), dtype=torch.long
        )

        self.tensor_data_summary = torch.zeros(
            (self.data[0].shape[0], self.max_dec_seq_length), dtype=torch.long
        )

        # Encode summary.
        for i in range(self.__len__()):
            encoded_data = torch.tensor(
                np.array(
                    list(map(self.encoder, self.data[1][i].split()))[
                        : self.max_dec_seq_length
                    ]
                )
            )

            encoded_data_in = encoded_data[:-1]
            encoded_data_out = encoded_data[1:]

            padding_amount = self.max_dec_seq_length - encoded_data_in.shape[0]

            encoded_data_in = torch.constant_pad_nd(
                encoded_data_in,
                (0, padding_amount),
            )

            encoded_data_out = torch.constant_pad_nd(
                encoded_data_out,
                (0, padding_amount),
            )

            encoded_data = torch.constant_pad_nd(
                encoded_data,
                (0, padding_amount - 1),
            )

            self.tensor_data_summary_in[i] = encoded_data_in
            self.tensor_data_summary_out[i] = encoded_data_out
            self.tensor_data_summary[i] = encoded_data
