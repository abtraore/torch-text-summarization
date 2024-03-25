import torch

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from data import load_dataset, make_maps, get_encoder_decoder, preproccess


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

        dialogue = self.dialogue[index]
        summary = self.summary[index]

        return dialogue, summary

    def trunc_pad(self):
        for i in range(self.__len__()):
            self.data[0][i] = torch.tensor(
                list(map(self.encoder, self.data[0][i].split()))[
                    : self.max_enc_seq_length
                ]
            )

        self.dialogue = self.data[0]

        self.dialogue = pad_sequence(self.dialogue)
        self.dialogue = torch.permute(self.dialogue, (1, 0))

        for i in range(self.__len__()):
            self.data[1][i] = torch.tensor(
                list(map(self.encoder, self.data[1][i].split()))[
                    : self.max_dec_seq_length
                ]
            )

        self.summary = self.data[1]

        self.summary = pad_sequence(self.summary)
        self.summary = torch.permute(self.summary, (1, 0))

        print(self.dialogue.shape)
        print(self.summary.shape)


# dt = SummarizationDataset("data/corpus")
# train_loader = DataLoader(dataset=dt, batch_size=8, shuffle=True)
