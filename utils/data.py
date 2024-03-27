import pandas as pd
import re
import numpy as np


def load_dataset(dataset_dir: str):
    # Get Training
    train_data = pd.read_json(f"{dataset_dir}/train.json")
    train_data.drop(["id"], axis=1, inplace=True)

    # Get test
    test_data = pd.read_json(f"{dataset_dir}/test.json")
    test_data.drop(["id"], axis=1, inplace=True)

    return train_data, test_data


def preproccess_util(input_data: str):
    lowercase = input_data.lower()
    removed_newlines = re.sub("\n|\r|\t", " ", lowercase)
    removed_double_spaces = " ".join(removed_newlines.split(" "))
    clean = "[SOS] " + removed_double_spaces + " [EOS]"
    return clean


def preproccess(input_data: pd.DataFrame):

    input_data["summary"] = input_data.apply(
        lambda row: preproccess_util(row["summary"]), axis=1
    )

    input_data["dialogue"] = input_data.apply(
        lambda row: preproccess_util(row["dialogue"]), axis=1
    )

    document = input_data["dialogue"].to_numpy()
    summary = input_data["summary"].to_numpy()

    return [document, summary]


def make_map(data):

    filters = '!"#$%&()*+,-./:;<=>?@\\^_`{|}~\t\n'
    pattern = "[" + filters + "]"

    stoi = {}
    stoi[""] = 0
    stoi["[SOS]"] = 1
    stoi["[EOS]"] = 2
    stoi["[UNK]"] = 3
    stoi[" "] = 4

    idx = 0
    for _, line in enumerate(data):
        line = re.sub(pattern, "", line)
        clean_words = [words.strip() for words in line.split()]

        for word in clean_words:

            if word not in stoi:
                stoi[word] = idx + 5
                idx += 1

    itos = {v: k for k, v in stoi.items()}

    return stoi, itos


def make_maps(data):

    stoi_d, itos_d = make_map(data[0])
    stoi_s, itos_s = make_map(data[1])

    stoi_s.update(stoi_d)
    itos_s.update(itos_d)

    return stoi_s, itos_s


def try_dict(dic, x, is_stoi=True):
    try:
        return dic[x]
    except:
        if is_stoi:
            return 3
        else:
            return "[UNK]"


def get_encoder_decoder(stoi, itos):

    encoder = lambda x: try_dict(stoi, x)
    decoder = lambda x: try_dict(itos, x, False)

    return encoder, decoder
