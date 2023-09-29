import numpy as np
import torch
import torch.utils.data as Data

src_vocab = {'Empty':0, 'A': 1, 'C': 2,'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 
                'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 
                'T': 17, 'V': 18, 'W': 19, 'Y': 20}

def input_max_len(feature):
    src_len = 0 # enc_input max sequence length
    for i in range(len(feature)):
        if len(feature[i])>src_len:
            src_len = len(feature[i])
    return src_len

def make_seq_data(features,src_len):
    enc_inputs = []
    for i in range(len(features)):
        enc_input = [[src_vocab[n] for n in list(features[i])]]

        while len(enc_input[0])<src_len:
            enc_input[0].append(0)

        enc_inputs.extend(enc_input)

    return torch.LongTensor(enc_inputs)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    if labels.shape[-1] == 1:
        correct = preds.eq(labels.squeeze(1)).double()
    else:
        correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)