import csv
import torch
import numpy as np
import pickle as pkl
from os.path import join
from tqdm import tqdm
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM


data = list()
prop, nonprop = 0, 0
with open('data/data.tsv', 'r') as file:
    reader = csv.reader(file, delimiter='\t')
    next(reader)

    for row in reader:
        if row[1] is "1" or (len(row[0]) > 50 and len(row[0]) < 200 and nonprop < 5000):
            data.append(row)
            if row[1] is "1":
                prop = prop + 1
            elif row[1] is "0":
                nonprop = nonprop + 1
print("nonprop:" + str(nonprop))
print("prop:" + str(prop))


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

sentences_embeddings = dict()
for row in tqdm(data):
    marked_text = '[CLS] ' + row[0] + ' [SEP]'
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)

    token_embeddings = []
    for token_i in range(len(tokenized_text)):
        hidden_layers = []
        for layer_i in range(len(encoded_layers)):
            vec = encoded_layers[layer_i][0][token_i]
            hidden_layers.append(vec)
        token_embeddings.append(hidden_layers)

    summed_last_4_layers = [torch.sum(torch.stack(layer)[-4:], 0) for layer in token_embeddings]

    sentences_embeddings[(row[0], row[1])] = (summed_last_4_layers, row[-1])

with open('data-from-bert-1.pkl', 'wb') as file:
    pkl.dump(sentences_embeddings, file)




