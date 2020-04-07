import torch
import torch.nn as nn
import torchtext
from torchtext.data import Field, TabularDataset, Iterator, batch
from torchtext.data import Iterator, BucketIterator
from torchtext.vocab import Vectors
from src.models import get_model
from torch.optim import Adam
from tqdm import tqdm
from torch.autograd import Variable
import spacy
import re
import numpy as np
import pandas as pd
import os

EPOCH = 2000
K = 128
H = 4
N = 6
B = 32
dropout = 0.1
MIN_FREQ = 2
MAX_LEN = 100
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Tokenizer(object):

    def __init__(self, lang):
        self.nlp = spacy.load(lang)

    def tokenize(self, sentence):
        sentence = re.sub(r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
        sentence = re.sub(r"[ ]+", " ", sentence)
        sentence = re.sub(r"\!+", "!", sentence)
        sentence = re.sub(r"\,+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        sentence = sentence.lower()
        return [tok.text for tok in self.nlp.tokenizer(sentence) if tok.text != " "]

BOS_WORD = '<sos>'
EOS_WORD = '<eos>'
TRG = Field(
    lower=True,
    tokenize=Tokenizer('fr_core_news_sm').tokenize,
    init_token=BOS_WORD,
    eos_token=EOS_WORD,
    batch_first=True
)
SRC = Field(
    lower=True,
    tokenize=Tokenizer('en_core_web_sm').tokenize,
    batch_first=True
)

path = 'C:/Users/bill/Documents/projects/data/tutorial/transformer/'
if not os.path.exists(os.path.join(path, 'temp.csv')):
    source = open(os.path.join(path, 'english.txt'), encoding='utf8').read().strip().split('\n')
    target = open(os.path.join(path, 'french.txt'), encoding='utf8').read().strip().split('\n')
    df = pd.DataFrame({
        'src' : source,
        'trg': target
    }, columns=["src", "trg"])
    df = df[(df['src'].str.count(' ') <= MAX_LEN) & (df['trg'].str.count(' ') <= MAX_LEN)]
    df.to_csv(os.path.join(path, 'temp.csv'), index=False)
data = TabularDataset(
    os.path.join(path, 'temp.csv'),
    format='csv',
    fields=[
        ('src', SRC),
        ('trg', TRG)
    ])
train, valid, test = data.split(split_ratio=[0.8, 0.1, 0.1])

SRC.build_vocab(train.src, min_freq=MIN_FREQ)
TRG.build_vocab(train.trg, min_freq=MIN_FREQ)
src_pad = SRC.vocab.stoi['<pad>']
trg_pad = TRG.vocab.stoi['<pad>']

itrain, ivalid = BucketIterator.splits(
    (train, valid),
    batch_sizes=(B, B),
    device=device,
    sort_key=lambda x: len(x.text),
    sort_within_batch=False,
    repeat=False
)
itest = Iterator(
    test,
    batch_size=B,
    device=device,
    sort=False,
    sort_within_batch=False,
    repeat=False
)

model = get_model(
    src_vocab=len(SRC.vocab),
    trg_vocab=len(TRG.vocab),
    K=K,
    H=H,
    N=N,
    dropout=dropout
)

def nopeak_mask(size):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask) == 0)
    np_mask = np_mask.cuda()
    return np_mask

def create_masks(src, trg):
    src_mask = (src != src_pad).unsqueeze(-2)
    if trg is not None:
        trg_mask = (trg != trg_pad).unsqueeze(-2)
        size = trg.size(1)
        np_mask = nopeak_mask(size)
        trg_mask = trg_mask & np_mask
    else:
        trg_mask = None
    return src_mask, trg_mask

def scoring(device, model, criterion, iterator):
    with torch.no_grad():
        total_loss = []
        for batch in iterator:
            src = batch.src.to(device)
            trg = batch.trg.to(device)
            src_mask, trg_mask = create_masks(src, trg[:, :-1])
            preds = model(src, trg_input, src_mask, trg_mask)
            loss = criterion(
                preds.view(-1, preds.size(-1)),
                trg[:, 1:].contiguous().view(-1))
            total_loss.append(loss.item())
    return np.mean(total_loss)


print("training model...")
model.to(device)
# model.train()

optimizer = Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
criterion = nn.CrossEntropyLoss(ignore_index=trg_pad)

for epoch in range(EPOCH):

    total_loss = []
    for batch in tqdm(itrain):
        # index of the tokens B x T
        src = batch.src.to(device)
        trg = batch.trg.to(device)

        optimizer.zero_grad()

        trg_input = trg[:, :-1]
        src_mask, trg_mask = create_masks(src, trg_input)
        preds = model(src, trg_input, src_mask, trg_mask)

        loss = criterion(
            preds.view(-1, preds.size(-1)),
            trg[:, 1:].contiguous().view(-1))

        total_loss.append(loss.item())

        loss.backward()
        optimizer.step()

    train_loss = np.mean(total_loss)
    valid_loss = scoring(device, model, criterion, ivalid)

    print("epoch %d, train_loss = %.3f, valid_loss = %.03f" % (epoch, train_loss, valid_loss))
    # print("epoch %d, train_loss = %.3f" % (epoch, train_loss))