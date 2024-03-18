# %% [markdown]
# # User Latent Dirichlet Allocation

# %%
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pickle
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('.'))
torch.autograd.set_detect_anomaly(True)

# %%

from lib.models import InductiveLDA as InductiveLDA
from lib import utils as utils

# %% [markdown]
# Import Data

# %%
class Corpus:
    def __init__(self, datadir):
        filenames = ['train.txt.npy', 'test.txt.npy']
        self.datapaths = [os.path.join(datadir, x) for x in filenames]
        with open(os.path.join(datadir, 'vocab.pkl'), 'rb') as f:
            self.vocab = pickle.load(f)
        self.train, self.test = [
            Data(dp, len(self.vocab)) for dp in self.datapaths]


class Data:
    def __init__(self, datapath, vocab_size):
        data = np.load(datapath, allow_pickle=True, encoding='bytes')
        self.data = np.array([np.bincount(x.astype('int'), minlength=vocab_size) for x in data if np.sum(x)>0])
        self.documents = data
        
    @property
    def size(self):
        return len(self.data)
    
    def get_batch(self, batch_size, start_id=None):
        if start_id is None:
            batch_idx = np.random.choice(np.arange(self.size), batch_size)
        else:
            batch_idx = np.arange(start_id, start_id + batch_size)
        batch_data = self.data[batch_idx]
        data_tensor = torch.from_numpy(batch_data).float()
        return data_tensor

# %%
corpus = Corpus("data/20news")

# %%
RANDOM_SEED = 2112
np.random.seed(RANDOM_SEED)

# %%
X_document = corpus.train.data

# %%
num_docs, vocab_size = X_document.shape
print(f"Number of documents: {num_docs}, vocab size: {vocab_size}")

# %%
class Dataset(torch.utils.data.Dataset):
    def __init__(self, X): self.X = X
    def __len__(self): return len(self.X)
    def __getitem__(self, index): return self.X[index]

# %%
dataset = Dataset(torch.tensor(X_document).to(torch.float32))

# %%
num_topics = 50
prodlda = False
conv = False
num_layers = 0 # num hidden layers in the NN block. There are at least (1+num_heads) hidden layers in the encoder and decoder. num_layers comes on top of those. Total number of hidden layers is (1+num_layers+num_heads).
num_neurons = 100
dropout = True
dropout_rate = 0.5
batch_normalization = True
prior_params = {'alpha': 1.0}
decoder_temperature = 1.0
encoder_temperature = 1.0

# %%
model = InductiveLDA(input_dim=vocab_size,
                    num_topics=num_topics,
                    prior_params=prior_params,
                    conv=conv,
                    prodlda=prodlda,
                    decoder_temperature=decoder_temperature,
                    encoder_temperature=encoder_temperature,
                    num_hidden_layers=num_layers,
                    num_neurons=num_neurons,
                    dropout=dropout,
                    dropout_rate=dropout_rate,
                    batch_normalization=batch_normalization,
                    )

# %%
lr = 2e-3
batch_size = 200
num_epochs = 10
beta = 1.0
learn_prior = False

# %%
# Train the model using default partitioning choice 
model.fit(lr=lr,
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True),
            epochs = num_epochs,
            beta = beta,
            mc_samples = 1,
            learn_prior = learn_prior,
            tensorboard = False,
            )

# %%
model.decoder.get_beta().detach()

# %%
utils.to_alpha(model.prior_params["alpha"])

# %%
model.eval()

# %%
from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import gensim.corpora as corpora

# %%
## turn the encoding in corpus.train.documents into a list of list of words using the vocab
id2word = {v: k for k, v in corpus.vocab.items()}
texts = [[id2word[i] for i in doc] for doc in corpus.train.documents]

# %%
Beta = model.decoder.get_beta().detach().numpy()

# %%
result = {}
result["topic-word-matrix"] = Beta

top_k = 10

if top_k > 0:
    topics_output = []
    for topic in result["topic-word-matrix"]:
        top_k_words = list(reversed([id2word[i] for i in np.argsort(topic)[-top_k:]]))
        topics_output.append(top_k_words)
    result["topics"] = topics_output

# %%
print(result["topics"])

# %%
# Initialize metric
npmi = CoherenceModel(
    topics=result["topics"],
    texts=texts,
    corpus=corpus.train.data,
    dictionary=Dictionary(texts),
    coherence="c_npmi",
    topn=top_k)

# %%
npmi.get_coherence()
print(f"Coherence: {npmi.get_coherence()}")

# %%



