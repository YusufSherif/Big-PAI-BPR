from copy import deepcopy

from torch.nn import *
from torch.utils.data import TensorDataset,DataLoader
from torch.nn.functional import logsigmoid
from torch import load, sigmoid, cat, rand, bmm, mean, matmul
from torch.optim import Adam
import numpy as np
import torch

import gensim

class word2vec():
    def __init__(self):
        self.vectors = gensim.models.KeyedVectors.load(
            "/Users/yusufsherif/Desktop/PMDL/chive-1.2-mc5_gensim/chive-1.2-mc5.kv")

    def getEmbedding(self, word):
        return self.vectors.get_vector(word)

    def get_vectors(self):
        return self.vectors.vectors

def load_embedding_weight():
    jap2vec = word2vec()
    embeding_weight = []
    embeding_weight = np.vstack((jap2vec.get_vectors(), np.zeros(300)))
    print("appended")
    embedding_weight = torch.tensor(embeding_weight)
    print("converted")
    return embedding_weight

jap2vec = word2vec()

jap2vec.getEmbedding('')

print('hello')