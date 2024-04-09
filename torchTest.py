from torch import load, sigmoid, cat, rand, bmm, mean, matmul
import torch
config = {}

config['textfeat'] = '/Users/yusufsherif/Downloads/data/train/feat/textfeatures'
config['smallnwjc2vec'] = '/Users/yusufsherif/Downloads/data/train/feat/smallnwjc2vec'
config['visualfeat'] = '/Users/yusufsherif/Downloads/data/train/feat/visualfeatures'
textfeat = load(config['textfeat'],map_location=torch.device('cpu'))
smallnwjc2vec = load(config['smallnwjc2vec'],map_location=torch.device('cpu'))
print('Hello')