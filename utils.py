import torch

# Daatset
TRAIN_PATH = './data/train.csv'
TEST_PATH = './data/test.csv'

BATCH_SIZE = 8
NUM_WORKERS = 1

#Model
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
NUM_CLS = 3
DROP = 0.3

#Train
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 10
LR = 2e-5

