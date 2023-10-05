import utils
import dataset
import model

import numpy as np
import pandas as pd
import chardet
import torch
import transformers
import seaborn as sns
import matplotlib.pyplot as plt

from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from torch import nn, optim

import warnings
warnings.filterwarnings("ignore")

with open(utils.TRAIN_PATH, 'rb') as f:
    result = chardet.detect(f.read())

encoding = result['encoding']

sentiment_mapping = {'neutral': 1, 'negative': 0, 'positive': 2}

# Train Data
all_train = pd.read_csv(utils.TRAIN_PATH, encoding=encoding)
all_train['sentiment'] = all_train['sentiment'].map(sentiment_mapping)

train_data = all_train[['text', 'sentiment']].dropna()

#Teat Data
all_test = pd.read_csv(utils.TEST_PATH, encoding=encoding)
all_test['sentiment'] = all_test['sentiment'].map(sentiment_mapping)

test_data = all_test[['text', 'sentiment']].dropna()

#Tokenizer
tokenizer = BertTokenizer.from_pretrained(utils.PRE_TRAINED_MODEL_NAME)

## Run this part of code to find the max_len of texts (BERT works with fixed length)
# token_lens = []

# for txt in train_data[0]:
#   tokens = tokenizer.encode(txt, max_length=512)
#   token_lens.append(len(tokens))

# sns.distplot(token_lens)
# plt.xlim([0, 256])
# plt.xlabel('Token count')

# Set max_len of tokenized texts
MAX_LEN = 5

# Load Train Dataset
train_loader = dataset.create_data_loader(
    train_data, tokenizer, MAX_LEN
)

# Load Test Dataset
test_loader = dataset.create_data_loader(
    test_data, tokenizer, MAX_LEN
)

#Model
bert_model = model.SentimentClassifier().to(utils.DEVICE)

#Train
optimizer = optim.AdamW(
    bert_model.parameters(), lr=utils.LR
)

total_steps = len(train_loader) * utils.EPOCHS

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss().to(utils.DEVICE)


if __name__ == "__main__":
    for epoch in range(utils.EPOCHS):

        bert_model.train()

        losses = []
        correct_preds = 0
  
        for d in train_loader:
            input_ids = d["input_ids"].to(utils.DEVICE)
            attention_mask = d["attention_mask"].to(utils.DEVICE)
            targets = d["targets"].to(utils.DEVICE)

            outputs = bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask
            )

            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)

            correct_preds += torch.sum(preds == targets)
            losses.append(loss.item())

            loss.backward()
            nn.utils.clip_grad_norm_(bert_model.parameters(), max_norm=1.0)
            optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

        print(
            f'Epoch: {epoch}, TrainLoss={np.mean(losses)}, TrainACC={correct_preds.double() / len(train_data)}'
        )

        bert_model.eval()

        test_losses = []
        test_correct_preds = 0
  
        for d_test in test_loader:
            input_ids = d_test["input_ids"].to(utils.DEVICE)
            attention_mask = d_test["attention_mask"].to(utils.DEVICE)
            targets = d_test["targets"].to(utils.DEVICE)

            with torch.no_grad():
                outputs = bert_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)

            test_correct_preds += torch.sum(preds == targets)
            test_losses.append(loss.item())

        print(
            f'Epoch: {epoch}, TestLoss={np.mean(test_losses)}, TestACC={test_correct_preds.double() / len(test_data)}'
        )
