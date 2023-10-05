import utils

from torch import nn
from transformers import BertModel

import utils
import torch.nn as nn

class SentimentClassifier(nn.Module):
    def __init__(self):
        super(SentimentClassifier, self).__init__()
        
        # Load the pre-trained BERT model
        self.bert = BertModel.from_pretrained(utils.PRE_TRAINED_MODEL_NAME)
        
        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Dropout layer after BERT
        self.dropout = nn.Dropout(p=utils.DROP)
        
        # Additional layers for classification
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 256)  # Add a hidden layer
        self.relu = nn.ReLU()  # Apply ReLU activation
        self.fc2 = nn.Linear(256, utils.NUM_CLS)  # Output layer for classification

    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get the pooled output
        pooled_output = bert_outputs.pooler_output
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Pass through additional layers
        x = self.fc1(pooled_output)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x