import utils

from torch import nn
from transformers import BertModel

# Model
class SentimentClassifier(nn.Module):

  def __init__(self,):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(utils.PRE_TRAINED_MODEL_NAME)
    self.drop = nn.Dropout(p=utils.DROP)
    self.out = nn.Linear(self.bert.config.hidden_size, utils.NUM_CLS)
  
  def forward(self, input_ids, attention_mask):
    bert_ouputs = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )

    pooled_output = bert_ouputs.pooler_output
    output = self.drop(pooled_output)

    return self.out(output)