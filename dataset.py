import utils
import torch

from torch import nn
from torch.utils.data import DataLoader



class SentimentData(nn.Module):

    def __init__(self, df, tokenizer, max_len):
        super().__init__()

        self.text = df.iloc[:, 0].to_numpy()
        self.labels = df.iloc[:, 1].astype('int64').to_numpy()
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, item):
        text = str(self.text[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=self.max_len,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        truncation=True,
        return_tensors='pt',
        )

        return {
        'text': text,
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        'targets': torch.tensor(label, dtype=torch.long) 
        }

def create_data_loader(data_df, tokenizer, max_len):
  ds = SentimentData(
    df=data_df,
    tokenizer=tokenizer,
    max_len=max_len
  )

  return DataLoader(
    ds,
    batch_size=utils.BATCH_SIZE,
    num_workers=utils.NUM_WORKERS
  )


