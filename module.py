from torch import nn
from transformers import BertModel
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.drop = nn.Dropout(p=0.3) ## For regularization with dropout probability 0.3.
        self.out = nn.Linear(self.bert.config.hidden_size,n_classes) ## append an Output fully connected layer representing the number of classes
    def forward(self, input_ids, attention_mask):
        returned = self.bert(input_ids=input_ids,attention_mask=attention_mask)
        pooled_output = returned["pooler_output"]
        output = self.drop(pooled_output)
        return self.out(output)