import torch
import torch.nn as nn
from transformers import BertModel

class BertSum(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super(BertSum, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.summarization_layer = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask, cls_positions):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Extract the CLS token embeddings
        cls_embeddings = hidden_states[torch.arange(hidden_states.size(0)).unsqueeze(1), cls_positions]
        
        # Get salience scores
        salience_scores = self.summarization_layer(cls_embeddings).squeeze(-1)
        return salience_scores
