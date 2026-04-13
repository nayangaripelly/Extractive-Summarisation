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
        # input_ids shape: (batch_size, num_sentences, seq_length)
        batch_size, num_sentences, seq_length = input_ids.size()
        
        # Flatten the input for BERT: (batch_size * num_sentences, seq_length)
        input_ids_flat = input_ids.view(-1, seq_length)
        attention_mask_flat = attention_mask.view(-1, seq_length)

        outputs = self.bert(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
        hidden_states_flat = outputs.last_hidden_state
        
        # Reshape back to (batch_size, num_sentences, seq_length, hidden_size)
        hidden_states = hidden_states_flat.view(batch_size, num_sentences, seq_length, -1)
        
        # Extract the CLS token embeddings (which are at index 0 for each sentence in our setup)
        # Note: In BERTSum format, each sentence starts with a CLS token.
        # cls_positions here might be just a list of zeros if tokenized per sentence, 
        # or the actual indices if tokenized as a single document.
        # Given the collate_fn, we are likely treating sentences separately.
        # But we need to use the provided cls_positions appropriately.
        
        # cls_positions shape: (batch_size, num_sentences)
        # We need to extract the hidden state at these positions for each sentence.
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, num_sentences)
        sentence_indices = torch.arange(num_sentences).unsqueeze(0).expand(batch_size, -1)
        
        cls_embeddings = hidden_states[batch_indices, sentence_indices, cls_positions]
        
        # Get salience scores
        salience_scores = self.summarization_layer(cls_embeddings).squeeze(-1)
        return salience_scores
