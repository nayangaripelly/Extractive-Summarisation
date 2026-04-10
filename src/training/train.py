import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from src.data_preprocessing.loader import get_dataset, preprocess_data
from src.model.summarizer import BertSum
from src.utils.selection import greedy_selection_with_trigram_blocking
import torch.optim as optim
from src.training.evaluate import evaluate

def train(model, dataloader, optimizer, loss_function):
    model.train()
    total_loss = 0
    for batch in dataloader:
        if batch is None:
            continue
        optimizer.zero_grad()
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        cls_positions = batch['cls_positions']
        
        labels = batch['labels']

        salience_scores = model(input_ids, attention_mask, cls_positions)
        
        # Ensure scores and labels have the same length
        min_len = min(salience_scores.size(1), labels.size(1))
        salience_scores = salience_scores[:, :min_len]
        labels = labels[:, :min_len]

        loss = loss_function(salience_scores, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader) if len(dataloader) > 0 else 0

if __name__ == '__main__':
    # Hyperparameters
    EPOCHS = 1 # For demonstration
    BATCH_SIZE = 4
    LEARNING_RATE = 2e-5

    # Load data
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = get_dataset('train').map(lambda x: preprocess_data(x, tokenizer))
    
    # Filter out examples where cls_positions might be empty or problematic
    train_dataset = train_dataset.filter(lambda x: len(x['cls_positions']) > 0)

    def collate_fn(batch):
        # Filter out None values that may have been returned by preprocess_data
        batch = [item for item in batch if item is not None]
        if not batch:
            return None

        input_ids = torch.stack([item['input_ids'].squeeze(0) for item in batch])
        attention_mask = torch.stack([item['attention_mask'].squeeze(0) for item in batch])
        
        # Pad cls_positions and labels
        max_cls = max([len(item['cls_positions']) for item in batch])
        cls_positions = torch.stack([torch.nn.functional.pad(item['cls_positions'], (0, max_cls - len(item['cls_positions']))) for item in batch])
        labels = torch.stack([torch.nn.functional.pad(item['labels'], (0, max_cls - len(item['labels']))) for item in batch])
        
        # Keep original sentences and highlights for the last part of the script
        original_sentences = [item['original_sentences'] for item in batch]
        highlights = [item['highlights'] for item in batch] if 'highlights' in batch[0] else []


        return {
            'input_ids': input_ids, 
            'attention_mask': attention_mask, 
            'cls_positions': cls_positions, 
            'labels': labels,
            'original_sentences': original_sentences,
            'highlights': highlights
        }

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # Initialize model, optimizer, and loss function
    model = BertSum()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = torch.nn.BCELoss()

    # Training loop
    for epoch in range(EPOCHS):
        train_loss = train(model, train_dataloader, optimizer, loss_function)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {train_loss:.4f}")

    # --- Evaluation Phase ---
    print("\nStarting evaluation on the test set...")
    avg_rouge_scores = evaluate(model, tokenizer)

    print("\n--- Evaluation Results ---")
    print(f"ROUGE-1: {avg_rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-2: {avg_rouge_scores['rouge2']:.4f}")
    print(f"ROUGE-L: {avg_rouge_scores['rougeL']:.4f}")

    # Example of how to use the trained model for prediction
    model.eval()
    sample = train_dataset[0]
    with torch.no_grad():
        salience_scores = model(sample['input_ids'].unsqueeze(0), sample['attention_mask'].unsqueeze(0), sample['cls_positions'].unsqueeze(0))
    
    original_sentences = sample['original_sentences']
    # Use the new selection logic
    summary = greedy_selection_with_trigram_blocking(original_sentences, salience_scores.squeeze(0).tolist())
    
    print("\n--- Example Summary ---")
    print("Original sentence count:", len(original_sentences))
    print("Selected sentence count:", len(summary.split('. '))) # Approximate
    print("\nSummary:\n", summary)
