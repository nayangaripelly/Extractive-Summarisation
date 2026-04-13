import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from src.data_preprocessing.loader import get_dataset, preprocess_data
from src.model.summarizer import BertSum
from src.utils.selection import greedy_selection_with_trigram_blocking
import torch.optim as optim
from src.training.evaluate import evaluate
import multiprocessing
from tqdm import tqdm

def train(model, dataloader, optimizer, loss_function, device):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        if batch is None:
            continue

        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        cls_positions = batch['cls_positions'].to(device)
        labels = batch['labels'].to(device)

        salience_scores = model(input_ids, attention_mask, cls_positions)

        min_len = min(salience_scores.size(1), labels.size(1))
        salience_scores = salience_scores[:, :min_len]
        labels = labels[:, :min_len]

        loss = loss_function(salience_scores, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    return total_loss / len(dataloader) if len(dataloader) > 0 else 0

if __name__ == '__main__':
    # Hyperparameters
    EPOCHS = 1 # For demonstration
    BATCH_SIZE = 1 # Reduced from 4 due to OOM (Out Of Memory, Exit Code 137)
    LEARNING_RATE = 2e-5

    # Load data
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Check if Mac GPU (MPS) is available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading and preprocessing data... (This will be fast if cached)")
    
    # Use all available cores for mapping. The results are automatically cached.
    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} cores for preprocessing.")

    train_dataset_raw = get_dataset('train')
    train_dataset = train_dataset_raw.map(
        lambda x: preprocess_data(x, tokenizer),
        num_proc=num_cores
    )
    
    # Filter out examples that couldn't be processed
    train_dataset = train_dataset.filter(lambda x: x is not None)

    def collate_fn(batch):
        batch = [item for item in batch if item is not None]
        if not batch:
            return None

        # Convert lists to tensors for each item
        for item in batch:
            item['input_ids'] = torch.tensor(item['input_ids'], dtype=torch.long)
            item['attention_mask'] = torch.tensor(item['attention_mask'], dtype=torch.long)
            item['cls_positions'] = torch.tensor(item['cls_positions'], dtype=torch.long)
            item['labels'] = torch.tensor(item['labels'], dtype=torch.float32)

        # Normalize sentence count per item (trim to the smallest available length)
        normalized = []
        for item in batch:
            num_sents = min(
                item['input_ids'].shape[0],
                item['attention_mask'].shape[0],
                item['cls_positions'].shape[0],
                item['labels'].shape[0],
            )
            normalized.append({
                'input_ids': item['input_ids'][:num_sents],
                'attention_mask': item['attention_mask'][:num_sents],
                'cls_positions': item['cls_positions'][:num_sents],
                'labels': item['labels'][:num_sents],
                'original_sentences': item['original_sentences'][:num_sents],
            })

        max_sents = max(item['input_ids'].shape[0] for item in normalized)

        input_ids_padded = []
        attention_mask_padded = []
        cls_positions_padded = []
        labels_padded = []
        original_sentences = []

        for item in normalized:
            pad_sents = max_sents - item['input_ids'].shape[0]

            input_ids_padded.append(
                torch.nn.functional.pad(item['input_ids'], (0, 0, 0, pad_sents), value=0)
            )
            attention_mask_padded.append(
                torch.nn.functional.pad(item['attention_mask'], (0, 0, 0, pad_sents), value=0)
            )
            cls_positions_padded.append(
                torch.nn.functional.pad(item['cls_positions'], (0, pad_sents), value=0)
            )
            labels_padded.append(
                torch.nn.functional.pad(item['labels'], (0, pad_sents), value=0.0)
            )
            original_sentences.append(item['original_sentences'])

        return {
            'input_ids': torch.stack(input_ids_padded),
            'attention_mask': torch.stack(attention_mask_padded),
            'cls_positions': torch.stack(cls_positions_padded),
            'labels': torch.stack(labels_padded),
            'original_sentences': original_sentences,
        }

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)

    # Initialize model, optimizer, and loss function
    model = BertSum().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = torch.nn.BCELoss()

    # Training loop
    for epoch in range(EPOCHS):
        train_loss = train(model, train_dataloader, optimizer, loss_function, device)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {train_loss:.4f}")

    # Example of how to use the trained model for prediction
    model.eval()
    
    # Get a single sample for prediction demonstration
    sample = train_dataset[0]

    with torch.no_grad():
        sample_input_ids = torch.tensor(sample['input_ids'], dtype=torch.long).unsqueeze(0).to(device)
        sample_attention_mask = torch.tensor(sample['attention_mask'], dtype=torch.long).unsqueeze(0).to(device)
        sample_cls_positions = torch.tensor(sample['cls_positions'], dtype=torch.long).unsqueeze(0).to(device)
        salience_scores = model(
            sample_input_ids,
            sample_attention_mask,
            sample_cls_positions
        ).cpu()
    
    original_sentences = sample['original_sentences']
    # Use the new selection logic
    summary = greedy_selection_with_trigram_blocking(original_sentences, salience_scores.squeeze(0).tolist())
    
    print("\n--- Example Summary ---")
    print("Original sentence count:", len(original_sentences))
    print("Selected sentence count:", len(summary.split('. '))) # Approximate
    print("\nSummary:\n", summary)

    # --- Evaluation Phase ---
    print("\nStarting evaluation on the test set...")
    avg_rouge_scores = evaluate(model, tokenizer, device)

    print("\n--- Evaluation Results ---")
    print(f"ROUGE-1: {avg_rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-2: {avg_rouge_scores['rouge2']:.4f}")
    print(f"ROUGE-L: {avg_rouge_scores['rougeL']:.4f}")
