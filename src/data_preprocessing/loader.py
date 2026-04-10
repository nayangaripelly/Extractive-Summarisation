from datasets import load_dataset
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer
from src.utils.label_generator import create_labels
import nltk
import ssl
import torch

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # For Python versions that don't have _create_unverified_context
    pass
else:
    # For others, like macOS, where certificate verification fails
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('punkt_tab')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

def get_dataset(split='train'):
    """
    Loads and preprocesses the CNN/Daily Mail dataset.
    """
    dataset = load_dataset("cnn_dailymail", "3.0.0", split=split)
    return dataset

def preprocess_data(example, tokenizer):
    """
    Tokenizes article for BERTSUM and creates labels.
    """
    sentences = sent_tokenize(example['article'])
    
    # Create labels
    labels = create_labels(sentences, example['highlights'])
    
    inputs = tokenizer(sentences, max_length=512, truncation=True, padding='max_length', return_tensors="pt")
    
    cls_positions = (inputs['input_ids'][0] == tokenizer.cls_token_id).nonzero(as_tuple=True)[0]
    
    example['input_ids'] = inputs['input_ids']
    example['attention_mask'] = inputs['attention_mask']
    example['cls_positions'] = cls_positions
    example['original_sentences'] = sentences
    example['labels'] = torch.tensor(labels, dtype=torch.float32)
    
    return example

if __name__ == '__main__':
    # Example of loading and preprocessing
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = get_dataset('train')
    sample = train_dataset[0]
    processed_sample = preprocess_data(sample, tokenizer)
    
    print("Input IDs shape:", processed_sample['input_ids'].shape)
    print("Attention mask shape:", processed_sample['attention_mask'].shape)
    print("CLS positions:", processed_sample['cls_positions'])
    print("Original sentences:", processed_sample['original_sentences'])
    print("Labels shape:", processed_sample['labels'].shape)
    print("Labels:", processed_sample['labels'])
