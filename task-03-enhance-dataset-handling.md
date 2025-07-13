# Task 3: Enhance Dataset Handling

## Status: In Progress

## Description
Replace the basic in-memory list with a robust dataset loading system using Hugging Face datasets, implementing proper batching, tokenization, and distributed data loading.

## Sub-tasks

### Sub-task 1: Hugging Face Datasets Integration
- Replace basic list with `datasets` library loader
- Support multiple formats: JSONL, Parquet, CSV, JSON
- Add dataset validation and preprocessing

### Sub-task 2: Data Collation and Batching
- Implement custom data collator for instruction-following format
- Add tokenization with proper padding and masking
- Support variable-length sequences

### Sub-task 3: Distributed Data Loading
- Integrate PyTorch DataLoader for efficient batching
- Add support for distributed sampling
- Implement shuffling and sampling strategies
- Handle large datasets without OOM

## Implementation Details

### Dataset Format
The dataset should contain dictionaries with:
- `prompt`: The input prompt/instruction
- `completion`: The expected completion/response
- Optional: `system_prompt`, `metadata`

### Features to Implement
1. **Flexible Loading**: Support local files and Hugging Face Hub datasets
2. **Preprocessing Pipeline**: Tokenization, padding, truncation
3. **Batch Sampling**: Efficient batching with DataLoader
4. **Memory Efficiency**: Streaming for large datasets
5. **Distributed Support**: Proper data sharding across ranks

## Code Structure
```python
class MeZODataset:
    def __init__(self, dataset_path, tokenizer, max_length=512):
        # Load dataset from various sources
        # Preprocess and tokenize
        
    def __getitem__(self, idx):
        # Return tokenized item
        
    def __len__(self):
        # Return dataset length

def create_dataloader(dataset, batch_size, distributed=False):
    # Create DataLoader with appropriate sampler
```

## Progress
- [ ] Create MeZODataset class
- [ ] Implement dataset loading from multiple sources
- [ ] Add tokenization and preprocessing
- [ ] Create data collator
- [ ] Implement distributed sampling
- [ ] Add unit tests

## Testing Plan
- Test with small synthetic datasets
- Validate tokenization and padding
- Test distributed loading with multiple GPUs
- Benchmark loading performance