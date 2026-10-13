from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH = DATA_DIR / "products.csv"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.pt"

df = pd.read_csv(CSV_PATH)
df['combined'] = df['combined'].fillna('').astype(str)

model_ckpt = "sentence-transformers/all-MiniLM-L6-v2"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt).to(device)

texts = df['combined'].to_list()

BATCH_SIZE = 256
all_embeddings = []

total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

for i in range(0, len(texts), BATCH_SIZE):
    batch_texts = texts[i : i + BATCH_SIZE]
    batch_num = i // BATCH_SIZE + 1

    encoded_input = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    # Move tokenized inputs to GPU
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    with torch.no_grad():
        model_output = model(**encoded_input)

    # Token embeddings
    token_embeddings = model_output.last_hidden_state

    # Attention mask
    attention_mask = encoded_input["attention_mask"]

    # Mean pooling
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    batch_embeddings = torch.sum(
        token_embeddings * mask,
        dim=1
    ) / torch.clamp(mask.sum(dim=1), min=1e-9)

    all_embeddings.append(batch_embeddings.cpu())

    print(f"Batch {batch_num}/{total_batches} done")

sentence_embeddings = torch.cat(all_embeddings, dim=0)
print(f"Final embeddings shape: {sentence_embeddings.shape}")

# Save embeddings
torch.save(sentence_embeddings, EMBEDDINGS_PATH)
print(f"Saved to {EMBEDDINGS_PATH}")