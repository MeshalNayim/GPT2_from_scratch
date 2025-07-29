import torch
import time
from llm.config import GPT_CONFIG_124M
from llm.model.gpt2 import GPTModel
from train_and_test.train import train_model
from data.download import download_text
from data.tokenizer import Tokenizer
from data.dataloader import create_dataloader_v1
GPT_CONFIG_124M = GPT_CONFIG_124M.to_dict()

text = download_text("the-verdict.txt")

train_ratio = 0.9
split_id = int(train_ratio*len(text))
train_data = text[:split_id]
val_data = text[split_id:]

train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M['context_length'],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)
tokenizer = Tokenizer()

# Sanity Check
total_characters = len(text)
total_tokens = len(tokenizer.encode(text))

if total_tokens * (train_ratio) < GPT_CONFIG_124M["context_length"]:
    print("Not enough tokens for the training loader. "
          "Try to lower the `GPT_CONFIG_124M['context_length']` or "
          "increase the `training_ratio`")
    
if total_tokens * (1-train_ratio) < GPT_CONFIG_124M["context_length"]:
    print("Not enough tokens for the validation loader. "
          "Try to lower the `GPT_CONFIG_124M['context_length']` or "
          "decrease the `training_ratio`")

# for input_ids, target_ids in train_loader:
#     print("Input IDs shape:", input_ids.shape)
#     print("Target IDs shape:", target_ids.shape)
#     print("Input IDs (first example):", input_ids[0])
#     print("Target IDs (first example):", target_ids[0])
#     break  # Only show one batch

# print(len(train_loader))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Note:
# Uncomment the following code to calculate the execution time
start_time = time.time()

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

num_epochs = 10
train_losses, val_losses, tokens_seen = train_model(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)

# Note:
# Uncomment the following code to show the execution time
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")
torch.save(model.state_dict(),"modelv2.pth")

