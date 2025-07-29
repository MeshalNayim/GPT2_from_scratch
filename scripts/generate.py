import torch
from llm.model.gpt2 import GPTModel
from llm.config import GPT_CONFIG_124M
from data.tokenizer import Tokenizer,text_to_token_ids,token_ids_to_text
from generate.generate import generate_text_simple

GPT_CONFIG_124M = GPT_CONFIG_124M.to_dict()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GPTModel(GPT_CONFIG_124M)

model.load_state_dict(torch.load("modelv1.pth",map_location=device))

model.to(device)

prompt = "Hey, How are you"

tokenizer = Tokenizer()
idx = text_to_token_ids(prompt,tokenizer)

reply_id = generate_text_simple(model = model,idx = idx, max_new_tokens=15, context_size= 256)

reply = token_ids_to_text(reply_id,tokenizer)

print(reply)