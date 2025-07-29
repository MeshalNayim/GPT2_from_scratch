# llm/data/tokenizer.py

import tiktoken # type: ignore
import torch

class Tokenizer:
    def __init__(self, encoding_name="gpt2"):
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    def encode(self, text):
        return self.tokenizer.encode(text, allowed_special = {'<|endoftext|>'})

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())