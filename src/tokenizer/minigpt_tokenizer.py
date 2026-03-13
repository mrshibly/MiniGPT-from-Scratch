import os
from tokenizers import Tokenizer

class MiniGPTTokenizer:
    """
    A simple wrapper around the Hugging Face Tokenizers object that implements
    encode and decode for our custom trained GPT tokenizer.
    """
    def __init__(self, tokenizer_path: str):
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")
        
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # End of Text token ID
        self.eot_id = self.tokenizer.token_to_id("<|endoftext|>")
        
        if self.eot_id is None:
            raise ValueError("<|endoftext|> token not found in vocab!")

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str, return_tensors: str = "pt") -> list:
        """
        Encodes a string into a list of token IDs (integers).
        
        If return_tensors == "pt", returns a PyTorch tensor (requires torch).
        Default returns a simple list.
        """
        tokens = self.tokenizer.encode(text).ids
        
        if return_tensors == "pt":
            import torch
            return torch.tensor(tokens, dtype=torch.long)
            
        return tokens

    def decode(self, token_ids: list) -> str:
        """
        Decodes a list of token IDs back into a string.
        """
        if hasattr(token_ids, "tolist"): # If it's a tensor
            token_ids = token_ids.tolist()
            
        return self.tokenizer.decode(token_ids)

if __name__ == "__main__":
    # Test our wrapper
    tokenizer_path = os.path.join("data", "tokenizer", "tokenizer.json")
    if os.path.exists(tokenizer_path):
        tokenizer = MiniGPTTokenizer(tokenizer_path)
        print(f"Loaded tokenizer with vocab size {tokenizer.vocab_size}")
        print(f"EOT Token ID: {tokenizer.eot_id}")
        
        test_string = "Hello, tokenizer! Can you read this?"
        tokens = tokenizer.encode(test_string, return_tensors="list")
        decoded = tokenizer.decode(tokens)
        
        print("\nTest input:")
        print(f"Text: '{test_string}'")
        print(f"IDs:  {tokens}")
        print(f"Decoded: '{decoded}'")
        print("Success!" if test_string == decoded else "Failed to cleanly decode.")
    else:
        print("Run `train_tokenizer.py` first to create tokenizer.json.")
