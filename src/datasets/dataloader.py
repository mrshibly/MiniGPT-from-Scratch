import os
import numpy as np
import torch

class TokenDataLoader:
    def __init__(self, data_file: str, device: str = "cpu"):
        """
        Loads a binary token dataset using memory mapping.
        This allows us to handle datasets larger than RAM.
        """
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found at {data_file}")
            
        self.device = device
        
        # Load the binary file as uint16
        print(f"Loading {data_file} via memmap...")
        self.data = np.memmap(data_file, dtype=np.uint16, mode='r')
        self.total_tokens = len(self.data)
        print(f"Dataset contains {self.total_tokens:,} tokens.")
        
    def get_batch(self, batch_size: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates a batch of input (X) and target (Y) sequences.
        Target sequences are simply input sequences shifted by 1 token.
        """
        # We need seq_len + 1 tokens for each sample to create X and Y
        # Generate random starting indices
        max_idx = self.total_tokens - seq_len - 1
        
        # If the file is too small
        if max_idx <= 0:
            raise ValueError(f"Dataset too small ({self.total_tokens} tokens) for seq_len {seq_len}.")
            
        ix = torch.randint(0, max_idx, (batch_size,))
        
        # Read the chunks natively via numpy slicing, then convert to torch LongTensors
        # Numpy fancy indexing is slow with memmap, so a loop list comprehension is often safer/faster.
        x_list = [self.data[i : i + seq_len].astype(np.int64) for i in ix]
        y_list = [self.data[i + 1 : i + seq_len + 1].astype(np.int64) for i in ix]
        
        # Stack into batch tensors
        x = torch.tensor(np.stack(x_list), dtype=torch.long, device=self.device)
        y = torch.tensor(np.stack(y_list), dtype=torch.long, device=self.device)
        
        return x, y


if __name__ == "__main__":
    # Smoke test the DataLoader
    train_bin = os.path.join("data", "processed", "train.bin")
    
    if not os.path.exists(train_bin):
        print(f"Error: {train_bin} not found. Run prepare.py first.")
    else:
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Detected device: {device}")
        
        loader = TokenDataLoader(train_bin, device=device)
        
        batch_size = 4
        seq_len = 16
        
        X, Y = loader.get_batch(batch_size, seq_len)
        
        print(f"\nBatch Size: {batch_size}")
        print(f"Sequence Length: {seq_len}")
        print(f"X shape: {X.shape}")
        print(f"Y shape: {Y.shape}")
        
        print("\nFirst sample in batch:")
        print(f"X[0]: {X[0].tolist()}")
        print(f"Y[0]: {Y[0].tolist()}")
        
        # Verify the shift
        is_shifted = (X[0][1:].tolist() == Y[0][:-1].tolist())
        print(f"\nTarget Shift Verification passed? {is_shifted}")
