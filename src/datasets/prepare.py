import os
import numpy as np
from tqdm import tqdm
import sys

# Add src to the path so we can import the tokenizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer.minigpt_tokenizer import MiniGPTTokenizer

def prepare_dataset(
    input_file: str, 
    tokenizer_path: str, 
    train_output: str, 
    val_output: str, 
    val_ratio: float = 0.1
):
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = MiniGPTTokenizer(tokenizer_path)
    
    # We use uint16 because our vocab size (16,384) fits easily (max 65,535)
    dtype = np.uint16
    
    print(f"Reading data from {input_file}...")
    
    # Read the whole file into memory since it's only 10MB
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.read()
    
    # Split the raw data into documents based on our spacing
    documents = data.split("\n\n")
    documents = [d.strip() for d in documents if d.strip()]
    
    print(f"Loaded {len(documents)} documents.")
    
    # Shuffle documents to ensure mixed train/val splits
    # We use a fixed seed for reproducibility
    np.random.seed(42)
    np.random.shuffle(documents)
    
    val_split_idx = int(len(documents) * val_ratio)
    val_docs = documents[:val_split_idx]
    train_docs = documents[val_split_idx:]
    
    print(f"Split into {len(train_docs)} train docs and {len(val_docs)} val docs.")
    
    # helper function to write a split to a .bin file
    def write_split(docs, out_path, split_name):
        print(f"Tokenizing {split_name} split...")
        
        # We will accumulate tokens and write them in chunks
        chunk_size = 1000000 # 1 Million tokens
        buffer = []
        
        # First calculate total tokens roughly to set up the memory mapped file
        # We don't know exact token count until we tokenize, so we just append to file
        
        with open(out_path, 'wb') as f:
            for doc in tqdm(docs, desc=f"Writing {split_name}"):
                # Encode text
                tokens = tokenizer.encode(doc, return_tensors="list")
                # Add End of Text token
                tokens.append(tokenizer.eot_id)
                buffer.extend(tokens)
                
                # Write to disk if buffer is large enough
                if len(buffer) >= chunk_size:
                    np_array = np.array(buffer, dtype=dtype)
                    f.write(np_array.tobytes())
                    buffer = []
            
            # Write remaining tokens
            if buffer:
                np_array = np.array(buffer, dtype=dtype)
                f.write(np_array.tobytes())
                
        # Get final size
        file_size_bytes = os.path.getsize(out_path)
        total_tokens = file_size_bytes // np.dtype(dtype).itemsize
        print(f"Finished {split_name}: {total_tokens:,} tokens saved to {out_path}.")
        return total_tokens

    os.makedirs(os.path.dirname(train_output), exist_ok=True)
    
    train_tokens = write_split(train_docs, train_output, "train")
    val_tokens = write_split(val_docs, val_output, "val")
    
    print(f"\nTotal tokens dataset: {train_tokens + val_tokens:,}")
    print("Pre-processing complete!")


if __name__ == "__main__":
    input_txt = os.path.join("data", "processed", "clean_sample.txt")
    tokenizer_json = os.path.join("data", "tokenizer", "tokenizer.json")
    train_bin = os.path.join("data", "processed", "train.bin")
    val_bin = os.path.join("data", "processed", "val.bin")
    
    if not os.path.exists(input_txt):
        print(f"Error: {input_txt} not found.")
    elif not os.path.exists(tokenizer_json):
        print(f"Error: {tokenizer_json} not found. Please train tokenizer first.")
    else:
        prepare_dataset(input_txt, tokenizer_json, train_bin, val_bin)
