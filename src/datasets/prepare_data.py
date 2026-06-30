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
    
    os.makedirs(os.path.dirname(train_output), exist_ok=True)
    
    # Open files in binary write mode
    train_f = open(train_output, 'wb')
    val_f = open(val_output, 'wb')
    
    train_buffer = []
    val_buffer = []
    chunk_size = 1000000 # Flush every 1 Million tokens
    
    train_tokens = 0
    val_tokens = 0
    doc_count = 0
    
    print(f"Reading and encoding {input_file} in streaming mode (low memory)...")
    
    # Process the file line by line to prevent loading large files in RAM
    with open(input_file, 'r', encoding='utf-8') as f:
        doc_lines = []
        for line in tqdm(f, desc="Tokenizing dataset stream"):
            stripped = line.strip()
            if not stripped:
                # Empty line marks the end of a document
                if doc_lines:
                    doc = " ".join(doc_lines)
                    doc_lines = []
                    
                    # Encode text and append EOT token
                    tokens = tokenizer.encode(doc, return_tensors="list")
                    tokens.append(tokenizer.eot_id)
                    
                    # Deterministic split: 90% train, 10% val (every 10th document is val)
                    if doc_count % 10 == 0:
                        val_buffer.extend(tokens)
                        if len(val_buffer) >= chunk_size:
                            np_array = np.array(val_buffer, dtype=dtype)
                            val_f.write(np_array.tobytes())
                            val_tokens += len(val_buffer)
                            val_buffer = []
                    else:
                        train_buffer.extend(tokens)
                        if len(train_buffer) >= chunk_size:
                            np_array = np.array(train_buffer, dtype=dtype)
                            train_f.write(np_array.tobytes())
                            train_tokens += len(train_buffer)
                            train_buffer = []
                    
                    doc_count += 1
            else:
                doc_lines.append(stripped)
                
        # Process the final document if the file didn't end with an empty line
        if doc_lines:
            doc = " ".join(doc_lines)
            tokens = tokenizer.encode(doc, return_tensors="list")
            tokens.append(tokenizer.eot_id)
            if doc_count % 10 == 0:
                val_buffer.extend(tokens)
            else:
                train_buffer.extend(tokens)
            doc_count += 1
            
    # Flush remaining tokens in buffer
    if train_buffer:
        np_array = np.array(train_buffer, dtype=dtype)
        train_f.write(np_array.tobytes())
        train_tokens += len(train_buffer)
    if val_buffer:
        np_array = np.array(val_buffer, dtype=dtype)
        val_f.write(np_array.tobytes())
        val_tokens += len(val_buffer)
        
    train_f.close()
    val_f.close()
    
    print(f"\nPreprocessing Complete!")
    print(f"Total Documents: {doc_count:,}")
    print(f"Train Tokens: {train_tokens:,} saved to {train_output}")
    print(f"Val Tokens: {val_tokens:,} saved to {val_output}")

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
