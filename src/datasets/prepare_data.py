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
    
    # We read the file in chunks of ~50 MB to keep RAM usage extremely low (~100-200MB)
    chunk_size_bytes = 50 * 1024 * 1024
    
    train_tokens = 0
    val_tokens = 0
    doc_count = 0
    
    total_size = os.path.getsize(input_file)
    print(f"Reading and encoding {input_file} in multi-threaded batch mode (extremely fast)...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        pbar = tqdm(total=total_size, unit="B", unit_scale=True, desc="Encoding dataset")
        leftover_text = ""
        
        while True:
            chunk = f.read(chunk_size_bytes)
            if not chunk:
                break
            
            pbar.update(len(chunk.encode('utf-8')))
            
            # Combine with leftover from previous chunk
            text = leftover_text + chunk
            
            # Find the last document boundary \n\n in the text to avoid cutting documents
            last_boundary = text.rfind("\n\n")
            if last_boundary != -1:
                chunk_text = text[:last_boundary]
                leftover_text = text[last_boundary+2:]
            else:
                chunk_text = text
                leftover_text = ""
                
            # Split into documents
            documents = [d.strip() for d in chunk_text.split("\n\n") if d.strip()]
            if not documents:
                continue
                
            # Use Hugging Face's multi-threaded Rust batch encoder for speed
            encodings = tokenizer.tokenizer.encode_batch(documents)
            
            train_buffer = []
            val_buffer = []
            
            for enc in encodings:
                tokens = enc.ids
                # Append End Of Text token
                tokens.append(tokenizer.eot_id)
                
                # Deterministic train/val split (90% train, 10% val)
                if doc_count % 10 == 0:
                    val_buffer.extend(tokens)
                else:
                    train_buffer.extend(tokens)
                doc_count += 1
                
            # Write buffers directly to disk
            if train_buffer:
                np_array = np.array(train_buffer, dtype=dtype)
                train_f.write(np_array.tobytes())
                train_tokens += len(train_buffer)
            if val_buffer:
                np_array = np.array(val_buffer, dtype=dtype)
                val_f.write(np_array.tobytes())
                val_tokens += len(val_buffer)
                
        # Process any remaining text in leftover
        if leftover_text.strip():
            documents = [d.strip() for d in leftover_text.split("\n\n") if d.strip()]
            encodings = tokenizer.tokenizer.encode_batch(documents)
            
            train_buffer = []
            val_buffer = []
            for enc in encodings:
                tokens = enc.ids
                tokens.append(tokenizer.eot_id)
                if doc_count % 10 == 0:
                    val_buffer.extend(tokens)
                else:
                    train_buffer.extend(tokens)
                doc_count += 1
                
            if train_buffer:
                np_array = np.array(train_buffer, dtype=dtype)
                train_f.write(np_array.tobytes())
                train_tokens += len(train_buffer)
            if val_buffer:
                np_array = np.array(val_buffer, dtype=dtype)
                val_f.write(np_array.tobytes())
                val_tokens += len(val_buffer)
                
        pbar.close()
        
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
