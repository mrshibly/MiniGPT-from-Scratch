import os
import argparse
from datasets import load_dataset
from tqdm import tqdm

def download_sample(max_bytes):
    print("Downloading FineWeb-Edu sample-10BT...")
    
    # Load dataset...
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    
    output_dir = os.path.join("data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "sample.txt")
    
    shard_size = 10**8 # 100MB per shard
    current_bytes = 0
    
    print(f"Saving ~{max_bytes / (1024*1024):.0f}MB of text to {output_file}...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        # Use a reasonable guess for characters per MB to show a progress bar
        pbar = tqdm(total=max_bytes, unit="B", unit_scale=True, desc="Downloading")
        for idx, row in enumerate(dataset):
            text = row["text"]
            f.write(text + "\n\n")
            
            bytes_added = len(text.encode("utf-8"))
            current_bytes += bytes_added
            pbar.update(bytes_added)
            
            if current_bytes >= max_bytes:
                break
        pbar.close()
        
    print(f"Successfully saved {current_bytes / (1024*1024):.2f} MB of text!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a sample of FineWeb-Edu dataset.")
    parser.add_argument("--max_gb", type=float, default=0.5, help="Maximum amount of data to download in GB (default: 0.5)")
    args = parser.parse_args()
    
    # Calculate max_bytes from GB
    mb_limit = args.max_gb * 1024
    max_bytes = int(args.max_gb * 1024 * 1024 * 1024)
    
    download_sample(max_bytes)
