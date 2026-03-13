import os
import re
from tqdm import tqdm

def clean_text_file(input_path: str, output_path: str):
    """
    Reads a raw text file, applies light cleaning, and writes to an output file.
    """
    print(f"Reading from {input_path}...")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Get total file size for progress bar
    total_size = os.path.getsize(input_path)
    
    processed_size = 0
    docs_kept = 0
    docs_removed = 0
    
    print(f"Writing to {output_path}...")
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
         
        # We read the file chunk by chunk based on double newlines (documents)
        # Because we saved it with "\n\n" between documents in download_fineweb.py
        doc_buffer = []
        
        pbar = tqdm(total=total_size, unit="B", unit_scale=True, desc="Cleaning")
        
        for line in f_in:
            processed_size += len(line.encode('utf-8'))
            pbar.update(len(line.encode('utf-8')))
            
            if line.strip() == "":
                # Document boundary reached
                if doc_buffer:
                    doc = " ".join(doc_buffer)
                    doc = clean_document(doc)
                    if len(doc) > 20: # Keep documents with reasonable length
                        f_out.write(doc + "\n\n")
                        docs_kept += 1
                    else:
                        docs_removed += 1
                    doc_buffer = []
            else:
                doc_buffer.append(line.strip())
                
        # Process last doc
        if doc_buffer:
            doc = " ".join(doc_buffer)
            doc = clean_document(doc)
            if len(doc) > 20:
                f_out.write(doc + "\n\n")
                docs_kept += 1
            else:
                docs_removed += 1
                
        pbar.close()
        
    print(f"\nDone! Kept: {docs_kept} docs. Removed: {docs_removed} docs.")
    print(f"Clean file size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")


def clean_document(text: str) -> str:
    """
    Applies regex cleaning to a single document.
    """
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    return text.strip()


if __name__ == "__main__":
    raw_file = os.path.join("data", "raw", "sample.txt")
    clean_file = os.path.join("data", "processed", "clean_sample.txt")
    
    if not os.path.exists(raw_file):
        print(f"Error: Could not find {raw_file}.")
        print("Please run src/datasets/download_fineweb.py first.")
    else:
        clean_text_file(raw_file, clean_file)
