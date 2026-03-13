import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

def train_minigpt_tokenizer(input_file: str, output_file: str, vocab_size: int = 16384):
    """
    Trains a Byte-Pair Encoding (BPE) tokenizer on the provided text file.
    """
    print(f"Training BPE tokenizer on {input_file}...")
    
    # Initialize a tokenizer with a BPE model
    tokenizer = Tokenizer(BPE(unk_token=None))
    
    # Use ByteLevel pre-tokenizer. This splits on whitespace and punctuation,
    # but preserves spaces as part of the tokens (like GPT-2/GPT-3 does).
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    
    # Our only special token for now is the end-of-text marker
    special_tokens = ["<|endoftext|>"]
    
    # Initialize the trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True
    )
    
    # Train the tokenizer
    tokenizer.train(files=[input_file], trainer=trainer)
    
    # We must add the byte-level decoder so that when we decode the token IDs,
    # the strange byte-level characters (like 'Ġ' for space) are converted back
    # into normal strings.
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    tokenizer.decoder = ByteLevelDecoder()

    # Ensure output directory exists and save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    tokenizer.save(output_file)
    
    print(f"Successfully trained and saved tokenizer to {output_file}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")

if __name__ == "__main__":
    input_txt = os.path.join("data", "processed", "clean_sample.txt")
    output_json = os.path.join("data", "tokenizer", "tokenizer.json")
    
    if not os.path.exists(input_txt):
        print(f"Error: Could not find {input_txt}. Please run clean_text.py first.")
    else:
        train_minigpt_tokenizer(input_txt, output_json)
