import os
import time
import pandas as pd
import urllib.request
from pathlib import Path
from datasets import load_dataset
from tokenizers import SentencePieceBPETokenizer, ByteLevelBPETokenizer

def save_dataset_to_txt(txt_column, txt_dir, hf_dataset_id):
    dataset = load_dataset(hf_dataset_id)
    os.makedirs(txt_dir, exist_ok=True)
    for split_key in dataset.keys():
        doc_path = f"{txt_dir}/{split_key}.txt"
        with open(doc_path, 'w') as f:
            for doc in dataset[split_key][txt_column]:
                f.write(doc+'\n')
                
def main():
    IS_BBPE = True
    path_namuwiki = [str(x) for x in Path("namuwiki-extracted-txt").glob("*.txt")]
    path_wiki = [str(x) for x in Path("wiki-txt").glob("*.txt")]
    path_kcbert = [str(x) for x in Path("kcbert2-txt").glob("*.txt")]    
    path_corpus = path_namuwiki + path_wiki + path_kcbert
    
    vocab_size = 18000
    limit_alphabet = 1000
    min_frequency = 30

    if IS_BBPE:
        tokenizer = ByteLevelBPETokenizer(unicode_normalizer="nfkc", trim_offsets=True)
        t1 = time.time()

        tokenizer.train(
            files=path_corpus,
            vocab_size=vocab_size,
            special_tokens=["<|endoftext|>"],
            min_frequency=min_frequency, 
            show_progress=True
        )

        tokenizer.save('korean_tokenizer_bbpe.json') 
        print("Elapsed time:", time.time() - t1)
        
    else:
        tokenizer = SentencePieceBPETokenizer(fuse_unk=True)
        t1 = time.time()

        tokenizer.train(
            files=path_corpus,
            vocab_size=vocab_size,
            special_tokens=["<unk>", "<s>", "</s>"],
            min_frequency=min_frequency, 
            limit_alphabet=limit_alphabet,
            show_progress=True
        )

        tokenizer.save('korean_tokenizer_bpe.json') 
        print("Elapsed time:", time.time() - t1)

if __name__ == '__main__':
    main()