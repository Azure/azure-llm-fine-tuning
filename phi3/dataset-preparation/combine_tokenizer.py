"""
Given two tokenizers, combine them and create a new tokenizer
Usage: python combine_tokenizers.py --tokenizer1 ../config/en/roberta_8 --tokenizer2 ../config/hi/roberta_8 --save_dir ../config/en/en_hi/roberta_8
"""

# Libraries for tokenizer
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
import argparse
import json
import os
from tqdm import tqdm
from transformers import AutoTokenizer
from timeit import default_timer as timer
import sys

def combine_tokenizers(args):
    # Load both the json files, take the union, and store it
    json1 = json.load(open(os.path.join(args.tokenizer1, 'vocab.json')))
    json2 = json.load(open(os.path.join(args.tokenizer2, 'vocab.json')))

    # Create a new vocabulary
    new_vocab = {}
    idx = 0
    for word in json1.keys():
        if word not in new_vocab.keys():
            new_vocab[word] = idx
            idx += 1

    # Add words from second tokenizer
    for word in json2.keys():
        if word not in new_vocab.keys():
            new_vocab[word] = idx
            idx += 1

    # Make the directory if necessary
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Save the vocab
    with open(os.path.join(args.save_dir, 'vocab.json'), 'w') as fp:
        json.dump(new_vocab, fp, ensure_ascii=False)

    # Merge the two merges file. Don't handle duplicates here
    # Concatenate them, but ignore the first line of the second file
    os.system('cat {} > {}'.format(os.path.join(args.tokenizer1, 'merges.txt'), os.path.join(args.save_dir, 'merges.txt')))
    os.system('tail -n +2 -q {} >> {}'.format(os.path.join(args.tokenizer2, 'merges.txt'), os.path.join(args.save_dir, 'merges.txt')))

    # Save other files
    os.system('cp {} {}'.format(os.path.join(args.tokenizer1, 'special_tokens_map.json'), args.save_dir))
    os.system('cp {} {}'.format(os.path.join(args.tokenizer1, 'tokenizer_config.json'), args.save_dir))

    # Instantiate the new tokenizer
    #tokenizer = AutoTokenizer.from_pretrained(args.save_dir, use_fast=True)

def main():
    parser = argparse.ArgumentParser()

    # Dataset Arguments
    parser.add_argument("--tokenizer1", type=str, default="tokenizer-minstral-orig-txt", help="")
    parser.add_argument("--tokenizer2", type=str, default="tokenizer-ko-nsmc-txt", help="")
    parser.add_argument("--save_dir", type=str, default="tokenizer-new-1000", help="")

    args = parser.parse_args()
    combine_tokenizers(args)

if __name__ == '__main__':
    main()