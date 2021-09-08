import torch
import argparse
import numpy as np
from collections import defaultdict

from transformers import GPT2LMHeadModel, GPT2Tokenizer

softmax = torch.nn.Softmax(dim=-1)

def assess(args, model, tokenizer, sentence):
    tokens = ["[CLS]"] + tokenizer.tokenize(sentence)
    tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
    tokens_ids = torch.tensor([tokens_ids,], dtype=torch.long).to(args.device)

    outputs = model(tokens_ids, labels=tokens_ids, return_dict=True)

    return outputs.loss.item()*tokens_ids.size(1)

def score(args, model, tokenizer, sentence):
    return assess(args, model, tokenizer, sentence) / assess(args, model, tokenizer, sentence.replace(' any ', ' '))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", default='gpt2',
                        help="GPT-2-like model, only MLM head is required.")
    parser.add_argument("--input_file", required=True, type=str, 
                        help="The input .tsv file with 2 columns.")
    parser.add_argument("--output_file", required=True, type=str, 
                        help="The output .tsv file.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
    model = model.to(device)
    model.eval()

    with open(args.input_file, encoding='utf-8') as ifh:
        with open(args.output_file, 'w', encoding='utf-8') as ofh:
            for line in ifh:
                sentence1, sentence2 = line.strip().split('\t')
                value1 = score(args, model, tokenizer, sentence1)
                value2 = score(args, model, tokenizer, sentence2)
                winner = 'left' if value1<value2 else 'right'
                print(f'{sentence1}\t{sentence2}\t{winner}', file=ofh)


if __name__ == "__main__":
    main()

