import torch
import argparse
import numpy as np
from collections import defaultdict

from transformers import BertTokenizer, BertForMaskedLM

softmax = torch.nn.Softmax(dim=-1)

def assess(args, model, tokenizer, sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    tokens[tokens.index('any')] = "[MASK]"

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(tokens)

    input_ids = torch.tensor([input_ids,], dtype=torch.long).to(args.device)
    segment_ids = torch.tensor([segment_ids,], dtype=torch.long).to(args.device)
    
    output = model(input_ids, token_type_ids=segment_ids, output_attentions=False)
    probs = softmax(output.get('logits',output.logits))    
    
    mask_postion = tokens.index('[MASK]')
    mask_probs = probs[0][mask_postion].cpu().detach().numpy()
    mask_indices = [x[0] for x in sorted(enumerate(mask_probs), key=lambda x:-x[1])]

    any_prob = mask_probs[tokenizer.vocab['any']]
    any_rank = mask_indices.index(tokenizer.vocab['any'])
    return any_prob, any_rank


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", default='bert-base-uncased', 
                        help="BERT-like model, only MLM head is required.")
    parser.add_argument("--input_file", required=True, type=str, 
                        help="The input .txt file.")
    parser.add_argument("--output_file", required=True, type=str, 
                        help="The output .tsv file.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device

    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=True)
    model = BertForMaskedLM.from_pretrained(args.model_name_or_path)
    model = model.to(device)
    model.eval()


    with open(args.input_file, encoding='utf-8') as ifh:
        with open(args.output_file, 'w', encoding='utf-8') as ofh:
            for line in ifh:
                sentence1, sentence2 = line.strip().split('\t')
                value1, _ = assess(args, model, tokenizer, sentence1)
                value2, _ = assess(args, model, tokenizer, sentence2)
                winner = 'left' if value1>value2 else 'right'
                print(f'{sentence1}\t{sentence2}\t{winner}', file=ofh)


if __name__ == "__main__":
    main()

