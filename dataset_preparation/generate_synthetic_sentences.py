import re
import torch
import functools
import numpy as np
import pandas as pd 
from nltk import word_tokenize
from pytorch_pretrained_bert import BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer

from pattern.en import conjugate, pluralize, lemma, lexeme,PRESENT,SG,PAST
import en_core_web_sm

# this patch is for running pattern under py3.7+, check it here https://github.com/clips/pattern/pull/294
def patch_pattern():
    from pattern import text
    original_read = text._read
    @functools.wraps(original_read)
    def patched_read(*args, **kwargs):
        try:
            for r in original_read(*args, **kwargs):
                yield r
        except RuntimeError:
            pass
    text._read = patched_read
patch_pattern()


# get words from the bert's tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
words=[]
for item in tokenizer.vocab.items():
    if re.match('[a-z]{2,}$',item[0]):
        words.append(item[0])

# select verbs and nouns
nlp = en_core_web_sm.load()
nouns = []
verbs = []
for word in words:
    if [t for t in nlp(word)][0].pos_ == 'NOUN' and len(nouns) < 150:
        nouns.append([t for t in nlp(word)][0].text)
    elif [t for t in nlp(word)][0].pos_ == 'VERB' and len(verbs) < 2000:
        verbs.append([t for t in nlp(word)][0].text)

# load transitive verbs
with open('transitive-verbs.txt', 'r') as f:
    transitive_verbs = f.readlines()
transitive_verbs = [x.strip('\n') for x in transitive_verbs]

modal = ['would','can','may','will','need','seem','allow','require']
verbs_n=[]
for verb in verbs:
    if (not lemma(verb) in verbs_n) and (not lemma(verb) in modal) and (lemma(verb) in transitive_verbs):
        verbs_n.append(lemma(verb))

nouns_n=[]
for noun in nouns:
    if not lemma(noun) in nouns_n:
        nouns_n.append(lemma(noun))

noun_errors = ['television','information','police','other','military','feet','support','education','specy','busines',
               'half','research','history','help','art','people','work','time','build','children','homes','end','think',
              'women','wasn','men','love','water']

nouns_n = list(set(nouns_n).difference(set(noun_errors)))[:100]

vow = ['e','a', 'i','o']
sentences = []
for subj in nouns_n:
    for verb in verbs_n:
        for dir_obj in nouns_n:
            if not subj == dir_obj:
                if subj[0] in vow and dir_obj[0] in vow:
                    sentences.append('An '+subj+ ' ' + conjugate(verb=verb,tense=PAST,number=SG) + ' an ' + dir_obj + '.')
                elif subj[0] in vow and (not dir_obj[0] in vow):
                    sentences.append('An '+subj+ ' ' + conjugate(verb=verb,tense=PAST,number=SG) + ' a ' + dir_obj + '.')
                elif (not subj[0] in vow) and dir_obj[0] in vow:
                    sentences.append('A '+subj+ ' ' + conjugate(verb=verb,tense=PAST,number=SG) + ' an ' + dir_obj + '.')
                else:
                    sentences.append('A '+subj+ ' ' + conjugate(verb=verb,tense=PAST,number=SG) + ' a ' + dir_obj + '.')


model_id = 'gpt2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_id)

def process(sentence):
    tokens = ["[CLS]"] + tokenizer.tokenize(sentence)
    tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
    tokens_ids = torch.tensor([tokens_ids,], dtype=torch.long).to(device)
    with torch.no_grad():
        outputs = model(tokens_ids, lm_labels=tokens_ids)
        log_likelihood = outputs.item()
    return np.exp(log_likelihood) 

pairs = {}
for sentence in sentences:
    pairs[sentence] = process(sentence)

df = pd.DataFrame.from_dict(pairs, orient='index').reset_index()
df = df.rename(columns={"index": "sentence", 0: "perplexity"})
df.sort_values(by='perplexity', ascending=True).to_csv('generated_sentences.csv')


