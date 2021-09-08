import pandas as pd
import functools
from pattern.en import conjugate, pluralize, lemma, lexeme,PRESENT,SG,PAST

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

# load all generated sentences
# and take best 20k of them
df = pd.read_csv('generated_sentences.csv').drop(columns=['Unnamed: 0'])
df = df.head(20000)

# generate negations and assertions
negation = []
assertion = []
for index,row in df.iterrows():
    words = row['sentence'].split(' ')
    neg = ''.join([words[0], ' ', words[1], ' didn\'t ', lemma(words[2]), ' any ',pluralize(words[4].strip('.')),'.'])
    ass = ''.join([words[0], ' ', words[1], ' ', words[2], ' any ',pluralize(words[4].strip('.')),'.'])
    negation.append(neg)
    assertion.append(ass)
# save them
with open('20k_sentences_with_negation.txt', 'w') as f:
    for item in negation:
        f.write("%s\n" % item)
with open('20k_sentences_without_negation.txt', 'w') as f:
    for item in assertion:
        f.write("%s\n" % item)

# generate quantifiers with lexical noun
no_lexical = []
some_lexical = []
for index,row in df.iterrows():
    words = row['sentence'].split(' ')
    sent_no = ''.join(['No ', pluralize(words[1]), ' ', words[2], ' any ',pluralize(words[4].strip('.')),'.'])
    sent_some = ''.join(['Some ', pluralize(words[1]), ' ', words[2], ' any ',pluralize(words[4].strip('.')),'.'])
    no_lexical.append(sent_no)
    some_lexical.append(sent_some)
# and save them
with open('20k_sentences_with_no_and_subject.txt', 'w') as f:
    for item in no_lexical:
        f.write("%s\n" % item)
with open('20k_sentences_with_some_and_subject.txt', 'w') as f:
    for item in some_lexical:
        f.write("%s\n" % item)

# generate sentences with many and few
few = []
many = []
for index,row in df.iterrows():
    words = row['sentence'].split(' ')
    sent_few = ''.join(['Few ', pluralize(words[1]), ' ', words[2], ' any ',pluralize(words[4].strip('.')),'.'])
    sent_many = ''.join(['Many ', pluralize(words[1]), ' ', words[2], ' any ',pluralize(words[4].strip('.')),'.'])
    few.append(sent_few)
    many.append(sent_many)
# and save them
with open('20k_sentences_with_few.txt', 'w') as f:
    for item in few:
        f.write("%s\n" % item)
with open('20k_sentences_with_many.txt', 'w') as f:
    for item in many:
        f.write("%s\n" % item)


# generate sentences with fewer/more/exactly/between/at least/at most
fewer = []
more = []
exactly = []
between = []
at_least = []
at_most = []
for index,row in df.iterrows():
    words = row['sentence'].split(' ')
    sent_fewer = ''.join(['Fewer than five ', pluralize(words[1]), ' ', words[2], ' any ',pluralize(words[4].strip('.')),'.'])
    sent_more = ''.join(['More than five ', pluralize(words[1]), ' ', words[2], ' any ',pluralize(words[4].strip('.')),'.'])
    sent_exactly = ''.join(['Exactly five ', pluralize(words[1]), ' ', words[2], ' any ',pluralize(words[4].strip('.')),'.'])
    sent_between = ''.join(['Between five and ten ', pluralize(words[1]), ' ', words[2], ' any ',pluralize(words[4].strip('.')),'.'])
    sent_at_least = ''.join(['At least five ', pluralize(words[1]), ' ', words[2], ' any ',pluralize(words[4].strip('.')),'.'])
    sent_at_most = ''.join(['At most five ', pluralize(words[1]), ' ', words[2], ' any ',pluralize(words[4].strip('.')),'.'])
    fewer.append(sent_fewer)
    more.append(sent_more)
    exactly.append(sent_exactly)
    between.append(sent_between)
    at_least.append(sent_at_least)
    at_most.append(sent_at_most)
# and save them
with open('20k_sentences_with_fewer.txt', 'w') as f:
    for item in fewer:
        f.write("%s\n" % item)
with open('20k_sentences_with_more.txt', 'w') as f:
    for item in more:
        f.write("%s\n" % item)
with open('20k_sentences_with_exactly.txt', 'w') as f:
    for item in exactly:
        f.write("%s\n" % item)
with open('20k_sentences_with_between.txt', 'w') as f:
    for item in between:
        f.write("%s\n" % item)
with open('20k_sentences_with_at_least.txt', 'w') as f:
    for item in at_least:
        f.write("%s\n" % item)
with open('20k_sentences_with_at_most.txt', 'w') as f:
    for item in at_most:
        f.write("%s\n" % item)


body = ['man','woman','girl','player','team','family','member','student','country',
    'mother','community','brother','school','wife','class','town','band','state',
    'church','government','director','university','society','office','union',
    'board','division','head','court','district','region']

thing = ['game','book','fire','world','career','life','hand','title','day','rock',
    'arm','project','version','level','program','door','record','room','story',
    'season','field','land','city','war','election','music','home','hair','thing',
    'name','face','part','position','century','line','number','force','year','eye',
    'power','heart','football','championship','light','death','set','way','system',
    'station','site','area','production','service','age','road','place','word','mind',
    'point','form','law','period','air','summer','side','development','role','use']


# generate sentences with noones/someones/...
no = []
some = []
for index,row in df.iterrows():
    words = row['sentence'].split(' ')
    if words[1] in body:
        sentence1 = ''.join(['Nobody ', words[2], ' any ',pluralize(words[4].strip('.')),'.'])
        sentence2 = ''.join(['No one ', words[2], ' any ',pluralize(words[4].strip('.')),'.'])
        no.append(sentence1)
        no.append(sentence2)
        sentence1 = ''.join(['Somebody ', words[2], ' any ',pluralize(words[4].strip('.')),'.'])
        sentence2 = ''.join(['Someone ', words[2], ' any ',pluralize(words[4].strip('.')),'.'])
        some.append(sentence1)
        some.append(sentence2)
    else:
        sentence = ''.join(['Nothing ', words[2], ' any ',pluralize(words[4].strip('.')),'.'])
        no.append(sentence)
        sentence = ''.join(['Something ', words[2], ' any ',pluralize(words[4].strip('.')),'.'])
        some.append(sentence)
# and save them
with open('8k_sentences_with_nobody.txt', 'w') as f:
    for item in no:
        f.write("%s\n" % item)
with open('8k_sentences_with_somebody.txt', 'w') as f:
    for item in some:
        f.write("%s\n" % item)        


