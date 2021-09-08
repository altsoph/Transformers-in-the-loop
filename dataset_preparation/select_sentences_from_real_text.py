import sys
from nltk import word_tokenize, sent_tokenize

if len(sys.argv)<2:
	print('specify input filename')
	exit()
fn = sys.argv[1]

STOPLIST = ('not', 'none', 'even', 'yet')
MODAL_VERBS = ("ain't","needn't","mustn't","oughtn't","wasn't","weren't","daren't","can't","shouldn't","couldn't")

c0, c1 = 0, 0
seen = set()
seen0 = set()

def parse_words(words):

	prev_not_word = None
	not_word	  = None
	next_not_word = None
	prev_any_word = None
	next_any_word = None

	for idx, word in enumerate(words):
		if "n't" in word:
			not_word = (words[idx-1]+word).lower()
			if idx>1: prev_not_word = words[idx-2].lower()
			if idx<len(words)-1: next_not_word = words[idx+1].lower()
		if word == 'any':
			if idx: prev_ang_word = words[idx-1].lower()
			if idx<len(words)-1: next_any_word = words[idx+1].lower()
	return prev_not_word, not_word, next_not_word, prev_any_word, next_any_word

for line in sent_tokenize(open(fn, encoding='utf-8').read().replace('\n', ' ')):
	sent = line.strip()
	sent = sent.strip("\'\"\“,вЂ™«")
	seen0.add( sent )

	c0 += 1

	# we are looking for simple and clean sentences like 'You don't have any flags up.'

	# remove cases with multiple 'any'
	if sent.count('any') != 1: continue
	# remove cases with multiple negation
	if sent.count("n't") != 1: continue
	# remove extremely short/long sentences
	if not(20<len(sent)<200): continue
	# we are interested only in cases there negation and 'any' are close enough
	if not(1<=sent.index('any') - sent.index("n't")<16): continue

	words = list(word_tokenize(sent))
	prev_not_word, not_word, next_not_word, prev_any_word, next_any_word = parse_words(words)

	# remove questions
	if "?" in sent: continue
	if prev_not_word is None: continue # not-word is the first one
	if prev_not_word.endswith(','): continue # , aren't we?

	# remove modal verb cases
	if not_word in MODAL_VERBS: continue

	# remove too complex cases
	if prev_not_word == 'and': continue
	if next_not_word in ('really', 'ever'): continue
	if not next_any_word: continue
	if next_any_word[0].isupper(): continue 
	if set(words)&set(STOPLIST): continue

	# we need only plurals
	if not next_any_word.endswith('s'): continue 
	# remove mass nouns cases ('any business')
	if next_any_word.endswith('ss'): continue 
	# remove cases like 'any children's toy'
	if next_any_word.endswith("'s"): continue 

	# remove frequent 'any' collocations irrelevant for us
	if next_any_word in ('means', 'was') or prev_any_word == 'just': continue

	# remove duplicates
	if sent in seen: continue
	seen.add(sent)
	print(sent)
	c1 += 1
	pass

# print(c1, c0)
