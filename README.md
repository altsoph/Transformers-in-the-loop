# Transformers-in-the-loop

This repo contains supplemental materials accompanying a preprint version of the "Transformers in the loop: Polarity in neural models of language" paper. 
Detailed explanation concerning the generation and use of this dataset can be found in the text of the paper.

## Abstract

*Representation of linguistic phenomena in computational language models is typically assessed against the predictions of existing linguistic theories of these phenomena. Using the notion of polarity as a case study, we show that this is not always the most adequate set-up. We probe polarity via so-called 'negative polarity items' (in particular, English {\it any}) in two pre-trained Transformer-based models (BERT and GPT-2). We show that -- at least for polarity -- metrics derived from language models are more consistent with data from psycholinguistic experiments than linguistic theory predictions. Establishing this allows us to more adequately evaluate the performance of language models and also to use language models to discover new insights into natural language grammar beyond existing linguistic theories. Overall, our results encourage a closer tie between experiments with human subjects and with language models. We propose methods to enable this closer tie, with language models as part of experimental pipeline, and show this pipeline at work.*

## Code

To run these scripts you'll need huggingface [transformers](https://github.com/huggingface/transformers) library, any version starting from 4.0.1 should work.
There are two similar scripts, 
  * `compare_pair_by_bert.py` makes an acceptability assessment of two sentences with `any` and points to more proper, using BERT;
  * `compare_pair_by_gpt.py` does the same but using GPT-2 model.

### How can I run it?
* Use the input file in .tsv format with two columns -- each with one of the sentences to compare.
* Run scripts like this:

```python scripts/compare_pair_by_bert.py --input_file datasets/real_positive_or_negative_with_any.tsv --output_file results.tsv``` 

## Natural data

We scraped the Gutenberg Project and a subset of English Wikipedia to obtain the list of sentences that contain *any*. Next, using a combination of heuristics, we filtered the result with regular expressions to produce two sets of sentences (the second set underwent additional manual filtration):
   * 3844 sentences with sentential negation and a plural object with *any* to the right to the verb;
   * 330 sentences with *nobody* / *no one* as subject and a plural object with *any* to the right.

The first set was modified to substitute the negated verb by its non-negated version, so we contrast 3844 sentences with negation and 3844 affirmative ones (*neg* vs. *aff*). In the second dataset, we substituted *nobody* for *somebody* and *no one* for *someone*, to check the *some* vs. *no* contrast.

### How to reproduce

You can use our script to find sentences with negation and *any* in any given English corpus:

```python dataset_preparation/select_sentences_from_real_text.py <corpus.txt>```

This script requires the `nltk` module.

## Synthetic data

We used the following procedure. First, we automatically identified the set of verbs and nouns to build our items from. To do so, we started with *bert-base-uncased* vocabulary. We ran all non-subword lexical tokens through a SpaCy POS. Further, we lemmatized the result using https://pypi.org/project/Pattern/ and dropped duplicates. Then, we filtered out modal verbs, singularia tantum nouns and some visible lemmatization mistakes. Finally, we filtered out non-transitive verbs to give the dataset a bit of a higher baseline of grammaticality.

We kept top 100 nouns and top 100 verbs from the resulting lists -- these are the lexical entries we will deal with. Then, we generated sentences with these words. For this, we iterate over the 100 nouns in the subject and the object positions (excluding cases where the same noun appears in both positions) and over the 100 verbs. The procedure gave us 990k sentences like these:

  * A girl crossed a road.
  * A community hosted a game.
  * An eye opened a fire.
  * A record put an air.

Some are more natural, make more sense and adhere to the verb's selectional restrictions better than the others. To control for this, we ran the sentences through GPT-2 and assigned perplexity to all candidates. Then we took the bottom 20k of the sentences (the most 'natural' ones) as the core of our synthetic dataset.

We tried to approximate the 'naturalness' of examples by a combination of measures. We rely on  insights from different models (GPT-2, BERT,  corpus-based statistical insights into verb transitivity) on different stages of the dataset creation. Still, some sentences sound intuitively 'weird'. We don't see this as a problem though -- we will not rely directly on the naturalness of individual the examples, rather we will measure the effect of the NPI across the dataset. The  amount of the examples will allow us to generalize across varying parts of the sentences to make sure that the results can be attributed to the parts we are interested in: items responsible for the monotonicity of the sentence. The quantity of test items is crucial for reproducing psycholinguistic experiments on LRMs -- while in the former one sentence gives rise to a number of observations when different human subjects make a judgment, in the latter one test sentence gives you one observation only.% Here the procedures of psycholinguistic studies and LRM studies necessarily diverge.

With this in mind, we use the 20k sentences produced by the previous steps to build the parts of our synthetic dataset. Each of the sentences has a pluralized (not anymore singular!)  object in combination with *any*: any roads. The subject type varies in different datasets comprising our synthetic data. 

Overall, sentences in all parts of our dataset vary in the type of context it instantiates (simple affirmative, negation, quantifiers of different monotonicity) -- but all sentences contain *any* in the object position in combination with a plural noun. We will manipulate the presence or absence of *any* to measure how *any* plays out with different types of environments.

### How to reproduce

You can use our scripts to generate our synthetic datasets from scratch.

To do so, you'll need the following libraries installed:
* nltk
* textacy
* pattern
* spacy (with the `en_core_web_sm` model)
* pandas
* numpy
* torch
* pytorch_pretrained_bert 

Here are the instructions:
1. First, use `python generate_transitive_verbs_list.py` to re-generate the list of transitive verbs (the script will store it in `transitive-verbs.txt`).
2. Then, use `python generate_synthetic_sentences.py` to generate a lot of sentences with their perplexity scores (it uses `transitive-verbs.txt`).
3. Finally, use `python format_examples.py` to select the best from generated sentences and produce all our datasets.

## Human assessment data

We also include the results of our psycholinguistic experiment on human subjects. You can find the details of the experiment in the paper. 

The raw results are provided in the file `datasets/human_assessments.tsv`, which has five columns:
  * the text of the *left* sentence;
  * the text of the *right* sentence;
  * the answer given by the participant (could be 'left' or right');
  * the unique id of the participant;
  * the participant competence score (ranging from 0. to 1.; participants with score <0.7 were filtered out).
  
**Note:** *left* and *right* don’t correspond to the actual positions of sentences as seen by participants in the course of the study — the sentences were randomly flipped.
  
