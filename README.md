Project 1 for CS421 – University of Illinois at Chicago
Name 1: hshahi6@uic.edu
Name 2: auppal8@uic.edu

---------------------------------------------------------Setup------------------------------------------------------------

1) Download StanfordCoreNLP zip file from https://stanfordnlp.github.io/CoreNLP/download.html
2) Extract the zip file in ROOT/executable/resources
3) Navigate to ROOT/executable/resources/stanford-corenlp-full-2018-02-27 (The zip file you just extracted)
3) Run the StanfordCoreNLP parser or port 8080 using the following command:
   
   java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 8080 -timeout 15000
   
4) Open a new terminal window/tab
5) Navigate to ROOT/executable folder
6) Give the following shell command to produce results.txt in output folder (~2-3 seconds per essay)
   
   sh run.sh
   
   Alternatively, you may type in the following sequence of commands:
   
    pip3 install nltk
    pip3 install stanfordcorenlp
    pip3 install 

    python3 driver.py -f test


--------------------------------------------------------Technique--------------------------------------------------------

PART A
======
Part A was relatively simpler as we only had to extract the number of sentences in the corpus, however there were some isses with the standard punkt sentence tokenizer. It would not take into account '\n' and it would label "It is a cat.It is my cat." as one sentence due to inadequate spacing between the sentences. In order to overcome this issue while also considering the presence of abbreviations, we created rule based system which splits the sentence on period and see if the constituents have any idependent meaning. If they do then it would create multiple sentences or keep the same sentence otherwise. We find max and min for high and low and normalize the score between 1 and 5 (1 is low, 5 is high). We did not feel the need to count finite verbs as the score differentiates between both classes, if needed, finite verbs can be found through dependency parsing.


PART B
======
In Part B, we have used wordnet corpora to figure out whether the word is an english word or not and we ensure that tokens which are not english words or a part of wordnet corpora e.g. punctuations and stopwords are not considered in the overall score computation. We normalize the number of errors by valid words and scale it between 0 and 4, where 0 means no errors and 4 means every word is a spelling error.


PART C(i)
=========
In this part we perform dependency parsing to find all the subject verb pairs in the document. We extract these pairs and validate whether the pair agrees or not based on a probability score and a threshold. The probability score is computed using bigram probability of subject verb as observed in the high quality essays. So, for example in this case NNS VBZ would have low probability and it would not count towards agreement score. We normalize the agreement score by the total number of pairs and scale it between 1 to 5. Where 1 is low and 5 is high.

We also treat prepositions and wh pronouns as a special case, where we consider the actual tokens along with POS tag instead of just POS tags in bigram probability computations. This ensures that preposition specific verbs are handled properly. 


PART C(ii)
==========
In this part we first identify all the verbs in the essay and then we extract the context surrounding the verb. This gives us a list of intervals, we merge these intervals to get longer non-overlapping segments of (token, pos_tag) tuples. We compute the probabilities of bigrams for high quality essays and we train a set of words in the same way as we treat prepositions in Part C(i). This set of word contains words like (has, am, I, are etc). This ensures that cases like "I am going" have high probability than "I are go". We normalize the score and scale it between 1 to 5.


NOTE: For all of these scores, we can see the difference between low and high classes on average as reported in ROOT/output/results.txt included in the folder. With more data our probabilistic approach for part C(i) and C(ii) is expected to improve.

