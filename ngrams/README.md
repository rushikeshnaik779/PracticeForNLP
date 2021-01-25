# "YOU SHALL KNOW THE NATURE OF A WORD BY THE COMPANY IT KEEPS". 
~John Rupert Firth


Creating N-grams with pure python 
```python
def generate_ngrams(sentence, n):

    # convert to lowercases 
    sentence = sentence.lower()

    # Replace all none alphanumeric characters with spaces 


    sentence = re.sub(r'[^a-zA-Z0-9]', ' ', sentence)

    # break sentence in tokens, remove empty tokens 
    tokens = [token for token in sentence.split(" ") if token != ""]

    # use the zip function to help us generate n-grams
    # Concatenate the tokens into ngrams and return 

    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]
```

Language Model
Types : 
- Statistical Language Model: 
 These models use traditional statistical techniques like N-grams, Hidden Markov Models (HMM) and certain linguistic rules to learn the probability distribution of words 
- Neural Language Models: These are new players in the NLP town and have surpassed the statistical langauge model in their effectiveness. They use different kinds of Neural Networks to model language


Steps : 
1) Create n-grams... I created Trigrams and creating model 
```python
from nltk import bigrams, trigrams

# count frequency of co-occurance 
for sentence in reuters.sents():
    for w1, w2, w3, in trigrams(sentence, pad_right=True, pad_left=True):
        model[(w1, w2)][w3] += 1
```

2) Create probability estimations 
```python
# let;s transform 

for w1_w2 in model:
    total_count = float(sum(model[w1_w2].values()))

    for w3 in model[w1_w2]:
        model[w1_w2][w3] /= total_count

```



3) To create random generated text with the same model 

```python
import random
def gen(text):
# starting words 

    sentence_finished = False
    while not sentence_finished:
        # select a random probability threshold 

        r = random.random()
        accumulator = .0

        for word in model[tuple(text[-2:])].keys():
            accumulator += model[tuple(text[-2:])][word]

            # select words that are above the probability threshold
            if accumulator >= r:
                text.append(word)
                break

        
        if text[-2:] == [None, None]:
            sentence_finished = True
        
    print(' '.join([t for t in text if t]))

```

4) Randomly Generated Text : 
### (OUTPUT)
![output image](https://github.com/rushikeshnaik779/PracticeForNLP/blob/main/ngrams/Screenshot%202021-01-25%20at%204.48.39%20PM.png)
