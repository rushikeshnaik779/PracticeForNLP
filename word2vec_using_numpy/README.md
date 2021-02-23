# Word2vec implementation using numpy 
## Objective : 
- to show the inner workings of Word2Vec in python using numpy

## Definition
- Word2vec is a technique for natural language processing. The word2vec algorithm uses a neural network model to learn word associations from a large corpus of text. Once trained, such a model can detect synonymous words or suggest additional words for a partial sentence. As the name implies, word2vec represents each distinct word with a particular list of numbers called a vector. The vectors are chosen carefully such that a simple mathematical function (the cosine similarity between the vectors) indicates the level of semantic similarity between the words represented by those vectors.

## Following are the steps 
1. DATA PREPARATION : 
```python
def tokenize(text):
    # obtains tokens with atlest 1 alphabet 
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())
    
tokenize('I love goa')
# output : ['i', 'love', 'goa']
```
Data Mapping:
```python
def mapping(tokens):
    word_to_id = dict()
    id_to_word = dict()
    
    for i, token in enumerate(set(tokens)):
        word_to_id[token] = i
        id_to_word[i] = token

    
    return word_to_id, id_to_word
    
    
 mapping(['i', 'love', 'goa'])
 
 # output : ({'goa': 1, 'i': 0, 'love': 2}, {0: 'i', 1: 'goa', 2: 'love'})
```
