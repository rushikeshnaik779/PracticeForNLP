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
