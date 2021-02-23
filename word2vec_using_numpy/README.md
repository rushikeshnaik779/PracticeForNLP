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

2. Initialization 
- word Embedding
- Dense Layer
- Parameters


```python

def initialize_wrd_emb(vocab_size, emb_size):
    WRD_EMB = np.random.randn(vocab_size, emb_size)*0.01
    assert(WRD_EMB.shape == (vocab_size, emb_size))
    return WRD_EMB
    
    
def initialize_dense(input_size, output_size):
    W = np.random.randn(output_size, input_size) * 0.01

    assert(W.shape == (output_size, input_size))

    return W
    
def initialize_parameters(vocab_size, emb_size):
    WRD_EMB  = initialize_wrd_emb(vocab_size, emb_size)
    W = initialize_dense(emb_size, vocab_size)

    parameters = {}
    parameters['WRD_EMB'] = WRD_EMB 
    parameters['W'] = W
    return parameters
    
 print(initialize_parameters(5, 5))
"""
OUTPUT:
{'WRD_EMB': array([[ 0.00797424, -0.00102398, -0.01053327,  0.01144288,  0.01087524],
       [-0.0014608 ,  0.00014197, -0.00152098, -0.00103424, -0.01377535],
       [ 0.01346266, -0.00090859,  0.00369364,  0.01376817, -0.0156851 ],
       [ 0.01112075,  0.01053788, -0.00350727, -0.00291178, -0.01632778],
       [ 0.00185193,  0.00080155,  0.00548211,  0.01214297,  0.00353309]]), 'W': array([[ 0.0163293 , -0.00488816, -0.00443603, -0.01363806, -0.01073577],
       [-0.01489897, -0.01363439, -0.01334275,  0.01253328, -0.00738896],
       [ 0.00059801, -0.00407217, -0.01031124,  0.00515161,  0.00789267],
       [ 0.00186501, -0.01324542,  0.0141502 ,  0.00133391, -0.01127588],
       [-0.00423727, -0.01283284,  0.00081796, -0.01422961,  0.00207391]])}
"""
```
