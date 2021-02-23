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

3. FORWARD AND BACKWARD PROP 
```python
def int_to_word_vecs(inds, parameters):
    """
    inds : numpy array. shape:(1, m)
    parameters : dict. weights to be trained
    """
    m = inds.shape[1]
    WRD_EMB = parameters['WRD_EMB']
    word_vec = WRD_EMB[inds.flatten(), :].T

    assert(word_vec.shape == (WRD_EMB.shape[1], m))

    return word_vec



def linear_dense(word_vec, parameters):
    """
    word_vec : numpy array. shape : (emb_size, m)
    parameters : dict. weights to be trained
    """

    m = word_vec.shape[1]
    W = parameters['W']
    Z = np.dot(W, word_vec)

    assert(Z.shape == (W.shape[0], m))

    return W, Z
    
def softmax(Z):
    """
    Z : output out of the dense layer. shape : (vocab_size, m)
    """
    softmax_out = np.divide(np.exp(Z), np.sum(np.exp(Z), axis=0, keepdims=True) + 0.001)

    assert(softmax_out.shape == Z.shape)

    return softmax_out
    
    
 def forward_propagation(inds, parameters):
    word_vec = int_to_word_vecs(inds, parameters)
    W, Z = linear_dense(word_vec, parameters)
    softmax_out = softmax(Z)

    caches = {}
    caches['inds'] = inds
    caches['word_vec'] = word_vec
    caches['W'] = W
    caches['Z'] = Z

    return softmax_out, caches


def softmax_backward(Y, softmax_out):
    """
    Y: labels of training data. shape:(vocab_size, m)
    softmax_out : output out of softmax. shape: (vocab_size, m)
    """

    dL_dZ = softmax_out - Y
    assert(dL_dZ.shape == softmax_out.shape)

    return dL_dZ
    
    
def dense_backward(dL_dZ, caches):
    """
    dL_dZ : shape : (vocab_size, m)
    caches: dict. results from each steps of forward propagation
    """
    W = caches['W']
    word_vec = caches['word_vec']
    m = word_vec.shape[1]

    dL_dW = (1/m) * np.dot(dL_dZ, word_vec.T)
    dL_dword_vec = np.dot(W.T, dL_dZ)

    assert( W.shape == dL_dW.shape)
    assert( word_vec.shape == dL_dword_vec.shape)

    return dL_dW, dL_dword_vec
    
def backward_propagation(Y, softmax_out, caches):
    dL_dZ = softmax_backward(Y, softmax_out)
    dL_dW, dL_dword_vec = dense_backward(dL_dZ, caches)
    
    gradients = dict()
    gradients['dL_dZ'] = dL_dZ
    gradients['dL_dW'] = dL_dW
    gradients['dL_dword_vec'] = dL_dword_vec
    
    return gradients

def update_parameters(parameters, caches, gradients, learning_rate):
    vocab_size, emb_size = parameters['WRD_EMB'].shape
    inds = caches['inds']
    dL_dword_vec = gradients['dL_dword_vec']
    m = inds.shape[-1]
    
    parameters['WRD_EMB'][inds.flatten(), :] -= dL_dword_vec.T * learning_rate

    parameters['W'] -= learning_rate * gradients['dL_dW']
    
    
    
```
