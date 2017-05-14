import gensim

_WORD2VEC_MODEL = None

def _load_model():
    global _WORD2VEC_MODEL
    _WORD2VEC_MODEL = gensim.models.KeyedVectors.load_word2vec_format('./../models/word2vector/GoogleNews-vectors-negative300.bin', binary=True) 

def _unload_model():
    global _WORD2VEC_MODEL
    del _WORD2VEC_MODEL
    _WORD2VEC_MODEL = None

def word2vec(word):
    if _WORD2VEC_MODEL is None:
        _load_model()

    return _WORD2VEC_MODEL[word]
