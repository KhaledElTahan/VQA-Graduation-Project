from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from re import sub
from abbreviations import expand
from word_preprocess import word2vec

def _is_word(str):
    return any(char.isdigit() or char.isalpha() for char in str)

def _remove_last_dot(str):
    if str[-1] == '.':
        return str[:-1]
    return str

def _remove_punc(words):
    return [_remove_last_dot(w) for w in words if _is_word(w)]

def _get_wordnet_pos(word_tag):
    if word_tag.startswith('J'):
        return wordnet.ADJ
    elif word_tag.startswith('V'):
        return wordnet.VERB
    elif word_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def tokenize(sentence):
    words = word_tokenize(sentence.lower())
    tokenized_words = []

    for i in range(len(words)):
        if i and words[i][0] == "'":
            del tokenized_words[-1]
            tokenized_words.append(words[i - 1] + words[i])
        else:
            tokenized_words.append(words[i])

    return tokenized_words

def preprocess(sentence):
    sentence = sub('[.]{2,}', '.', sentence)
    words = tokenize(sentence)
    words = _remove_punc(words)

    tagged_words = pos_tag(expand(pos_tag(words)))
    del words

    wordnet_lemmatizer = WordNetLemmatizer()
    return [wordnet_lemmatizer.lemmatize(w, pos=_get_wordnet_pos(t)) for w, t in tagged_words]

def sentence2vecs(sentence):
    return [word2vec(w) for w in preprocess(sentence)]

