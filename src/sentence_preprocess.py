from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from re import sub
from abbreviations import contractions

def _expand_abbreviations(words):
    return words

def _is_word(str):
    return any(char.isdigit() or char.isalpha() for char in str)

def _remove_punc(words):
    return [w for w in words if _is_word(w)]

def _get_wordnet_pos(word_tag):
    if word_tag.startswith('J'):
        return wordnet.ADJ
    elif word_tag.startswith('V'):
        return wordnet.VERB
    elif word_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess(sentence):
    sentence = sub('[.]{2,}', '.', sentence)
    words = word_tokenize(sentence)
    words = _remove_punc(words)
    words = _expand_abbreviations(words)

    tagged_words = pos_tag(words)
    del words

    wordnet_lemmatizer = WordNetLemmatizer()
    return [wordnet_lemmatizer.lemmatize(w, pos=_get_wordnet_pos(t)) for w, t in tagged_words]



