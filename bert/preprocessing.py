# import libraries

# standard import 
import numpy  as np
import pandas as pd
import itertools
import random
import math  
import copy

# NLP
import re
import spacy
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
wnl = WordNetLemmatizer()

# time
from tqdm import tqdm 

# warnings
import warnings
warnings.filterwarnings('ignore') # ignore some warnings

# set seed
SEED = 517


# pre-process
# Before pre-processing the training set, we can also the duplicate rows (not necessary)

# set the emoji list
def extend_emojis(emojis):
    #Add spaces between emojis characters as both version are present in the data
    extended = [' '.join(emoji) for emoji in emojis]
    return [re.escape(emoji) for emoji in emojis + extended]


positive_emojis = extend_emojis([':)', ":')", ':d', ';)', '^_^',
                                   ';]', ':3', 'x3', 'xxx', 'xx', ':*', 'c:', ':o'])
negative_emojis = extend_emojis(
    ['<\\3', ':(', ":'(", '-_-', '._.', '- ___ -'])
neutral_emojis = extend_emojis([':/', ':|', ':||', ':l'])
lolface_emojis = extend_emojis([':p', 'xp', ';p'])

# replace emoji
def parse_emojis(tweet):
    tweet_parsed = tweet
    tweet_parsed = re.sub('|'.join(positive_emojis), '<smile>', tweet_parsed)
    tweet_parsed = re.sub('|'.join(negative_emojis), '<sadface>', tweet_parsed)
    tweet_parsed = re.sub('|'.join(neutral_emojis),
                          '<neutralface>', tweet_parsed)
    tweet_parsed = re.sub('|'.join(lolface_emojis), '<lolface>', tweet_parsed)
    return tweet_parsed

# def remove_punctuation(txt):
#     for c in string.punctuation:
#         txt = txt.replace(c,"")
#     return txt

def remove_digit(txt):
    x = re.sub(r'\b[-+]?[.\d]*[\d]+[:,.\d]*(st|nd|rd|th)?\b', '', txt)
    return x

# remove elong words
def reduce_elong(txt):
    return re.sub(r'\b(\S*?)(.)\2{2,}\b', r'\1\2', txt)

# prepare stopword list
def remove_stopwords(txt):
    txt = txt.lower().split()
    txt = [word for word in txt if word not in stopword]
    txt = ' '.join(txt)
    return txt
stopword = ['<url>', '<hashtag>', '#', '<user>']

# lemmatize
def lemmatize_txt(txt):
    lemmatize_txt = ' '.join(wnl.lemmatize(txt.split()[i]) for i in range(len(txt.split())))
    return lemmatize_txt

# combine all defined pre-processing functions
def preprocess(x):
    x = parse_emojis(x)
    x = remove_stopwords(x)
    x = remove_digit(x)
    x =re.sub(r'\((.*?)\)', r'\1', x)
    x = re.sub(r'\.\.\. <url>$', '', x)
    x = reduce_elong(x)
    x = lemmatize_txt(x)
    x = re.sub(r'\((.*?)\)', r'\1', x)
    return x