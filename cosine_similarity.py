from __future__ import division
import math
import pandas as pd 
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
from nltk.stem.porter import PorterStemmer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import operator
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity


people = pd.read_csv("/media/satyam/DiskG/ML codes/Coursera/Course1/people_wiki.csv")

obama = people[people.name == "Barack Obama"]
obamatext = obama['text'].iloc[0]

train_set = obama['text'].tolist()
list1 = (obamatext).split()
counts = Counter(list1)

"""Training set should be a series or a list"""
vectorizer = CountVectorizer(stop_words = 'english')
vectorizer.fit_transform(train_set)

test_set = people.text.tolist()
smatrix = vectorizer.transform(test_set)

smartix = smatrix.todense()

otfidf = TfidfTransformer(norm="l2")
otfidf.fit(smatrix)


otf_idf_matrix = otfidf.transform(smartix)
otf_idf_matrix = otf_idf_matrix.toarray()
n = obama.index

list_words = vectorizer.vocabulary_.keys()
list_words.sort()
list_freq = [0]*len(list_words)

for i in range(len(vectorizer.vocabulary_)):
    list_freq[i] = otf_idf_matrix[n, i]

dict_otfidf = dict(zip(list_words, list_freq))

otfidf_final = sorted(dict_otfidf.iteritems(), key=operator.itemgetter(1), reverse=True)

otfidf_df = pd.DataFrame(dict_otfidf.items(), columns=['Word', 'otf-IDF Value'])
otfidf_df = otfidf_df.sort_values(['otf-IDF Value'], ascending=False)



beckham = people[people.name == "David Beckham"]
beckhamtext = beckham['text'].iloc[0]

train_set = beckham['text'].tolist()
list1 = (beckhamtext).split()
counts = Counter(list1)

"""Training set should be a series or a list"""
vectorizer = CountVectorizer(stop_words = 'english')
vectorizer.fit_transform(train_set)

test_set = people.text.tolist()
smatrix = vectorizer.transform(test_set)

smartix = smatrix.todense()

btfidf = TfidfTransformer(norm="l2")
btfidf.fit(smatrix)


btf_idf_matrix = btfidf.transform(smartix)
btf_idf_matrix = btf_idf_matrix.toarray()
n = beckham.index

list_words = vectorizer.vocabulary_.keys()
list_words.sort()
list_freq = [0]*len(list_words)

for i in range(len(vectorizer.vocabulary_)):
    list_freq[i] = btf_idf_matrix[n, i]

dict_btfidf = dict(zip(list_words, list_freq))

btfidf_final = sorted(dict_btfidf.iteritems(), key=operator.itemgetter(1), reverse=True)

btfidf_df = pd.DataFrame(dict_btfidf.items(), columns=['Word', 'btf-IDF Value'])
btfidf_df = btfidf_df.sort_values(['btf-IDF Value'], ascending=False)

print (otfidf_df.head())
print (btfidf_df.head())

calc= pd.merge(otfidf_df, btfidf_df, on='Word')
print (calc.head())
print (len(calc))

obcosine_similarity = (1 - cosine(calc["btf-IDF Value"], calc["otf-IDF Value"]))

print ("The cosine similarity between Obama and Beckham is ", obcosine_similarity[0])


clinton = people[people.name == "Bill Clinton"]
clintontext = clinton['text'].iloc[0]

train_set = clinton['text'].tolist()
list1 = (clintontext).split()
counts = Counter(list1)

"""Training set should be a series or a list"""
vectorizer = CountVectorizer(stop_words = 'english')
vectorizer.fit_transform(train_set)

test_set = people.text.tolist()
smatrix = vectorizer.transform(test_set)

smartix = smatrix.todense()

ctfidf = TfidfTransformer(norm="l2")
ctfidf.fit(smatrix)


ctf_idf_matrix = ctfidf.transform(smartix)
ctf_idf_matrix = ctf_idf_matrix.toarray()
n = clinton.index

list_words = vectorizer.vocabulary_.keys()
list_words.sort()
list_freq = [0]*len(list_words)

for i in range(len(vectorizer.vocabulary_)):
    list_freq[i] = ctf_idf_matrix[n, i]

dict_ctfidf = dict(zip(list_words, list_freq))

ctfidf_final = sorted(dict_ctfidf.iteritems(), key=operator.itemgetter(1), reverse=True)

ctfidf_df = pd.DataFrame(dict_ctfidf.items(), columns=['Word', 'ctf-IDF Value'])
ctfidf_df = ctfidf_df.sort_values(['ctf-IDF Value'], ascending=False)

print (otfidf_df.head())
print (ctfidf_df.head())

calc= pd.merge(otfidf_df, ctfidf_df, on='Word')
print (calc.head())

occosine_similarity = (1 - cosine(calc["ctf-IDF Value"], calc["otf-IDF Value"]))

print ("The cosine similarity between Obama and Clinton is ", occosine_similarity[0])
