#!/usr/bin/env python
# coding: utf-8

# In[7]:


import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx


# In[8]:


# Input
input = 'A biography, or simply bio, is a detailed description of a persons life. It involves more than just the basic facts like education, work, relationships, and death; it portrays a persons experience of these life events. Unlike a profile or curriculum vitae (résumé), a biography presents a subjects life story, highlighting various aspects of his or her life, including intimate details of experience, and may include an analysis of the subjects personality. Biographical works are usually non-fiction, but fiction can also be used to portray a persons life. One in-depth form of biographical coverage is called legacy writing. Works in diverse media, from literature to film, form the genre known as biography. An authorized biography is written with the permission, cooperation, and at times, participation of a subject or a subjects heirs. An autobiography is written by the person himself or herself, sometimes with the assistance of a collaborator or ghostwriter.'


# In[14]:


def read_article(input):
    article = input.split(". ")
    sentences = []

    for sentence in article:
#         print(sentence)
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop()

    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []

    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]

    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1

    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


def generate_summary(file_name, top_n=2):
    nltk.download("stopwords")
    stop_words = stopwords.words('english')
    summarize_text = []
    sentences =  read_article(file_name)
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    for i in range(top_n):
      summarize_text.append(" ".join(ranked_sentence[i][1]))
    print("Summarized Text: \n", ". ".join(summarize_text))
    text_file = open("results.txt", "w")
    text_file.write(". ".join(summarize_text))
    text_file.close()


# In[15]:


st.title('Text Summarization Tool')


# In[16]:


st.write('by Sebastian Duerr')


# In[17]:


user_input = st.text_area("Please insert the text you want to summarize here.", 'A biography, or simply bio, is a detailed description of a persons life. It involves more than just the basic facts like education, work, relationships, and death; it portrays a persons experience of these life events. Unlike a profile or curriculum vitae (résumé), a biography presents a subjects life story, highlighting various aspects of his or her life, including intimate details of experience, and may include an analysis of the subjects personality. Biographical works are usually non-fiction, but fiction can also be used to portray a persons life. One in-depth form of biographical coverage is called legacy writing. Works in diverse media, from literature to film, form the genre known as biography. An authorized biography is written with the permission, cooperation, and at times, participation of a subject or a subjects heirs. An autobiography is written by the person himself or herself, sometimes with the assistance of a collaborator or ghostwriter.')

if st.button('Summarize'):
    generate_summary(user_input)
    text_file = open("results.txt", "r")
    st.write(text_file.read())
