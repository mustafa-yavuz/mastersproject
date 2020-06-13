import numpy as np
import pandas as pd
import pickle
import torch
import re
import string

from tqdm.auto import tqdm, trange
# from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentPoolEmbeddings
# from flair.data import Sentence
from sklearn.feature_extraction.text import TfidfVectorizer


# def embedding() :
#     # initialize the word embeddings
#     glove_embedding = WordEmbeddings('glove')
#     flair_embedding_forward = FlairEmbeddings('news-forward-fast')
#     flair_embedding_backward = FlairEmbeddings('news-backward-fast')

#     # initialize the document embeddings, mode = mean
#     document_embeddings = DocumentPoolEmbeddings([glove_embedding,
#                                                 flair_embedding_backward,
#                                                 flair_embedding_forward])
    
#     return document_embeddings

# e4 = embedding()

def text_cleaning(data,spacy_model):

    data = data.lower()
    data = re.sub('\$(.*?)\$',' ',data)
    data = re.sub('\[*?\]', ' ', data)
    data = re.sub(f'[{re.escape(string.punctuation)}]', ' ', data)
    data = re.sub('\w*\d\w*', ' ', data)
    data = data.replace("\n"," ")
    data = spacy_model(" ".join(data.split()))
    
    # lemmatization
    data = ' '.join([token.lemma_ for token in data])

    # removing PRON after lemmatization
    data = data.replace("-PRON-","")
    
    data = re.sub('[^a-zA-Z0-9 -]','',data)
    data = re.sub(r"\b[a-zA-Z]\b", "", data)
    data = re.sub(r" mm ", " ", data)

    return " ".join(data.split())

def find_top_n(text,vectorizer,n):

    response = vectorizer.transform([text])

    feature_array = np.array(vectorizer.get_feature_names())
    tfidf_sorting = np.argsort(response.toarray()).flatten()[::-1]

    return feature_array[tfidf_sorting][:n]

# e4 = embedding()

# def get_embedding(text):
    
#     sentence = Sentence(text)
#     e4.embed(sentence)
    
#     return sentence.embedding

# def tfidf_sentences(corpus,vectorizer,n):
    
#     tfidf = []
    
#     # Concatenating tf-idf words into sentences
#     for text in tqdm(corpus):
#         tfidf.append(" ".join(find_top_n(text,vectorizer,n)))
    
#     tfidf_sentences = [get_embedding(text) for text in tqdm(tfidf)]
    
#     return tfidf_sentences
def pro_labeling(doc,ft_model,cosine,pro_1,pro_2,pro_3,pro_4,pro_5):
    
    production = []
    production_cos_score = []
    
    for text in doc:
        
        text = ft_model.get_sentence_vector(text)
        
        lbl= []

        lbl.append(1-cosine(text,pro_1))
        lbl.append(1-cosine(text,pro_2))
        lbl.append(1-cosine(text,pro_3))
        lbl.append(1-cosine(text,pro_4))
        lbl.append(1-cosine(text,pro_5))

        if lbl.index(max(lbl))==0 or lbl.index(max(lbl))==1:
            production.append('SLM or DMLS')
            production_cos_score.append(max(lbl))
        else:
            production.append('FDM or FFF or EAM')
            production_cos_score.append(max(lbl))
    
    return production , production_cos_score

# def production_labeling(sentences,cos,label_1,label_2,label_3,label_4,label_5):
    
#     production = []
#     production_cos_score = []
    
#     for text in tqdm(sentences):

#         lbl= []

#         lbl.append(cos(text,label_1))
#         lbl.append(cos(text,label_2))
#         lbl.append(cos(text,label_3))
#         lbl.append(cos(text,label_4))
#         lbl.append(cos(text,label_5))

#         if lbl.index(max(lbl))==0 or lbl.index(max(lbl))==1:
#             production.append('SLM or DMLS')
#             production_cos_score.append(max(lbl))
#         else:
#             production.append('FDM or FFF or EAM')
#             production_cos_score.append(max(lbl))
    
#     # Finding cosine scores
#     production_cos_score = list(map(float,production_cos_score))
#     production_cos_score = list(map(lambda x: round(x,2), production_cos_score)) 
    
#     return production , production_cos_score
def mat_labeling(doc,ft_model,cosine,metal,ceramic,polymer):

    material = []  
    material_cos_score = []

    for text in doc:
        
        text = ft_model.get_sentence_vector(text)
        
        lbl= []

        lbl.append(1-cosine(text,metal))
        lbl.append(1-cosine(text,ceramic))
        lbl.append(1-cosine(text,polymer))

        if lbl.index(max(lbl))==0:
            material.append('Metal')
            material_cos_score.append(max(lbl))
        elif lbl.index(max(lbl))==1:
            material.append('Ceramic')
            material_cos_score.append(max(lbl))
        else:
            material.append('Polymer')
            material_cos_score.append(max(lbl))
    
    return material , material_cos_score

# def material_labeling(sentences,cos,metal,ceramic,polymer):

#     material = []  
#     material_cos_score = []

#     for text in tqdm(sentences):

#         lbl= []

#         lbl.append(cos(text,metal))
#         lbl.append(cos(text,ceramic))
#         lbl.append(cos(text,polymer))

#         if lbl.index(max(lbl))==0:
#             material.append('Metal')
#             material_cos_score.append(max(lbl))
#         elif lbl.index(max(lbl))==1:
#             material.append('Ceramic')
#             material_cos_score.append(max(lbl))
#         else:
#             material.append('Polymer')
#             material_cos_score.append(max(lbl))
            
#     # Finding cosine scores
#     material_cos_score = list(map(float,material_cos_score))
#     material_cos_score = list(map(lambda x: round(x,2), material_cos_score)) 
    
#     return material , material_cos_score

def feature_labeling(sentences,cos,feature_1,feature_2,feature_3,feature_4,feature_5,feature_6):

    feature = []
    feature_cos_score = []
    
    for text in tqdm(sentences):

        lbl= []

        lbl.append(cos(text,feature_1))
        lbl.append(cos(text,feature_2))
        lbl.append(cos(text,feature_3))
        lbl.append(cos(text,feature_4))
        lbl.append(cos(text,feature_5))
        lbl.append(cos(text,feature_6))

        if lbl.index(max(lbl))==0:
            feature.append('Fracture toughness or Work of fracture')
            feature_cos_score.append(max(lbl))
        elif lbl.index(max(lbl))==1:
            feature.append('Tensile strength or ultimate tensile strength')
            feature_cos_score.append(max(lbl))
        elif lbl.index(max(lbl))==2:
            feature.append('Yield strength')
            feature_cos_score.append(max(lbl))
        elif lbl.index(max(lbl))==3:
            feature.append('Elastic modulus or Young’s modulus')
            feature_cos_score.append(max(lbl))
        elif lbl.index(max(lbl))==4:
            feature.append('Strain at break or strain at fracture or fracture strain')
            feature_cos_score.append(max(lbl))
        else:
            feature.append('Weibull modulus')
            feature_cos_score.append(max(lbl))

    # Finding cosine scores
    feature_cos_score = list(map(float,feature_cos_score))
    feature_cos_score = list(map(lambda x: round(x,2), feature_cos_score)) 
    
    return feature , feature_cos_score

