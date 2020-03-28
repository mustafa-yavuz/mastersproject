"""
Reads pdf files of all papers and creates flair embeddings for all papers.
Dumps results to file corpus.pkl
"""
import numpy as np
import pandas as pd
import string
import os
import re
import pickle
from io import StringIO
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentPoolEmbeddings, Sentence


def embedding() :
    # initialize the word embeddings
    glove_embedding = WordEmbeddings('glove')
    flair_embedding_forward = FlairEmbeddings('news-forward-fast')
    flair_embedding_backward = FlairEmbeddings('news-backward-fast')

    # initialize the document embeddings, mode = mean
    document_embeddings = DocumentPoolEmbeddings([glove_embedding,
                                                flair_embedding_backward,
                                                flair_embedding_forward])
    
    return document_embeddings
    
def pdfparser(pdffile):

    with open(pdffile, mode='rb') as f:
        rsrcmgr = PDFResourceManager()
        retstr = StringIO()
        #codec = 'utf-8'
        laparams = LAParams()
        data =[]
        device = TextConverter(rsrcmgr, retstr, laparams=laparams)
        # Create a PDF interpreter object.
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        # Process each page contained in the document.
        for page in PDFPage.get_pages(f):
            interpreter.process_page(page)
            data = retstr.getvalue()
            #print(data)

        # Cleaning the data
        data = data.lower()
        data = re.sub('\[*?\]', ' ', data)
        data = re.sub('[%s]' % re.escape(string.punctuation), ' ', data)
        data = re.sub('\w*\d\w*', ' ', data)
        data = data.replace("\n", " ")

        return data

def create_corpus(pdf_folder):
    corpus = []
    document_embeddings = embedding()

    for file1 in os.listdir(pdf_folder):
        if file1.endswith(".pdf"):
            pdf=pdfparser(pdf_folder+file1)
            sentence = Sentence(pdf)
            document_embeddings.embed(sentence)
            corpus.append(sentence.get_embedding().detach().numpy())

    #Save corpus to a file
    with open('corpus.pkl', 'wb') as f:
        pickle.dump(corpus, f)

