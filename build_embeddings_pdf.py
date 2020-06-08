"""
This script is for when using local pdf papers.
Reads pdf files of all papers and dumps texts to file corpus.pkl
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

def pdfparser(pdffile):  
    with open(pdffile, mode='rb') as f:
        rsrcmgr = PDFResourceManager()
        retstr = StringIO()
        laparams = LAParams()
        data =[]
        device = TextConverter(rsrcmgr, retstr, laparams=laparams)

        # Create a PDF interpreter object.
        interpreter = PDFPageInterpreter(rsrcmgr, device)

        # Process each page contained in the document.
        for page in PDFPage.get_pages(f):
            interpreter.process_page(page)
            data = retstr.getvalue()

        # Cleaning the data 
        # text_cleaning in utils may be used (spacy added)
        
        data = data.lower()
        data = re.sub('\[*?\]', ' ', data)
        data = re.sub(f'[{re.escape(string.punctuation)}s]', ' ', data)
        data = re.sub('\w*\d\w*', ' ', data)
        data = data.replace("\n", " ")

        return data

def create_pdf_corpus(pdf_folder):

    for file1 in os.listdir(pdf_folder):
        if file1.endswith(".pdf"):
            pdf=pdfparser(pdf_folder+file1)

            #Save corpus to a pickle file
            with open(f"{file1}.txt", "w") as file:
                file.write(pdf)