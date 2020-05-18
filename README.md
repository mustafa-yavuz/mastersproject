### ML for Material Science
This project is an implementation of unsupervised and supervised learning methods on Material science publications.
Final goal of this project is to allow researchers to cluster similar papers, find papers with specific keywords, and classify their own papers.

# Data
* papers.csv : Csv file containing abstract and titles of papers.
* flair_embd_paper.pkl : Pickle file that holds a list which contains flair embeddings of abstracts for 10000 papers.

# Code
* text_extraction.py : Queries Arxiv api to extract abstract and titles of papers. Preprocesses the text and creates flair embeddings.
* build_embeddings_pdf.py : Parses pdfs in local directory.Preprocesses the whole text including title,abstract,body and summary. Creates flair embeddings.
* clustering_comparison.ipynb : Compares performances of several clustering algorithms. 
* pca_visualization.ipynb : Visualizes clustered labels on 2-D space.
* data_labeling.ipynb : Finds labels of the papers for given topics by utilizing cosine similarity.
* topic_modeling.ipynb : Implements non-negative matrix factorization method to find topics to the papers.
* classification.ipynb : Implements several classification algorithms by using labels from k-means algorithm.

