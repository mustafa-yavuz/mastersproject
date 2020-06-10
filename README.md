<img src='https://i.imgur.com/TqHwhXT.jpg' title='Poster' width='' />

# ML for Material Science
This project is an implementation of unsupervised and supervised learning methods on Material science publications.
Final goal of this project is to allow researchers to cluster similar papers, find papers with specific keywords, and classify their own papers.

### Data
* Link for downloading pdfs from Arxiv : [arxiv-API](https://arxiv.org/help/api/user-manual)
* papers.csv : Csv file containing abstract and titles of papers.
* [flair_embd_paper.pkl](https://drive.google.com/open?id=1H_L5ZwIZrrxbJ5O24IsooymUUv3lmuVW) : Pickle file that holds a list which contains flair embeddings(in gpu tensor format) of abstracts for 10000 papers. This list can be used as the documents embeddings if you don't want to wait 2-3 hours to get your own embeddings.
* abstract_list.pkl : Pickle file that holds a list which contains abstracts for 10000 papers.
* title_list.pkl : Pickle file that holds a list which contains titles for 10000 papers.

### Code
* Link for document embeddings : [flairNLP](https://github.com/flairNLP/flair) and [fasttext](https://fasttext.cc/docs/en/english-vectors.html)
* text_extraction.py : Queries Arxiv api to extract abstract and titles of papers. Preprocesses the text and store them in pickle files.
* build_embeddings_pdf.py : Parses pdfs in local directory.Preprocesses the whole text including title,abstract,body and summary.Creates corresponding text files.
* clustering_comparison.ipynb : Compares performances of several clustering algorithms. 
* pca_visualization.ipynb : Visualizes clustered labels on 2-D space.
* data_labeling.ipynb : Finds labels of the papers for given topics by utilizing cosine similarity.
* topic_modeling.ipynb : Implements non-negative matrix factorization method to find topics to the papers.
* classification.ipynb : Implements several classification algorithms by using labels from k-means algorithm.

