"""
Queries arxiv API and parses the urls of papers.
Reads all papers under Material Science tag and fetch their abstracts and titles
Dumps results to files abstract.pkl and titles.pkl
"""
import urllib.request
import feedparser
import pickle

# Base api query url
base_url = 'http://export.arxiv.org/api/query?';

# Search parameters- Limits the number of papers accessed
search_query = 'cat:cond-mat.mtrl-sci' 
start = 0                     
max_results = 10000

query = 'search_query=%s&start=%i&max_results=%i' % (search_query,
                                                     start,
                                                     max_results)


feedparser._FeedParserMixin.namespaces['http://arxiv.org/schemas/atom'] = 'arxiv'

# perform a GET request using the base_url and query
response = urllib.request.urlopen(base_url+query).read()

# parse the response using feedparser
feed = feedparser.parse(response)

# print out feed information
print ('Feed title: %s' % feed.feed.title)


# print opensearch metadata
print ('totalResults for this query: %s' % feed.feed.opensearch_totalresults)
print ('itemsPerPage for this query: %s' % feed.feed.opensearch_itemsperpage)
print ('startIndex for this query: %s'   % feed.feed.opensearch_startindex)

# Run through each entry, and print out information
title_list = []
abstract_list = []
for entry in feed.entries:
   
    title_list.append(entry.title)
    print ('Title:  %s' % entry.title)

    # get the links to the abs page and pdf 
    for link in entry.links:
        if link.rel == 'alternate':
            print ('abs page link: %s' % link.href)
        elif link.title == 'pdf':
            print ('pdf link: %s' % link.href)
    

    #print ('Abstract: %s' %  entry.summary)
    abstract_list.append(entry.summary)

# Dump abstract and title lists to pickle files
with open('abstract_10000.pkl', 'wb') as f:
    pickle.dump(abstract_list, f)
with open('title_10000.pkl', 'wb') as f:
    pickle.dump(title_list, f)
