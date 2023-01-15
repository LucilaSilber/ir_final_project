from flask import Flask, request, jsonify
#import pyspark
import sys
from collections import Counter, OrderedDict, defaultdict
#import itertools
#from itertools import islice, count, groupby
import pandas as pd
import numpy as np
import os
import re
import nltk
#from operator import itemgetter
from nltk.stem.porter import *
from nltk.corpus import stopwords
#from time import time
#from pathlib import Path
import pickle
from google.cloud import storage
import math
from numpy.linalg import norm
#import requests
#from scikit-learn.preprocessing import Binarizer
import sklearn
from sklearn import preprocessing
#from scipy import spatial
#import builtins
nltk.download('stopwords')
from inverted_index_gcp import *
#from pyspark.sql import *
#from pyspark.sql.functions import *
#from pyspark import SparkContext, SparkConf, SparkFiles
#from pyspark.sql import SQLContext

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# load indexes and pre-processed data from bucket
client = storage.Client()
# # Text_InvertedIndex
bucket = client.get_bucket('ir_project_lssl')
blob = bucket.get_blob(f'postings_gcp_text_wnorm/textindex.pkl')
pickle_in = blob.download_as_string()
textindex = pickle.loads(pickle_in)
# # Title_InvertedIndex
bucket = client.get_bucket('ir_project_lssl')
blob = bucket.get_blob(f'postings_gcp_title_wnorm/titleindex.pkl')
pickle_in = blob.download_as_string()
titleindex = pickle.loads(pickle_in)
# # AnchorText_InvertedIndex
bucket = client.get_bucket('ir_project_lssl')
blob = bucket.get_blob(f'postings_gcp_anchor_wnorm/anchortextindex.pkl')
pickle_in = blob.download_as_string()
anchortextindex = pickle.loads(pickle_in)
# # read page title
bucket = client.get_bucket('ir_project_lssl')
blob = bucket.get_blob(f'gcp_title_dict/title_dict.pkl')
pickle_in = blob.download_as_string()
pagetitles = pickle.loads(pickle_in)
indexes = [textindex, titleindex, anchortextindex]
prefixes = {textindex: 'postings_gcp_text_wnorm',titleindex: 'postings_gcp_title_wnorm',anchortextindex: 'postings_gcp_anchor_wnorm'}
# # read page rank
bucket = client.get_bucket('ir_project_lssl')
pagerank = pd.read_csv('gs://ir_project_lssl/pr/part-00000-0c9c0329-d315-483d-a1d0-70def0e3f0af-c000.csv',header=None,names=['id','rank'])
pagerank_dict = dict(zip(pagerank['id'],pagerank['rank']))
# # read page views
bucket = client.get_bucket('ir_project_lssl')
blob = bucket.get_blob(f'pageviews_gcp/pageviews-202108-user.pkl')
pickle_in = blob.download_as_string()
pageview = pickle.loads(pickle_in)
pageviews_df = pd.DataFrame(pageview.items(), columns=['id', 'views'])

# helper functions
def generate_query_tfidf_vector(query_to_search, index):
    """
    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']
    index:           inverted index loaded from the corresponding files.
    Returns:
    -----------
    vectorized query with tfidf scores
    """
    epsilon = .0000001
    total_vocab_size = len(query_to_search)
    Q = np.zeros((total_vocab_size))
    term_vector = list(index.posting_locs.keys())
    counter = Counter(query_to_search)
    for token in np.unique(query_to_search):
        if token in index.posting_locs.keys():  # avoid terms that do not appear in the index.
            tf = counter[token] / len(query_to_search)  # term frequency divided by the length of the query
            df = index.df[token]
            idf = math.log(len(index.DL) / (df + epsilon), 10)  # smoothing
            try:
                ind = query_to_search.index(token)
                Q[ind] = tf * idf
            except:
                pass
    return Q


def get_posting_iter(index, prefix, query):
    """
    This function returning the iterator working with posting list for each term from the query.

    Parameters:
    ----------
    index: inverted index
    prefix: bucket location of index files
    query: a list of terms, who's posting lists we retrieve
    """
    queryterms = {}
    for term in query:
        if term not in index.posting_locs.keys():
            continue
        else:
            queryterms[term] = index.posting_locs[term]
    words, pls = zip(*index.posting_lists_iter(prefix, queryterms))
    return words, pls


def get_candidate_documents_and_scores(query_to_search, index, words, pls):
    """
    Generate a dictionary representing a pool of up to 5000 candidate documents for a given query. This function will go through every token in query_to_search
    and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
    Then it will populate the dictionary 'candidates.'
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the document.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    words,pls: iterator for working with posting.

    Returns:
    -----------
    dictionary of candidates. In the following format:
                                                               key: pair (doc_id,term)
                                                               value: tfidf score.
    """
    candidates = {}
    for term in np.unique(query_to_search):
        if term in words:
            list_of_doc = pls[words.index(term)][:5000]
            normlized_tfidf = [(doc_id, (freq / index.DL[doc_id]) * math.log(len(index.DL) / index.df[term], 10)) for
                               doc_id, freq in list_of_doc]
            for doc_id, tfidf in normlized_tfidf:
                candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf
    return candidates


def generate_document_tfidf_matrix(query_to_search, index, words, pls):
    """
    Generate a DataFrame `D` of tfidf scores for a given query.
    Rows will be the documents candidates for a given query
    Columns will be the unique terms in the query.
    The value for a given document and term will be its tfidf score.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.


    words,pls: iterator for working with posting.

    Returns:
    -----------
    DataFrame of tfidf scores.
    """

    total_vocab_size = len(words)
    candidates_scores = get_candidate_documents_and_scores(query_to_search, index, words,pls)
    unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
    D = np.zeros((len(unique_candidates), total_vocab_size))
    D = pd.DataFrame(D)
    D.index = unique_candidates
    D.columns = words

    def get_id_term(row):
        return row.index.to_series().apply(lambda x: (x, row.name))

    D = D.apply(lambda row: get_id_term(row))
    D = D.applymap(lambda row: candidates_scores.get(row, 0))
    return D


def cosine_similarity(D, Q, index):
    """
    Calculate the cosine similarity for each candidate document in D and a given query (e.g., Q).
    Generate a dictionary of cosine similarity scores
    key: doc_id
    value: cosine similarity score

    Parameters:
    -----------
    D: DataFrame of tfidf scores.

    Q: vectorized query with tfidf scores

    Index

    Returns:
    -----------
    dictionary of cosine similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: cosine similarty score.
    """

    def cosine(A, B, index, doc_id):
        # (X * Y).sum(axis=1) / np.linalg.norm(X, axis=1) / np.linalg.norm(Y, axis=1)
        mine = np.dot(A, B) / index.DN[doc_id] / np.linalg.norm(B)
        return mine

    cos_sim = D.apply(lambda x: cosine(x, Q, index, x.name), axis=1)
    return dict(cos_sim)


def get_top_n(sim_dict, N=300):
    """
    Sort and return the highest N documents according to the cosine similarity score.
    Generate a dictionary of cosine similarity scores

    Parameters:
    -----------
    sim_dict: a dictionary of similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))

    N: Integer (how many documents to retrieve). By default N = 3

    Returns:
    -----------
    a ranked list of pairs (doc_id, score) in the length of N.
    """
    return sorted([(doc_id, score) for doc_id, score in sim_dict.items()], key=lambda x: x[1], reverse=True)[:N]


def merge_results(body_scores, title_scores, anchor_scores, body_w, title_w, anchor_w, rank_w, N=300):
    '''
    Merge the cosine similarity results for all 3 indexes using a weight vector (for each components).

    Parameters:
    -----------
    Cosine_Sim_dicts: body_scores, title_scores, anchor_scores
    Weight_Vector: body_w, title_w, anchor_w, rank_w
    N: Number of results to return

    Returns:
    -----------
    Sorted list of doc_id with the highest weighted score.
    '''
    weigthedbody = dict([(pair[0], body_w * pair[1]) for pair in body_scores])
    weigthedtitle = dict([(pair[0], title_w * pair[1]) for pair in title_scores])
    weigthedanchor = dict([(pair[0], anchor_w * pair[1]) for pair in anchor_scores])

    merged = weigthedbody
    for key in weigthedtitle.keys():
        merged[key] = merged.get(key, 0) + weigthedtitle[key]
    for key in weigthedanchor.keys():
        merged[key] = merged.get(key, 0) + weigthedanchor[key]
    sorted_merged = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:N]
    return [x[0] for x in sorted_merged]




# stopwords and regex
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    results = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(results)
    # BEGIN SOLUTION
    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    querylen = len(tokens)
    query = [word for word in tokens if word not in all_stopwords]
    for index in indexes:
        exists = [term for term in query if len(index.posting_locs[term]) != 0]
        if len(exists) < 1:
            continue
        words, pls = get_posting_iter(index, prefixes[index], query)
        proc_query = generate_query_tfidf_vector(query, index)
        D = generate_document_tfidf_matrix(query, index, words, pls)
        res = cosine_similarity(D, proc_query, index)
        results.append(get_top_n(res))
    if len(results) < 1:
        return (results)
    if querylen < 3:
        merged = merge_results(results[0], results[1], results[2], 0.25, 0.65, 0.10, 0, N=100)
    else:
        merged = merge_results(results[0], results[1], results[2], 0.25, 0.65, 0.10, 0, N=100)
    final = [(doc_id, pagetitles[doc_id]) for doc_id in merged]
    # END SOLUTION
    return jsonify(final)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    results = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(results)
    # BEGIN SOLUTION
    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    query = [word for word in tokens if word not in all_stopwords]
    words, pls = get_posting_iter(textindex, prefixes[textindex], query)
    proc_query = generate_query_tfidf_vector(query, textindex)
    D = generate_document_tfidf_matrix(query, textindex, words, pls)
    res = cosine_similarity(D, proc_query, textindex)
    results = get_top_n(res, N=100)
    final = [(doc_id[0], pagetitles[doc_id[0]]) for doc_id in results]
    # END SOLUTION
    return jsonify(final)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO 
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. For example, a document 
        with a title that matches two distinct query words will be ranked before a 
        document with a title that matches only one distinct query word, 
        regardless of the number of times the term appeared in the title (or 
        query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    results = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(results)
    # BEGIN SOLUTION
    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    query = [word for word in tokens if word not in all_stopwords]
    words, pls = get_posting_iter(titleindex, prefixes[titleindex], query)
    proc_query = generate_query_tfidf_vector(query, titleindex)
    proc_query = proc_query.reshape(-1, 1)
    Qtransformer = preprocessing.Binarizer().fit(proc_query)
    proc_query = Qtransformer.transform(proc_query)
    D = generate_document_tfidf_matrix(query, titleindex, words, pls)
    d_columns = D.columns
    d_index = D.index
    Dtransformer = preprocessing.Binarizer().fit(D)
    D = pd.DataFrame(Dtransformer.transform(D), columns=d_columns, index=d_index)
    res = cosine_similarity(D, proc_query, titleindex)
    results = get_top_n(res, N=len(titleindex.DL))
    final = [(doc_id[0], pagetitles[doc_id[0]]) for doc_id in results]
    # END SOLUTION
    return jsonify(final)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment 
        3 (GCP part) to do the tokenization and remove stopwords. For example, 
        a document with a anchor text that matches two distinct query words will 
        be ranked before a document with anchor text that matches only one 
        distinct query word, regardless of the number of times the term appeared 
        in the anchor text (or query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    results = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(results)
    # BEGIN SOLUTION
    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    query = [word for word in tokens if word not in all_stopwords]
    words, pls = get_posting_iter(anchortextindex, prefixes[anchortextindex], query)
    proc_query = generate_query_tfidf_vector(query, anchortextindex)
    proc_query = proc_query.reshape(-1, 1)
    Qtransformer = preprocessing.Binarizer().fit(proc_query)
    proc_query = Qtransformer.transform(proc_query)
    D = generate_document_tfidf_matrix(query, anchortextindex, words, pls)
    d_columns = D.columns
    d_index = D.index
    Dtransformer = preprocessing.Binarizer().fit(D)
    D = pd.DataFrame(Dtransformer.transform(D), columns=d_columns, index=d_index)
    res = cosine_similarity(D, proc_query, anchortextindex)
    results = get_top_n(res, N=len(anchortextindex.DL))
    final = [(doc_id[0], pagetitles[doc_id[0]]) for doc_id in results]
    # END SOLUTION
    return jsonify(final)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = list(pagerank[pagerank['id'].isin(wiki_ids)]['rank'].values)
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = list(pageviews_df[pageviews_df['id'].isin(wiki_ids)]['views'].values)
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)


## deprecated functions and experiments

# def bm25(index, D, query, b=0.75, k=1.5):
#     tf_D = D
#     doc_len = index.DL
#     df = index.df
#     N = len(doc_len)
#     avgdl = sum(doc_len) / len(doc_len)
#
#     def calc_idf(query):
#         """
#         This function calculate the idf values according to the BM25 idf formula for each term in the query.
#
#         Parameters:
#         -----------
#         query: list of token representing the query. For example: ['look', 'blue', 'sky']
#
#         Returns:
#         -----------
#         idf: dictionary of idf scores. As follows:
#                                                     key: term
#                                                     value: bm25 idf score
#         """
#         # YOUR CODE HERE
#         query_set = list(set(query))
#         idf = {}
#         for term in query_set:
#             n_t = df[term]
#             numer = N - n_t + 0.5
#             denom = n_t + 0.5
#             frac = (numer / denom) + 1
#             idf[term] = math.log(frac)
#         return idf
#
#     def score(query, row, idf):
#         """
#         This function calculate the bm25 score for given query and document.
#
#         Parameters:
#         -----------
#         query: list of token representing the query. For example: ['look', 'blue', 'sky']
#         doc_id: integer, document id.
#
#         Returns:
#         -----------
#         score: float, bm25 score.
#         """
#         # YOUR CODE HERE
#         # tf_dict = tf_D.loc[doc_id]
#         score = 0
#         docl = row['doclen']
#         for term in query:
#             value = row[query.index(term)]
#             numer = value * k
#             denom = value + (k * (1 - b + (b * (docl / avgdl))))
#             value = (numer / denom) * idf[term]
#             score += value
#         return score
#
#     results = {}
#     idf = calc_idf(query)
#     #     for docid in D.index:
#     #         results[docid] = score(query,docid,idf)
#     D['doclen'] = D.index.map(doc_len)
#     bm_sim = D.apply(lambda row: score(query, row, idf), axis=1)
#     return dict(bm_sim)

# def search_bm25():
#     ''' Returns up to a 100 of your best search results for the query. This is
#         th|e place to put forward your best search engine, and you are free to
#         implement the retrieval whoever you'd like within the bound of the
#         project requirements (efficiency, quality, etc.). That means it is up to
#         you to decide on whether to use stemming, remove stopwords, use
#         PageRank, query expansion, etc.
#
#         To issue a query navigate to a URL like:
#          http://YOUR_SERVER_DOMAIN/search?query=hello+world
#         where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
#         if you're using ngrok on Colab or your external IP on GCP.
#     Returns:
#     --------
#         list of up to 100 search results, ordered from best to worst where each
#         element is a tuple (wiki_id, title).
#     '''
#     results = []
#     query = 'LinkedIn'
#     if len(query) == 0:
#       return jsonify(results)
#     # BEGIN SOLUTION
#     tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
#     query =  [word for word in tokens if word not in all_stopwords]
#     for index in indexes:
#         words, pls = get_posting_iter(index, prefixes[index],query)
#         proc_query = generate_query_tfidf_vector(query, index)
#         D = generate_document_tfidf_matrix(query, index, words, pls)
#         res = bm25(index, D, query)
#         results.append(get_top_n(res))
#     merged = merge_results(results[0], results[1], results[2], 0.25, 0.4, 0.25, 0.1, N=100)
#     final = [(doc_id,pagetitles[doc_id]) for doc_id in merged]
#     # END SOLUTION
#     return (final)

# def query_exp(query):
#     results = []
#     for index in indexes:
#         words, pls = get_posting_iter(index, prefixes[index],query)
#         proc_query = generate_query_tfidf_vector(query, index)
#         D = generate_document_tfidf_matrix(query, index, words, pls)
#         res = cosine_similarity(D, proc_query)
#         results.append(get_top_n(res))
#     merged = merge_results(results[0], results[1], results[2], 0.25, 0.65, 0.10, 0.1, N=100)
#     texts = doc_text_DF.filter(doc_text_DF.id.isin(merged)).select(col("text"))
#     #texts = [doc_text_DF.filter(doc_text_DF.id == doc_id).collect[0][1] for doc_id in merged]
#     model = Word2Vec().setVectorSize(5).setSeed(42).fit(texts.rdd)
#     expend = []
#     for term in query:
#         expend.append(model.findSynonyms(term, 1))
#     query = query+expend
#     return query