import re
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from metrics import metric

def run_bm25(queries, corpus, true_passage):
    token_pattern = re.compile(r"(?u)\b\w\w+\b")
    tokenize = token_pattern.findall
    
    tokenized_corpus = [tokenize(passage) for passage in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    
    if len(queries) == 1:
        query = queries[0] 
        print("\nQUERY :")
        print(query)
        tokenized_query = tokenize(query)
        doc_scores = bm25.get_scores(tokenized_query).tolist()
        predicted_doc = doc_scores.index(max(doc_scores))
        
        print("PREDICTED CONTEXT :")
        print(corpus[predicted_doc])
        
        if true_passage != None:
            prediction_order = sorted(list(range(len(doc_scores))), key=lambda i:doc_scores[i], reverse=True)
            index_of_true = prediction_order.index(true_passage)
            print("INDEX OF TRUE CONTEXT : {}".format(index_of_true))
    else:
        predicted_passage = []

        for query in tqdm(queries):
            tokenized_query = tokenize(query)
            doc_scores = bm25.get_scores(tokenized_query)
            prediction_order = sorted(list(range(len(doc_scores))), key=lambda i:doc_scores[i], reverse=True)
            predicted_passage += [prediction_order]
        
        print("Metric : {}".format(metric(true_passage, predicted_passage)))
