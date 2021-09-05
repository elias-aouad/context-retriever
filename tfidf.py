from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import trange

from metrics import metric


def run_tfidf(queries, corpus, true_passage):
    tfidf = TfidfVectorizer().fit(corpus)
    
    tfidf_passage = tfidf.transform(corpus)
    tfidf_question = tfidf.transform(queries)
    
    question_context_similarity = (tfidf_passage @ tfidf_question.T).toarray()
    
    if len(queries) == 1:
        query = queries[0]
        print("\nQUERY :")
        print(query)
        
        doc_scores = question_context_similarity[:, 0].tolist()
        predicted_doc = doc_scores.index(max(doc_scores))
        
        print("PREDICTED CONTEXT :")
        print(corpus[predicted_doc])
        
        if true_passage != None:
            prediction_order = sorted(list(range(len(doc_scores))), key=lambda i:doc_scores[i], reverse=True)
            index_of_true = prediction_order.index(true_passage)
            print("INDEX OF TRUE CONTEXT : {}".format(index_of_true))
    else:
        predicted_passage = []

        for i in trange(question_context_similarity.shape[1]):
            doc_scores = question_context_similarity[:, i]
            prediction_order = sorted(list(range(len(doc_scores))), key=lambda i:doc_scores[i], reverse=True)
            predicted_passage += [prediction_order]
        
        print("Metric : {}".format(metric(true_passage, predicted_passage)))

