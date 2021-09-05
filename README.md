# Context Retriever


First, install the required packages through this command line :

```python
pip install -r requirements.txt
```

To run the context retriever, you must first choose a query.

For that, you can either:
- choose a set (ex : train or dev)
- choose a sample from a set (ex : train[2] or dev[7] or ...)
- choose a new query (ex : football ...)

Three different models are implemented here :

- tf-idf:

```python
python context-retriever.py --model tfidf --query train[0]
```

- bm25:

```python
python context-retriever.py --model bm25 --query train[0]
```

- bert :

```python
python context-retriever.py --model bert --query train[0]
```


# Metric

For each document, the method will calculate a score which should evaluate the similarity between the query and the document.
Hence, we can sort the document according to their scores, the first one being the most similar, and the last one being the least similar.

Hence, in training mode, if we know which document is the real context of the query, then a good assessment to the prediction is to keep track of the index of the true context.

Hence, per prediction, the prediction metric will be evaluated as 0.9^index , which will be equal to :
- 1 if the model outputted the real context as the most similar document to the query (index=0)
- 0 if the model outputted the real context as the not very similar document to the query (index >> 1)

Once I compute this metrics for all queries, I simply take the average which gives me a metric for the performance on the set.

# Different models

- BM25
- TF-IDF: In this pipeline, I tried two approaches : raw text and preprocessed text. The preprocessed text refers to a removal of stop words and stemming of the words in the text.
- BERT : In this pipeline, I tried two approaches : First, I used original BERT, extracted all the CLS hidden states for each layer, and computed similarities using dot products between queries and passages. I deduced that layer 5 is the most suited for the task. Then, I finetuned the first 5 layers of BERT on the task, using the cosine similarity as a loss (using positive and negative pairs to avoid overfitting), and again computed similarities.

All the details of the models and how they were trained can be found in the notebooks, the command line method is just for testing the models on a specific query or set.


# Results

To sum up, here is a table for the different methods used in this work :

model name | metric on train set | metric on dev set
--- | --- | ---
BM25 | 0.37 | 0.39
TF-IDF (no preprocess) | 0.70 | 0.64
TF-IDF (with preprocess) | 0.69 | 0.65
BERT (5th layer) | 0.15 | 0.13
BERT finetuned | 0.34 | 0.18


