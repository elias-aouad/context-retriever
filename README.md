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
