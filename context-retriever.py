import os

import argparse

import numpy as np

from load_file import load_file
from bm25 import run_bm25
from tfidf import run_tfidf
from bert import run_bert


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument("--query", type=str)
    parse.add_argument("--train_data_path", type=str)
    parse.add_argument("--dev_data_path", type=str)
    parse.add_argument("--model", type=str)
    
    args = parse.parse_args()
    
    if args.train_data_path == None:
        train_data_path = os.path.join(".", "data", "train.jsonl")
    else:
        train_data_path = args.train_data_path    


    if args.dev_data_path == None:
        dev_data_path = os.path.join(".", "data", "dev.jsonl")
    else:
        dev_data_path = args.dev_data_path
    query = args.query
    
    train = load_file(train_data_path)
    dev = load_file(dev_data_path)
    corpus = train["passage"] + dev["passage"]
    
    
    # preprocess the query
    
    char = ["train[", "dev[", "]"]
    index = {c:query.find(c) for c in char}
    if index["train["] != -1:
        i = query.find("[") + 1
        j = query.find("]")
        sample_index = int(query[i:j])
        query = train["question"][sample_index]
        queries = [query]
        true_passage = sample_index
    elif index["dev["] != -1:
        i = query.find("[") + 1
        j = query.find("]")
        sample_index = int(query[i:j])
        query = dev["question"][sample_index]
        queries = [query]
        true_passage = len(train["question"]) + sample_index
    elif query == "train":
        queries = train["question"]
        true_passage = np.arange(len(train["question"]))
    elif query == "dev":
        queries = dev["question"]
        true_passage = len(train["question"]) + np.arange(len(dev["question"]))
    else:
        queries = [query]
        true_passage = None
    
    
    # run the model
    
    if args.model == "bm25":
        run_bm25(queries, corpus, true_passage)
    
    elif args.model == "tfidf":
        run_tfidf(queries, corpus, true_passage)
    
    elif args.model == "bert":
        cls_hidden_state_path = "finetuned_cls_passage_hidden_state.pt"
        WEIGHTS_PATH = os.path.join(".", "weights", "miniBERT.pt")
        run_bert(queries, corpus, cls_hidden_state_path, true_passage, WEIGHTS_PATH)

if __name__ == "__main__":
    main()



