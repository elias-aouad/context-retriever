import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from transformers import BertConfig, BertModel, BertTokenizer

from tqdm import tqdm

from metrics import metric


def resize_bert(bert, keep_layers):
    state_dict = dict(bert.state_dict())
    new_state_dict = {}
    
    for elem in state_dict:
        if "layer" in elem:
            elem_split = elem.split(".")
            layer_num = int(elem_split[elem_split.index("layer") + 1])
            
            if layer_num >= keep_layers:
                continue
        new_state_dict[elem] = state_dict[elem]
    
    config = bert.config; config.num_hidden_layers = keep_layers
    new_bert = BertModel(config)
    new_bert.load_state_dict(new_state_dict)
    
    return new_bert


def get_extended_attention_mask(attention_mask, input_shape, device=torch.device('cpu')):
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )
    extended_attention_mask = extended_attention_mask.float()  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


class SentenceEncoder(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.num_layers = self.bert.config.num_hidden_layers
        self.hidden_size = self.bert.config.hidden_size

    def forward(self, input_ids, attention_mask): 
        hidden_state = self.bert.embeddings(input_ids)
        ext_attention_mask = get_extended_attention_mask(attention_mask, hidden_state.shape).detach().data
        for i in range(self.bert.config.num_hidden_layers):
            hidden_state, = self.bert.encoder.layer[i](hidden_state, attention_mask=ext_attention_mask)
        return hidden_state[:, 0]


def run_bert(queries, corpus, corpus_cls_hidden_state_path, true_passage, WEIGHTS_PATH):
    
    # define device on which to perform computations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load pre-computed [CLS] hidden states for passages in our corpus
    cls_passage_hidden_state = torch.load("finetuned_cls_passage_hidden_state.pt", map_location="cpu")
    
    # define model for encoding text
    try:
        config = BertConfig(num_hidden_layers=5)
        bert = BertModel(config)
        model = SentenceEncoder(bert)
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
        model.to(device)
        model.eval()
    except:
        bert = BertModel.from_pretrained("bert-base-uncased")
        resized_bert = resize_bert(bert, 5)
        model = SentenceEncoder(resized_bert)
        model.to(device)
        model.eval()
        
    # define tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    if len(queries) == 1:
        query = queries[0]
        print("\nQUERY :")
        print(query)
        
        encoded_question = tokenizer(queries, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
        input_ids = encoded_question["input_ids"].to(device)
        attention_mask = encoded_question["attention_mask"].to(device)
        
        cls_question_hidden_state = model(input_ids, attention_mask)
        
        question_context_similarity = cls_passage_hidden_state @ cls_question_hidden_state.T
        
        doc_scores = question_context_similarity[:, 0].tolist()
        predicted_doc = doc_scores.index(max(doc_scores))
        
        print("PREDICTED CONTEXT :")
        print(corpus[predicted_doc])
        
        if true_passage != None:
            prediction_order = sorted(list(range(len(doc_scores))), key=lambda i:doc_scores[i], reverse=True)
            index_of_true = prediction_order.index(true_passage)
            print("INDEX OF TRUE CONTEXT : {}".format(index_of_true))
    else:
        encoded_question = tokenizer(queries, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
        
        # Create an iterator of our data with torch DataLoader 
        question_dataset = TensorDataset(encoded_question['input_ids'], encoded_question['attention_mask'])
        question_sampler = SequentialSampler(question_dataset)
        question_dataloader = DataLoader(question_dataset, sampler=question_sampler, batch_size=batch_size)
        
        
        for step, batch in tqdm(enumerate(question_dataloader)):

            batch = tuple(t.to(device) for t in batch)
        
            b_input_ids, b_input_mask = batch
        
            if step == 0:
                cls_question_hidden_state = model(b_input_ids, b_input_mask).detach().cpu()
            else:
                x = model(b_input_ids, b_input_mask).detach().cpu()
                cls_question_hidden_state = torch.cat([cls_question_hidden_state, x], axis=0)
        
        
        question_context_similarity = cls_passage_hidden_state @ cls_question_hidden_state.T
        
        predicted_passage = []
        for i in trange(question_context_similarity.shape[1]):
            doc_scores = question_context_similarity[:, i].tolist()
            prediction_order = sorted(list(range(len(doc_scores))), key=lambda i:doc_scores[i], reverse=True)
            predicted_passage += [prediction_order]
        
        print("Metric : {}".format(metric(true_passage, predicted_passage)))
        
