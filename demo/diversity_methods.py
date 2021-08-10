import pandas as pd
import numpy as np
import torch

sarcasm_embeddings = torch.load("data/sarcasm_embeddings.pt")
df = pd.read_csv("data/sarcasm_dataframe.csv")

def get_embeddings(comments):
    selected = df['Comment'].isin(comments)
    embeddings = []
    for vec in df['vector']:
        if list(vec.cpu().numpy()) in selected['vector']:
            embeddings.append(vec)
    
    return embeddings


def greedy_selection(query, num_to_recommend):
    pass

def topic_diversification(query, num_to_recommend):
    pass

def get_similar_comments(query, num_to_recommend):
    pass

def compute_diversity():
    pass

def compare_diversity():
    pass
            
