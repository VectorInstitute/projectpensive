import pandas as pd
import numpy as np
import torch
import time
from sentence_transformers import SentenceTransformer, util

if __name__ == "__main__":

    # Open data
    data = pd.read_csv('../civility/recommender/train-balanced-sarcasm-processed.csv')
    corpus = data['comment'].to_list()
    embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    start_time = time.time()
    # Generate Embeddings
    sarcasm_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    end_time = time.time()
    print("Time for computing embeddings:"+ str(end_time-start_time))
    
    # Save embeddings to pickle file
    torch.save(sarcasm_embeddings, 'sarcasm-embeddings-processed.pt')