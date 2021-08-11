import pandas as pd
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist
from sentence_transformers import SentenceTransformer, util
from collections import OrderedDict

embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

sarcasm_embeddings = torch.load("data/sarcasm_embeddings.pt", map_location=torch.device('cpu'))

def load_dataframe():
    dataset = pd.read_csv("../civility/recommender/train-balanced-sarcasm.csv")
    dataset = dataset.drop(["label", "score", "ups", "downs", "date", "created_utc"], 1)
    dataset = dataset[["comment", "parent_comment", "author", "subreddit"]]
    return dataset


dataset = load_dataframe()
corpus = dataset['comment'].to_list()

# Add vector embeddings as column in df
vectors = []
for vector in sarcasm_embeddings:
    vectors.append(list(vector.cpu().numpy()))

dataset['vector'] = vectors

def get_embeddings(comments):
    selected = df['Comment'].isin(comments)
    embeddings = []
    for vec in df['vector']:
        if list(vec.cpu().numpy()) in selected['vector']:
            embeddings.append(vec)
    
    return embeddings

def get_similar_comments(query, n):
    """
    Parameters
    query (string): the text of the post
    n (int): number of posts to recommend
    
    Returns df of top n similar comments
    """
    # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
    top_k = min(n, len(corpus))
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    similarities = []
    pairs = []

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.pytorch_cos_sim(query_embedding.cpu(), sarcasm_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    for score, idx in zip(top_results[0], top_results[1]):
        pairs.append(tuple((corpus[idx], score)))
    
    recommend_frame = []
    for val in pairs:
        recommend_frame.append({'Comment':val[0],'Similarity':val[1].numpy()})
     
    df = pd.DataFrame(recommend_frame)
    df_sim = df.copy()
    df_sim = df_sim = df_sim.set_index(['Comment'])
    df = df.join(dataset.set_index('comment'), on='Comment')
    return df, df_sim

def calculate_quality(c, R, df, df_sim):
    """
    *add
    """
    quality = 0
    rel_diversity = 0
    
    if len(R) == 0:
        rel_diversity = 1
        
    vector = np.array(df['vector'][df['Comment'] == c].to_numpy()[0]).reshape(1, -1)
    diversity = []
    for item in R:
        diversity.append(1 - cosine_similarity(vector, np.array(df_sim['vector'][df_sim['Comment'] == item].to_numpy()[0]).reshape(1, -1)))
        
    rel_diversity = sum(diversity)/len(R) # relative diversity
    
    similarity = df['Similarity'][df['Comment'] == c].to_numpy()[0] # similarity
    
    quality = rel_diversity[0][0] * similarity # quality
    return quality


def greedy_selection(query, num_to_recommend):
    """
    Parameters
    query (string): the text of the post
    n (int): number of posts to recommend
    
    Returns df with diverse comments
    """
    C_prime = get_similar_comments(query, 500)[0]
    
    df_temp = C_prime.copy()
    recommendations = ['dummy']
    recommendations[0] = C_prime["Comment"][0]  # first item is always the one with the highest similarity

    index = df_temp[(df_temp.Comment == recommendations[0])].index

    df_temp = df_temp.drop(index)
    
    # set num_to_recommend = 50 to get top 50 recommendations
    for i in range(num_to_recommend):
        qualities = {}
        # Calculate the quality of each subreddit
        for item in df_temp['Comment']:
            qualities[item] = calculate_quality(item, recommendations, df_temp, C_prime)

        highest_quality = max(qualities.values())
        highest_quality_subreddit = max(qualities, key= lambda x: qualities[x])
        recommendations.append(highest_quality_subreddit)

        index = df_temp[(df_temp.Comment == recommendations[-1])].index
        df_temp = df_temp.drop(index)
        
    similarities = []
    for item in recommendations:
        sim = C_prime['Similarity'][C_prime['Comment'] == item].to_numpy()[0]
        similarities.append(sim)

    pairs = list(zip(recommendations, similarities))
    recommend_frame = []
    for val in pairs:
        recommend_frame.append({'Comment':val[0],'Similarity':val[1].item()})    

    df_sim = pd.DataFrame(recommend_frame)
    df = df_sim.copy()
    df = df.join(dataset.set_index('comment'), on='Comment')
    df_sim = df_sim.set_index(['Comment'])
    df.reset_index()
    df_sim.reset_index()
    df = df.drop(columns=['vector'])
    pd.set_option("display.max_colwidth", 300)
    return df, df_sim

def topic_diversification(query, n):
    """
    Parameters
    query (string): the text of the post
    n (int): number of posts to recommend
    
    Returns df with diverse comments
    """
    N = 5 * n
    C_prime = get_similar_comments(query, N)[0]
    
    # Prepare df for pariwise distance
    df_ils = C_prime.copy()
    df_ils = df_ils.set_index(['Comment'])
    
    ils = {}
    # set ILS for first item
    ils[df_ils.head(1)['Similarity'].index.values.item(0)] = df_ils.head(1)['Similarity'].values[0].item()
    for i in range(2, N+1):
        top_n = df_ils.head(i - 1)
        top_n = top_n[['Similarity']]
        bottom = df_ils.tail(len(df_ils) - i + 1)
        bottom = bottom[['Similarity']]
        for item in bottom.index:
            rowData = bottom.loc[[item] , :]
            top_n = top_n.append(rowData)
            ils[item] = sum( [x for x in pdist(top_n)] ) / len(top_n) # ILS Calculation
            top_n= top_n.drop(index=item)
            
    
    dissimilarity_rank = {k: v for k, v in sorted(ils.items(), key=lambda item: item[1], reverse=True)}
    
    # a,b ∈ [0,1]
    a = 0.01
    b = 0.99
    new_rank = {}
    dissimilarity_rank = OrderedDict(dissimilarity_rank)
    for item in df_ils.index:
        P = C_prime.index[C_prime['Comment'] == item]
        Pd = list(dissimilarity_rank.keys()).index(item)
        new_rank[item] = ((a * P) + (b * Pd))[0]
    
    final_ranks = {k: v for k, v in sorted(new_rank.items(), key=lambda item: item[1], reverse=False)}
    
    data = []
    for comment, score in final_ranks.items():
        data.append({'Comment': comment,'Rank': score})

    df_sim = pd.DataFrame(data)
    ils_rank = []
    for item in df_sim['Comment']:
        ils_rank.append(dissimilarity_rank[item])

    df_sim['ILS Score'] = ils_rank
    df_sim = df_sim.sort_values(by=['Rank'], ascending=False)
    df_sim = df_sim.head(n)
    df = df_sim.copy()
    df = df.join(dataset.set_index('comment'), on='Comment')
    df_sim = df_sim.drop(columns=['Rank'])
    df_sim = df_sim.set_index(['Comment'])
    df.reset_index()
    df_sim.reset_index()
    df = df.drop(columns=['vector'])
    pd.set_option("display.max_colwidth", 300)
    return df, df_sim

def compute_diversity(df, n):
    dis_similarity = [x for x in pdist(df)]
    avg_dissim_greedy = (sum(dis_similarity))/((n/2)*(n-1))
    return avg_dissim_greedy

def compare_diversity(avg_dissim_algo, avg_dissim_control):
    percent_change = ((avg_dissim_algo - avg_dissim_control)/avg_dissim_control)*100
    return round(percent_change, 2)