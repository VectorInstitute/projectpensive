import pandas as pd
import numpy as np
import torch
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist
from sentence_transformers import util
from collections import OrderedDict


def get_similar_comments(embedder, dataset, corpus, sarcasm_embeddings, query, n):
    """
    Parameters
    query (string): the text of the post
    n (int): number of posts to recommend
    
    Returns df of top n similar comments
    """
#     embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
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
        recommend_frame.append({'comment':val[0],'similarity':val[1].cpu().numpy()})
     
    df = pd.DataFrame(recommend_frame)
    df_sim = df.copy()
    df_sim = df_sim = df_sim.set_index(['comment'])
    df = df.join(dataset.set_index('comment'), on='comment')
    return df, df_sim


def calculate_quality(c, R, df, df_sim):
    """
    *add
    """
    quality = 0
    rel_diversity = 0
    
    if len(R) == 0:
        rel_diversity = 1
        
    vector = np.array(df['vector'][df['comment'] == c].to_numpy()[0]).reshape(1, -1)
    diversity = []
    for item in R:
        diversity.append(
            1 - cosine_similarity(
                vector,
                np.array(df_sim['vector'][df_sim['comment'] == item].to_numpy()[0]).reshape(1, -1)
            )
        )
        
    rel_diversity = sum(diversity)/len(R) # relative diversity
    
    similarity = df['similarity'][df['comment'] == c].to_numpy()[0] # similarity
    
    quality = rel_diversity[0][0] * similarity # quality
    return quality


def greedy_selection(embedder, dataset, corpus, sarcasm_embeddings, query, num_to_recommend):
    """
    Parameters
    query (string): the text of the post
    n (int): number of posts to recommend
    
    Returns df with diverse comments
    """
    C_prime = get_similar_comments(embedder, dataset, corpus, sarcasm_embeddings, query, 500)[0]
    
    df_temp = C_prime.copy()
    recommendations = ['dummy']
    recommendations[0] = C_prime["comment"][0]  # first item is always the one with the highest similarity

    index = df_temp[(df_temp.comment == recommendations[0])].index

    df_temp = df_temp.drop(index)
    
    # set num_to_recommend = 50 to get top 50 recommendations
    for i in range(num_to_recommend):
        qualities = {}
        # Calculate the quality of each subreddit
        for item in df_temp['comment']:
            qualities[item] = calculate_quality(item, recommendations, df_temp, C_prime)

        highest_quality = max(qualities.values())
        highest_quality_subreddit = max(qualities, key= lambda x: qualities[x])
        recommendations.append(highest_quality_subreddit)

        index = df_temp[(df_temp.comment == recommendations[-1])].index
        df_temp = df_temp.drop(index)
        
    similarities = []
    for item in recommendations:
        sim = C_prime['similarity'][C_prime['comment'] == item].to_numpy()[0]
        similarities.append(sim)

    pairs = list(zip(recommendations, similarities))
    recommend_frame = []
    for val in pairs:
        recommend_frame.append({'comment':val[0],'similarity':val[1].item()})    

    df_sim = pd.DataFrame(recommend_frame)
    df = df_sim.copy()
    df = df.join(dataset.set_index('comment'), on='comment')
    df_sim = df_sim.set_index(['comment'])
    df = df.reset_index()
    df = df.drop(columns=['vector','index'])
    pd.set_option("display.max_colwidth", 300)
    return df, df_sim


def topic_diversification(embedder, dataset, corpus, sarcasm_embeddings, query, n):
    """
    Parameters
    query (string): the text of the post
    n (int): number of posts to recommend
    
    Returns df with diverse comments
    """
    N = 5 * n
    C_prime = get_similar_comments(embedder, dataset, corpus, sarcasm_embeddings, query, N)[0]
    
    # Prepare df for pariwise distance
    df_ils = C_prime.copy()
    df_ils = df_ils.set_index(['comment'])
    
    ils = {}
    # set ILS for first item
    ils[df_ils.head(1)['similarity'].index.values.item(0)] = df_ils.head(1)['similarity'].values[0].item()
    for i in range(2, N+1):
        top_n = df_ils.head(i - 1)
        top_n = top_n[['similarity']]
        bottom = df_ils.tail(len(df_ils) - i + 1)
        bottom = bottom[['similarity']]
        for item in bottom.index:
            row_data = bottom.loc[[item], :]
            top_n = top_n.append(row_data)
            ils[item] = sum([x for x in pdist(top_n)]) / len(top_n)  # ILS Calculation
            top_n = top_n.drop(index=item)

    dissimilarity_rank = {k: v for k, v in sorted(ils.items(), key=lambda item: item[1], reverse=True)}
    
    # a,b ∈ [0,1]
    a = 0.01
    b = 0.99
    new_rank = {}
    dissimilarity_rank = OrderedDict(dissimilarity_rank)
    for item in df_ils.index:
        P = C_prime.index[C_prime['comment'] == item]
        Pd = list(dissimilarity_rank.keys()).index(item)
        new_rank[item] = ((a * P) + (b * Pd))[0]
    
    final_ranks = {k: v for k, v in sorted(new_rank.items(), key=lambda item: item[1], reverse=False)}
    
    data = []
    for comment, score in final_ranks.items():
        data.append({'comment': comment,'rank': score})

    df_sim = pd.DataFrame(data)
    ils_rank = []
    for item in df_sim['comment']:
        ils_rank.append(dissimilarity_rank[item])

    df_sim['ILS Score'] = ils_rank
    df_sim = df_sim.sort_values(by=['rank'], ascending=False)
    df_sim = df_sim.head(n)
    df = df_sim.copy()
    df = df.join(dataset.set_index('comment'), on='comment')
    df_sim = df_sim.drop(columns=['rank'])
    df_sim = df_sim.set_index(['comment'])
    df = df.reset_index()
    df = df.drop(columns=['vector', 'index'])
    pd.set_option("display.max_colwidth", 300)
    return df, df_sim

def compute_diversity(df, n):
    dis_similarity = [x for x in pdist(df)]
    avg_dissim_greedy = (sum(dis_similarity))/((n/2)*(n-1))
    return avg_dissim_greedy


def compare_diversity(avg_dissim_algo, avg_dissim_control):
    percent_change = ((avg_dissim_algo - avg_dissim_control)/avg_dissim_control)*100
    return round(percent_change, 2)

# Subreddit Methods
def get_similar_subreddits(dfnew, target, num_subs_to_reccomend):
    similarities = []
    sub_name_vector = eval(dfnew['vector'][dfnew['labels'] == target].to_numpy()[0])
    sub_name_vector_reshaped = np.array(sub_name_vector).reshape(1, -1)
    for vector in dfnew['vector'].tolist():
        vector = eval(vector)
        vector_reshaped = np.array(vector).reshape(1, -1)
        similarities.append(cosine_similarity(sub_name_vector_reshaped, vector_reshaped))

    pairs = list(zip(dfnew['labels'], similarities, dfnew['vector']))
    closest_subs = sorted(pairs, key=lambda item: item[1], reverse=True)[1:num_subs_to_reccomend+1]
    recommend_frame = []
    for val in closest_subs:
        recommend_frame.append({'subreddit':val[0],'similarity':val[1].item(0), 'vector':val[2]})

    df = pd.DataFrame(recommend_frame)
    df_sim = df.copy()
    df_sim = df_sim.drop(columns=['vector'])
    df_sim = df_sim.set_index(['subreddit'])
    return df, df_sim

def calculate_subreddit_quality(c, R, df, df_sim):
    quality = 0
    rel_diversity = 0
    
    if len(R) == 0:
        rel_diversity = 1
    
    vector = eval(df['vector'][df['subreddit'] == c].to_numpy()[0])
    vector_reshaped = np.array(vector).reshape(1, -1)
    diversity = []
    for item in R:
        item_vector = eval(df_sim['vector'][df_sim['subreddit'] == item].to_numpy()[0])
        item_vector_reshaped = np.array(item_vector).reshape(1, -1)
        diversity.append(1 - cosine_similarity(vector_reshaped, item_vector_reshaped))
        
    rel_diversity = sum(diversity)/len(R) # relative diversity
    
    similarity = df['similarity'][df['subreddit'] == c].to_numpy()[0] # similarity
    
    quality = rel_diversity[0][0] * similarity # quality
    return quality

def subreddit_greedy_selection(dfnew, target, num_subs_to_reccomend):
    # Step 1: Select the best x = 500 cases according to their similarity to the target query. Set C'
    C_prime = get_similar_subreddits(dfnew, target, 200)[0]
    
    # Step 2: Add the most similar item from C' as the first item in the result set R and drop this item from C'
    df_temp = C_prime.copy()
    recommendations = ['dummy']
    recommendations[0] = C_prime["subreddit"][0]  # first item is always the one with the highest similarity
    index = df_temp[(df_temp.subreddit == recommendations[0])].index
    df_temp = df_temp.drop(index)
    
    # Step 3: During each subsequent iteration, the item selected is the one with the highest quality 
    # with respect to the set of cases selected during the previous iteration
    # set k = 50 to get top 50 recommendations
    for i in range(num_subs_to_reccomend):
        qualities = {}
        # Calculate the quality of each subreddit
        for item in df_temp['subreddit']:
            qualities[item] = calculate_subreddit_quality(item, recommendations, df_temp, C_prime)

        highest_quality = max(qualities.values())
        highest_quality_subreddit = max(qualities, key= lambda x: qualities[x])
        recommendations.append(highest_quality_subreddit)

        index = df_temp[(df_temp.subreddit == recommendations[-1])].index
        df_temp = df_temp.drop(index)

    # Evaluate the recommendations
    similarities = []
    for item in recommendations:
        sim = C_prime['similarity'][C_prime['subreddit'] == item].to_numpy()[0]
        similarities.append(sim)

    pairs = list(zip(recommendations, similarities))
    recommend_frame = []
    for val in pairs:
        recommend_frame.append({'subreddit':val[0],'similarity':val[1].item(0)})    
    df_sim = pd.DataFrame(recommend_frame)
    df = df_sim.copy()
    df = df.reset_index()
    df = df.drop(columns=['index'])
    df_sim = df_sim.set_index(['subreddit'])
    return df, df_sim
