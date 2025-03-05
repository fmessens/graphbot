import yaml
import json
import os
import numpy as np

import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoModelWithLMHead
from datasets import Dataset
from networkx.readwrite import json_graph
import networkx as nx

with open('settings.yml', 'r') as f:
    settings = yaml.safe_load(f)

def write_embeddings(contx_folder):
    full_dag_p = os.path.join(contx_folder, settings['graph_loc'])
    with open(full_dag_p, 'r') as f:
        chunks_json = json.load(f)
    embedder = Embedder(settings['model_ckpt'])
    chunks_df = pd.DataFrame(chunks_json['nodes'])
    chunks_dataset = Dataset.from_pandas(chunks_df)
    embeddings_dataset = chunks_dataset.map(
            lambda x: {"embeddings": embedder.get_embeddings(x["explain"])
                    .detach()
                    .numpy()[0]})
    embdsdf = embeddings_dataset.to_pandas()
    
    emb_path = os.path.join(contx_folder, settings['embedding_location'])
    with open(emb_path, 'a') as f:
        embdsdf.to_json(f, orient='records', lines=True) # type: ignore



class Embedder():
    def __init__(self, model_ckpt):
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        self.model = AutoModel.from_pretrained(model_ckpt)

    def cls_pooling(self, model_output):
        return model_output.last_hidden_state[:, 0]

    def get_embeddings(self, text_list):
        encoded_input = self.tokenizer(
            text_list, padding=True, truncation=True, return_tensors="pt"
        )
        encoded_input = {k: v for k, v in encoded_input.items()}
        model_output = self.model(**encoded_input)
        return self.cls_pooling(model_output)
    

def read_json_file(filename):
    with open(filename) as f:
        js_graph = json.load(f)
    return json_graph.node_link_graph(js_graph)


def random_walk_with_custom_restart(G, initial_scores, restart_prob=1, max_iter=100, tol=1e-6):
    """
    Perform Random Walk with Restart (RWR) on a NetworkX graph with custom initial scores.

    Parameters:
    - G: networkx.Graph, the graph
    - initial_scores: dict, initial scores or probabilities for each node to be the restart node
    - restart_prob: float, probability of restarting from the initial score vector in each step
    - max_iter: int, maximum number of iterations to allow
    - tol: float, tolerance for convergence

    Returns:
    - rank: dict, node scores after RWR, representing the probability distribution
    """
    # Initialize the probability vectors
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    
    # Convert initial_scores dict to a vector
    init_score_vec = initial_scores
    #init_score_vec = init_score_vec / init_score_vec.sum()  # Normalize to sum to 1

    prob = init_score_vec.copy()  # Start with the initial score distribution
    prev_prob = np.zeros(n)       # To check for convergence

    # Precompute transition matrix
    adj_matrix = nx.to_numpy_array(G) 
    adj_matrix += np.eye(n) * restart_prob
    degree_matrix = np.sum(adj_matrix, axis=1)
    transition_matrix = adj_matrix / degree_matrix[:, None]

    # Iterate until convergence
    for i in range(max_iter):
        # Perform random walk with restart
        prob = transition_matrix @ prob

        # Check for convergence
        if np.linalg.norm(prob - prev_prob, 1) < tol:
            break
        prev_prob = prob.copy()

    # Convert final probabilities to a dictionary
    rank = {node: prob[idx] for idx, node in enumerate(nodes)}
    return rank

    

class QueryIndex:
    def __init__(self, embedding_location, graph_loc):
        usefile = embedding_location
        self.embeddings_df = pd.read_json(usefile, lines=True)
        self.emb_np = np.stack(self.embeddings_df['embeddings'])
        self.dag = read_json_file(graph_loc)
        self.embedder = Embedder(settings['model_ckpt'])

    def query_G(self, query):
        query_embedding = (self.embedder
                           .get_embeddings([query])
                           .detach().numpy())
        scores = (np.matmul(query_embedding,
                            self.emb_np.T)
                  .squeeze())
        rwr_scores = random_walk_with_custom_restart(self.dag, 
                                                     scores)
        self.embeddings_df['rwr_scores'] = rwr_scores
        select_df = self.embeddings_df.sort_values('rwr_scores').iloc[-settings['samples_return']:]
        return select_df[['members', 'file_name', 'functionstr',
                          'explain', 'rwr_scores']]
    
    def query(self, query):
        query_embedding = (self.embedder
                           .get_embeddings([query])
                           .detach().numpy())
        embeddings_dataset = Dataset.from_pandas(self.embeddings_df)
        embeddings_dataset = embeddings_dataset.add_faiss_index(column="embeddings")
        scores, samples = (embeddings_dataset
                           .get_nearest_examples(
                               "embeddings",
                               query_embedding,
                               k=settings['samples_return']
                           ))
        samples_df = pd.DataFrame(samples)
        samples_df['scores'] = scores
        return samples_df[['members', 'file_name', 'functionstr',
                          'explain', 'scores']]
    

def get_Qobj(context_path):
    embedding_location = os.path.join(context_path, settings['embedding_location'])
    graph_loc = os.path.join(context_path, settings['graph_loc'])
    QObj = QueryIndex(embedding_location, graph_loc)
    return QObj


def retrieve_chunks(QObj, query):
    if settings['RAG_system'] == 'Graph':
        outdf = QObj.query_G(query)
    elif settings['RAG_system'] == 'Faiss':
        outdf = QObj.query(query)
    else:
        raise ValueError('RAG_system setting (settings.yml) must be one of "Graph", "Faiss"')
    conc_explain = outdf.apply(lambda x: x['explain'] + '/n/n' + x['functionstr'], axis=1)
    return conc_explain.tolist()