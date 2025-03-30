import yaml
import json
import os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoModelWithLMHead
from datasets import Dataset
from networkx.readwrite import json_graph
import networkx as nx

with open("settings.yml", "r") as f:
    settings = yaml.safe_load(f)


def write_embeddings(contx_folder):
    """write_embeddings

    This function writes embeddings for a list of texts to a json file. The embeddings are generated using a pre-trained transformer model and the Embedder class. The function takes as input the context folder, which contains the necessary files for embedding generation, and writes the embeddings to a file located in the same context folder.

    Args:
        contx_folder (str): The path to the context folder containing the necessary files for embedding generation.

    Returns:
        None"""
    full_dag_p = os.path.join(contx_folder, settings["graph_loc"])
    with open(full_dag_p, "r") as f:
        chunks_json = json.load(f)
    embedder = Embedder(settings["model_ckpt"])
    chunks_df = pd.DataFrame(chunks_json["nodes"])
    chunks_dataset = Dataset.from_pandas(chunks_df)
    embeddings_dataset = chunks_dataset.map(
        lambda x: {
            "embeddings": embedder.get_embeddings(x["explain"]).detach().numpy()[0]
        }
    )
    embdsdf = embeddings_dataset.to_pandas()
    emb_path = os.path.join(contx_folder, settings["embedding_location"])
    with open(emb_path, "a") as f:
        embdsdf.to_json(f, orient="records", lines=True)


class Embedder:

    def __init__(self, model_ckpt):
        """__init__

        Initializes an Embedder object with a pre-trained model checkpoint. The checkpoint is used to initialize both the tokenizer and the model. The tokenizer is responsible for encoding text input into a format that the model can understand, and the model is responsible for generating embeddings for the input.

        Args:
            model_ckpt (str): The path to the pre-trained model checkpoint. This can be a local file path or a URL pointing to a remote file.

        Returns:
            Embedder: An initialized Embedder object."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        self.model = AutoModel.from_pretrained(model_ckpt)

    def cls_pooling(self, model_output):
        """cls pooling operation

        [This function is used to perform class (CLS) pooling on the model output. In transformer models, the first token of each input sequence is often used as a class representation. This function returns the last hidden state of the first token, which can be used as a fixed-length representation of the input sequence for classification tasks.]

        Args:
            model_output (transformers.models.bert.modeling_bert.BertModelOutput): [The output of a transformer model, which contains the last hidden state of the model.]

        Returns:
            torch.Tensor: [The last hidden state of the first token, which can be used as a fixed-length representation of the input sequence for classification tasks.]
        """
        return model_output.last_hidden_state[:, 0]

    def get_embeddings(self, text_list):
        """get_embeddings

        This function is used to generate embeddings for a list of texts using a pre-trained transformer model.

        Args:
            text_list (List[str]): A list of strings to generate embeddings for.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, embedding_size) containing the embeddings for each text in the input list.
        """
        encoded_input = self.tokenizer(
            text_list, padding=True, truncation=True, return_tensors="pt"
        )
        encoded_input = {k: v for (k, v) in encoded_input.items()}
        model_output = self.model(**encoded_input)
        return self.cls_pooling(model_output)


def read_json_file(filename):
    """read_json_file

    This function reads a json file that contains a networkx graph structure and converts it back to a networkx graph object.

    Args:
        filename (str): The name of the json file to be read.

    Returns:
        nx.Graph: A networkx graph object representing the graph structure in the json file.
    """
    with open(filename) as f:
        js_graph = json.load(f)
    return json_graph.node_link_graph(js_graph)


def random_walk_with_custom_restart(
    G, initial_scores, restart_prob=1, max_iter=100, tol=1e-06
):
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
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    init_score_vec = initial_scores
    prob = init_score_vec.copy()
    prev_prob = np.zeros(n)
    adj_matrix = nx.to_numpy_array(G)
    adj_matrix += np.eye(n) * restart_prob
    degree_matrix = np.sum(adj_matrix, axis=1)
    transition_matrix = adj_matrix / degree_matrix[:, None]
    for i in range(max_iter):
        prob = transition_matrix @ prob
        if np.linalg.norm(prob - prev_prob, 1) < tol:
            break
        prev_prob = prob.copy()
    rank = {node: prob[idx] for (idx, node) in enumerate(nodes)}
    return rank


class QueryIndex:

    def __init__(self, embedding_location, graph_loc):
        """__init__

        Initializes a QueryIndex object by reading an embeddings dataframe from a json file located at `embedding_location`, and creating an embeddings numpy array and an embedder object. Additionally, reads a graph object from a json file located at `graph_loc` using the `read_json_file` function.

        Args:
            embedding_location (str): The path to the json file containing the embeddings dataframe. This can be a local file path or a URL pointing to a remote file.
            graph_loc (str): The path to the json file containing the networkx graph structure. This can be a local file path or a URL pointing to a remote file.

        Returns:
            QueryIndex: An initialized QueryIndex object."""
        usefile = embedding_location
        self.embeddings_df = pd.read_json(usefile, lines=True)
        self.emb_np = np.stack(self.embeddings_df["embeddings"])
        self.dag = read_json_file(graph_loc)
        self.embedder = Embedder(settings["model_ckpt"])

    def query_G(self, query):
        """query_G

        This function is used to retrieve a subset of the embeddings dataframe based on the relevance scores obtained from a personalized PageRank algorithm. It takes a query string as input and returns a dataframe containing the members, file_name, functionstr, explain, and rwr_scores for the top 'samples_return' number of embeddings.

        Args:
            query (str): A string to use as a query for the embeddings dataframe.

        Returns:
            pandas.DataFrame: A dataframe containing the top 'samples_return' number of embeddings based on their relevance scores to the input query.
        """
        query_embedding = self.embedder.get_embeddings([query]).detach().numpy()
        scores = np.matmul(query_embedding, self.emb_np.T).squeeze()
        rwr_scores = random_walk_with_custom_restart(self.dag, scores)
        self.embeddings_df["rwr_scores"] = rwr_scores
        select_df = self.embeddings_df.sort_values("rwr_scores").iloc[
            -settings["samples_return"] :
        ]
        return select_df[
            ["members", "file_name", "functionstr", "explain", "rwr_scores"]
        ]

    def query(self, query):
        """query

        This function queries a dataset using a given query embedding and returns the nearest examples along with their scores.

        Args:
            query (str): A string to generate an embedding for and query the dataset.

        Returns:
            pd.DataFrame: A dataframe containing the members, file name, function string, explanation, and scores of the nearest examples.
        """
        query_embedding = self.embedder.get_embeddings([query]).detach().numpy()
        embeddings_dataset = Dataset.from_pandas(self.embeddings_df)
        embeddings_dataset = embeddings_dataset.add_faiss_index(column="embeddings")
        (scores, samples) = embeddings_dataset.get_nearest_examples(
            "embeddings", query_embedding, k=settings["samples_return"]
        )
        samples_df = pd.DataFrame(samples)
        samples_df["scores"] = scores
        return samples_df[["members", "file_name", "functionstr", "explain", "scores"]]


def get_Qobj(context_path):
    """get_Qobj

    This function returns a QueryIndex object for a given context path. The context path is used to locate the embedding dataframe and graph structure files, which are read into a QueryIndex object.

    Args:
        context_path (str): The path to the directory containing the embedding dataframe and graph structure files. This can be a local file path or a URL pointing to a remote directory.

    Returns:
        QueryIndex: An initialized QueryIndex object containing the embedding dataframe, embeddings numpy array, embedder object, and graph object.
    """
    embedding_location = os.path.join(context_path, settings["embedding_location"])
    graph_loc = os.path.join(context_path, settings["graph_loc"])
    QObj = QueryIndex(embedding_location, graph_loc)
    return QObj


def retrieve_chunks(QObj, query):
    """retrieve_chunks

    This function retrieves a list of chunks from a QueryObject (QObj) based on a given query. It first queries the QObj using either the Graph or Faiss system, depending on the RAG_system setting in the configuration file. It then concatenates the explanation and function string for each chunk and returns the list of concatenated strings.

    Args:
        QObj (QueryObject): An object containing the embeddings dataframe and embedder.
        query (str): A string to generate an embedding for and query the dataset.

    Returns:
        list: A list of strings containing the members, file name, function string, explanation, and scores of the nearest examples or the top 'samples_return' number of embeddings based on their relevance scores to the input query.
    """
    if settings["RAG_system"] == "Graph":
        outdf = QObj.query_G(query)
    elif settings["RAG_system"] == "Faiss":
        outdf = QObj.query(query)
    else:
        raise ValueError(
            'RAG_system setting (settings.yml) must be one of "Graph", "Faiss"'
        )
    conc_explain = outdf.apply(
        lambda x: x["explain"] + "/n/n" + x["functionstr"], axis=1
    )
    return conc_explain.tolist()
