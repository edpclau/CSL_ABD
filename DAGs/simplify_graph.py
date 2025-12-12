## Helper functions
import networkx as nx
import pandas as pd


def simplify_graph(adjacency_matrix: pd.DataFrame = None, graph: nx.DiGraph = None) -> pd.DataFrame:
    """
    Simplify a directed graph by pooling features with the same biomarker prefix.

    Parameters:
    graph (nx.DiGraph): The input directed graph to be simplified.
    adjacency_matrix (pd.DataFrame): The adjacency matrix of the graph.
    Returns:
    pd.DataFrame: A simplified adjacency matrix with pooled features.
    """
    # If a graph is provided, extract its adjacency matrix
    if graph is not None:
        adj = nx.to_pandas_adjacency(graph, dtype=int)
    else:
        adj = adjacency_matrix.copy()
    # Ensure the adjacency matrix is square
    assert adj.shape[0] == adj.shape[1], "Adjacency matrix must be square."
    # Ensure the adjacency matrix is labeled correctly
    assert all(adj.index == adj.columns), "Adjacency matrix must have matching row and column labels."
    
    # Pool features
    adj.columns = adj.columns.str.replace('_.+', '', regex=True)
    adj.index = adj.index.str.replace('_.+', '', regex=True)
    
    # Groupby col name
    adj = adj.groupby(adj.index).mean()
    adj = adj.T.groupby(adj.columns).mean().T

    # Convert to binary adjacency matrix
    return (adj > 0).astype(int)