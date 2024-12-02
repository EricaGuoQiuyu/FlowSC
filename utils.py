import networkx as nx
import scipy.io as sio
import torch
import numpy as np
import random
from copy import deepcopy
import subprocess
from scipy.sparse import csr_matrix
import collections

class Find_Candidates:
    def __init__(self, data_graph_file, query_graph_file):
        self.data_graph_file = data_graph_file
        self.query_graph_file = query_graph_file

        self.query_graph = self.load_graph(self.query_graph_file)
        self.query_size = len(self.query_graph[0])
 
    def load_graph(self, graph_file):
        nid = list()
        nlabel = list()
        nindeg = list()
        elabel = list()
        e_u = list()
        e_v = list()
        v_neigh = list()

        with open(graph_file) as f2:
            while True:
                line = f2.readline()
                if not line:
                    break

                if line.startswith('t'):
                    parts = line.strip().split()
                    num_nodes = int(parts[1])
                    num_edges = int(parts[2])

                    nid.clear()
                    nlabel.clear()
                    nindeg.clear()
                    elabel.clear()
                    e_u.clear()
                    e_v.clear()
                    v_neigh = [[] for _ in range(num_nodes)]

                    for _ in range(num_nodes):
                        node_info = f2.readline().strip().split()
                        node_id = int(node_info[1])
                        node_label = int(node_info[2])
                        node_degree = int(node_info[3])

                        nid.append(node_id)
                        nlabel.append(node_label)
                        nindeg.append(node_degree)

                    for _ in range(num_edges):
                        edge_info = f2.readline().strip().split()
                        u = int(edge_info[1])
                        v = int(edge_info[2])

                        e_u.append(u)
                        e_v.append(v)
                        elabel.append(1)  
                        v_neigh[u].append(v)

        g_nid = deepcopy(nid)
        g_nlabel = deepcopy(nlabel)
        g_indeg = deepcopy(nindeg)
        g_edges = [deepcopy(e_u), deepcopy(e_v)] 
        g_elabel = deepcopy(elabel)
        g_v_neigh = deepcopy(v_neigh)
        
        unique_labels = list(set(g_nlabel))
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels, start=1)}
        g_nlabel = [label_mapping[label] for label in g_nlabel]

        graph_info = [
            g_nid,
            g_nlabel,
            g_indeg,
            g_edges,
            g_elabel,
            g_v_neigh
        ]

        return graph_info

    def update_query(self, query_graph_file):
        self.query_graph_file = query_graph_file  
        self.query_graph = self.load_graph(self.query_graph_file)  
        self.query_size = len(self.query_graph[0])  

    def cpp_GQL(self):
        encoding = 'utf-8'
        base_command = ['filter/build1/matching/SubgraphMatching.out', '-d', self.data_graph_file, '-q', self.query_graph_file, '-filter', 'GQL']
        output = subprocess.run(base_command, capture_output=True)
        baseline_visit = output.stdout.decode(encoding).split('\n')

        for i in range(len(baseline_visit)):

            if 'Candidate Sizes' in baseline_visit[i]:
                sizes_line = baseline_visit[i]
                cs_size = []
                parts = sizes_line.split(": ")
                for part in parts[1].split():
                    cs_size.append(int(part))

                nodes = [] 
                edges_s = [] 
                edges_t = [] 

                nodes_line = baseline_visit[i+1]
                edges_line = baseline_visit[i+2]

                parts = nodes_line.split(": ")
                for part in parts[1].split(" | "):
                    query_candidates = []
                    for candidate in part.split():
                        query_candidates.append(int(candidate))
                    nodes.append(query_candidates)

                parts = edges_line.split(": ")
                for part in parts[1].split():
                    node_pair = part.split(",")
                    node_s = int(node_pair[0][1:])
                    node_t = int(node_pair[1][:-1])
                    edges_s.append(node_s)
                    edges_t.append(node_t)

                edges = [edges_s, edges_t]
            
            if 'num of candidate nodes' in baseline_visit[i]:
                parts = baseline_visit[i].split(": ")
                num_candidates = int(parts[1])

        return [self.query_graph[0], self.query_graph[1], self.query_graph[2], self.query_graph[3]], [cs_size, num_candidates, nodes, edges]

def generate_adjacency_matrix(edges, num_nodes):
    rows = edges[0] 
    cols = edges[1]
    data = np.ones(len(rows), dtype=float)
    
    adjacency_matrix = csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    adjacency_matrix.setdiag(1.0) #avoiding zeroing out features of vertices with no neighbors(it is not message-passing actually)

    return adjacency_matrix

def generate_query_adj(edges, num_nodes):
    rows = edges[0] + edges[1]  #bidirectional edges
    cols = edges[1] + edges[0]
    data = np.ones(len(rows), dtype=float)
    
    adjacency_matrix = csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    adjacency_matrix.setdiag(1.0)

    return adjacency_matrix   

def bfs_layers(edge_list, root): 
    layers = {}
    visited = set()
    queue = collections.deque([(root, 0)])
    node_to_layer = {}

    adj_list = {i: set() for i in range(max(max(edge_list[0]), max(edge_list[1]))+1)}
    for source, target in zip(edge_list[0], edge_list[1]):
        adj_list[source].add(target)
        adj_list[target].add(source)

    while queue:
        node, layer = queue.popleft()
        if node not in visited:
            visited.add(node)
            if layer not in layers:
                layers[layer] = []
            layers[layer].append(node)
            node_to_layer[node] = layer
            neighbors = adj_list[node]
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append((neighbor, layer + 1))

    return layers

def query_flow(query_adj, q_labels, layers, depth = 1, previous_nodes = None):

    if depth > len(layers):    
        return
    
    if previous_nodes is None:
        previous_nodes = layers[0]

    if depth < len(layers):
        current_nodes = layers[depth]

    else:
        current_nodes = None

    previous_labels = [q_labels[node] for node in previous_nodes]
    l_features = torch.tensor(previous_labels, dtype = torch.float)

    yield from query_flow(query_adj, q_labels, layers, depth+1, current_nodes)
    
    if current_nodes is not None:
        upper_lower_adj = query_adj[previous_nodes, :][:, np.concatenate((current_nodes, previous_nodes))]
        crow_indices = torch.tensor(upper_lower_adj.indptr, dtype = torch.int64)
        col_indices = torch.tensor(upper_lower_adj.indices, dtype = torch.int64)
        values = torch.tensor(upper_lower_adj.data, dtype = torch.float)
        adj_torch = torch.sparse_csr_tensor(crow_indices, col_indices, values, torch.Size(upper_lower_adj.shape))

        yield adj_torch, l_features

    else:
        yield None, l_features #leaf node layer

def data_flow(q_labels, node_cluster, layers, root_idx, adj_matrix, depth = 1, previous_nodes = None, previous_labels = None): #初始集合为某一个cluster内的所有candidates
    
    if depth > len(layers):
        return
    
    if previous_nodes is None:
        previous_nodes = node_cluster[root_idx]
        previous_labels = {node: q_labels[root_idx] for node in previous_nodes}
    
    if depth < len(layers):
        valid_cluster_idx = layers[depth]
        current_nodes = set()
        current_labels = {}

        for cluster_idx in valid_cluster_idx:
            for node in node_cluster[cluster_idx]:
                current_nodes.add(node)
                current_labels[node] = q_labels[cluster_idx]
        
        current_nodes = list(current_nodes)

    else:
        current_nodes = None
        current_labels = None

    previous_labels = [previous_labels[node] for node in previous_nodes]
    l_features = torch.tensor(previous_labels, dtype = torch.float)

    yield from data_flow(q_labels, node_cluster, layers, root_idx, adj_matrix, depth+1, current_nodes, current_labels)

    if current_nodes is not None:
        upper_lower_adj = adj_matrix[np.array(previous_nodes), :][:, np.concatenate((np.array(current_nodes), np.array(previous_nodes)))]
        crow_indices = torch.tensor(upper_lower_adj.indptr, dtype = torch.int64)
        col_indices = torch.tensor(upper_lower_adj.indices, dtype = torch.int64)
        values = torch.tensor(upper_lower_adj.data, dtype = torch.float)
        adj_torch = torch.sparse_csr_tensor(crow_indices, col_indices, values, torch.Size(upper_lower_adj.shape))

        yield adj_torch, l_features

    else:
        yield None, l_features #leaf node layer

    