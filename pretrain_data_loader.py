import numpy as np
import random
from torch.utils.data import Dataset
import networkx as nx
import os
from utils import Find_Candidates

class Graph_Pairs(Dataset):
    def __init__(self, dataset_name_list, root_path, groundtruth_path, batch_query_num):
        super(Graph_Pairs, self).__init__()

        self.dataset_name_list = dataset_name_list
        self.root_path = root_path
        self.groundtruth_path = groundtruth_path
        self.batch_query_num = batch_query_num

        self.query_data_pairs = [] 
        self.groundtruth_dict = {}
        for dataset_name in dataset_name_list:
            with open(os.path.join(groundtruth_path, f"{dataset_name}.txt"), 'r') as file:
                for line in file:
                    log_file, embedding_value = line.strip().split(': ')
                    self.groundtruth_dict[log_file] = float(embedding_value)


            data_graph_path = os.path.join(root_path, dataset_name, 'data_graph', f"{dataset_name}.graph")
            query_graph_dir = os.path.join(root_path, dataset_name, 'query_graph')

            query_graph_files = os.listdir(query_graph_dir)
            for query_file in query_graph_files:
                query_path = os.path.join(query_graph_dir, query_file)
                self.query_data_pairs.append((data_graph_path, query_path))
        
        random.shuffle(self.query_data_pairs)

    def __len__(self):
        return len(self.query_data_pairs) // self.batch_query_num

    def __getitem__(self, index):

        start_idx = index * self.batch_query_num
        end_idx = start_idx + self.batch_query_num
        batch_pairs = self.query_data_pairs[start_idx:end_idx]

        g_info = []
        q_info = []
        c_info = []
        n_info = []

        for data_graph_path, query_graph_path in batch_pairs:
            finder = Find_Candidates(data_graph_path, query_graph_path)
            output_q_info, output_g_info = finder.cpp_GQL()
            
            g_info.append(output_g_info)
            q_info.append(output_q_info)
            c_info.append(self.groundtruth_dict[os.path.abspath(query_graph_path)])
            n_info.append(query_graph_path)

        return g_info, q_info, c_info, n_info
