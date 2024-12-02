import torch
import numpy as np
import random
from torch.utils.data import Dataset
import networkx as nx
import scipy.sparse as sp
import os
from utils import Find_Candidates

class Graph_Pairs(Dataset):
    def __init__(self, dataset_name, query_size, root_path, groundtruth_path, batch_query_num, mode):
        super(Graph_Pairs, self).__init__()

        self.batch_query_num = batch_query_num

        self.mode = mode
        self.flag = 0

        self.data_graph_path = root_path + dataset_name + '/data_graph/' + '{}.graph'.format(dataset_name)
        self.query_graph_path = root_path + dataset_name + '/query_graph/query_{}/'.format(query_size)
        self.query_graph_files = os.listdir(self.query_graph_path)
        self.query_graph_num = len(self.query_graph_files)

        self.groundtruth_dict = {}
        with open(groundtruth_path + dataset_name + '.txt', 'r') as file:
            for line in file:
                log_file, embedding_value = line.strip().split(': ')
                self.groundtruth_dict[log_file] = float(embedding_value)

        self.k_folds = 5
        self.current_fold = 0

        self.createe_folds()
    
    def createe_folds(self):
        query_graphs_list = list(range(self.query_graph_num))
        np.random.shuffle(query_graphs_list)

        self.folds = np.array_split(query_graphs_list, self.k_folds)    

    def set_fold(self, fold):
        self.current_fold = fold
        self.create_batch()        
    
    def create_batch(self):
        test_idx = self.folds[self.current_fold]
        train_idx = np.concatenate([self.folds[i] for i in range(self.k_folds) if i != self.current_fold])

        self.test_query_num = len(test_idx)

        self.train_idx_batches = []
        self.test_idx_batches = []

        self.num_train_batches = len(train_idx) // self.batch_query_num
        self.num_test_batches = len(test_idx) // self.batch_query_num

        for i in range(self.num_train_batches):
            start_idx = i * self.batch_query_num
            end_idx = start_idx + self.batch_query_num
            self.train_idx_batches.append(train_idx[start_idx:end_idx])
        
        for i in range(self.num_test_batches):
            start_idx = i * self.batch_query_num
            end_idx = start_idx + self.batch_query_num
            self.test_idx_batches.append(test_idx[start_idx:end_idx])
        
        if len(train_idx) % self.batch_query_num != 0:
            self.train_idx_batches.append(train_idx[self.num_train_batches * self.batch_query_num : ])
        
        if len(test_idx) % self.batch_query_num != 0:
            self.test_idx_batches.append(test_idx[self.num_test_batches * self.batch_query_num : ])

    def __getitem__(self, index):

        if self.mode == 'train':
            batched_train_idx = self.train_idx_batches[index]
            query_graph_files = [self.query_graph_path + self.query_graph_files[n] for n in batched_train_idx]
        
        elif self.mode == 'test':
            batched_test_idx = self.test_idx_batches[index]
            query_graph_files = [self.query_graph_path + self.query_graph_files[n] for n in batched_test_idx]

        g_info = []
        q_info = []
        c_info = []
        n_info = []

        for query_file in query_graph_files:
            if self.flag == 0:
                self.finder = Find_Candidates(self.data_graph_path, query_file) 
                output_q_info, output_g_info = self.finder.cpp_GQL()
                g_info.append(output_g_info)
                q_info.append(output_q_info)
                c_info.append(self.groundtruth_dict[query_file])
                n_info.append(query_file)

                self.flag = 1 

            elif self.flag == 1:
                self.finder.update_query(query_file)
                output_q_info, output_g_info = self.finder.cpp_GQL()
                g_info.append(output_g_info)
                q_info.append(output_q_info)
                c_info.append(self.groundtruth_dict[query_file])
                n_info.append(query_file)
        
        return g_info, q_info, c_info, n_info
    
    def __len__(self):
        if self.mode == 'train':
            return self.num_train_batches
        elif self.mode == 'test':
            return self.num_test_batches
















