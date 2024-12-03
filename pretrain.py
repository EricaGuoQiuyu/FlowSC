import numpy as np
import torch
from pretrain_data_loader import Graph_Pairs
from torch.utils.data import DataLoader
from learner import QErrorLoss, Flow_Learner, MSLELoss
from utils import generate_adjacency_matrix, bfs_layers, data_flow, query_flow, generate_query_adj
import argparse
import time

def _collate_(samples):
    g_info, q_info, c_info, n_info = map(list, zip(*samples))
    return g_info, q_info, c_info, n_info

def main_counting():
    #pretrain on 20% of the queries from each dataset across all query sizes
    dataset_name_list = ['yeast', 'human', 'hprd', 'wordnet', 'dblp', 'eu2005', 'youtube', 'patents']

    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
    batch_query_num = args.batch_query_num
    out_dim= 128
    eval_loss = QErrorLoss()
    msle_loss = MSLELoss()

    root_path = args.root_path
    pretrain_groundtruth_path = args.pretrain_groundtruth_path

    dataset = Graph_Pairs(dataset_name_list, root_path, pretrain_groundtruth_path, batch_query_num=batch_query_num)

    model = Flow_Learner(out_dim = out_dim).to(device) 
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    model.train()

    best_test_loss = float('inf')
    for epoch in range(20): 

        print('starting the {} epoch pretraining'.format(epoch))
        
        train_data = DataLoader(dataset, shuffle = True, num_workers=8, collate_fn = _collate_)
        batch_loss = []
        for b, (g_info, q_info, c_info, n_info) in enumerate(train_data):
            all_loss = 0
            all_predictions = []
            all_true_counts = []

            for i in range(batch_query_num): 

                cs_size, candidates_size, g_nodes, g_edges = g_info[0][i]
                q_vertices, q_labels, q_degrees, q_edges= q_info[0][i]

                true_counts = torch.tensor(c_info[0][i]).to(device)
                query_name = n_info[0][i]

                root = cs_size.index(max(cs_size))
                layers = bfs_layers(q_edges, root)                   

                query_adj = generate_query_adj(q_edges, len(q_vertices))
                query_generator = query_flow(query_adj, q_labels, layers)

                adj = generate_adjacency_matrix(g_edges, candidates_size)
                data_generator = data_flow(q_labels, g_nodes, layers, root, adj)             

                counts = model(query_generator, data_generator, device)

                all_predictions.append(counts)
                all_true_counts.append(true_counts)

                loss = eval_loss(counts, true_counts) 
                all_loss += loss                    

            prediction_tensor = torch.stack(all_predictions)
            true_count_tensor = torch.stack(all_true_counts)
            loss = msle_loss(prediction_tensor, true_count_tensor) + all_loss / batch_query_num
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            mean_loss = loss.item()
            print('Batch loss: ', mean_loss)
            batch_loss.append(mean_loss)

        avg_loss = np.mean(np.array(batch_loss))
        print('Epoch {} average loss: {}'.format(epoch, avg_loss))

        if avg_loss < best_test_loss:
            best_test_loss = avg_loss
            torch.save(model.state_dict(), 'new_pretrained_model.pt')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--root_path', type=str, help='raw dataset path', default='/data/qiuyug/matching_data/dataset_pretrain/')
    argparser.add_argument('--pretrain_groundtruth_path', type=str, help='path to exact counts of queries for pretraining', default='/data/qiuyug/matching_data/pretrain_gt/')
    argparser.add_argument('--batch_query_num', type=int, help='batch size', default=10)
    args = argparser.parse_args()
    main_counting()



    
    