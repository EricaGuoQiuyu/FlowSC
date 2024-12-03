import numpy as np
import os
import torch
from data_loader import Graph_Pairs
from torch.utils.data import DataLoader
from learner import QErrorLoss, Flow_Learner
from utils import generate_adjacency_matrix, bfs_layers, data_flow, query_flow, generate_query_adj
import argparse
import time
from datetime import datetime

def _collate_(samples):
    g_info, q_info, c_info, n_info = map(list, zip(*samples))
    return g_info, q_info, c_info, n_info

def main_counting():
    #dataset_name_list = ['yeast', 'human', 'hprd', 'wordnet', 'dblp', 'eu2005', 'youtube', 'patents']
    dataset_name = 'wordnet' #args.dataset_name
    query_size = 4 #args.query_size

    device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
    batch_query_num = args.batch_query_num
    out_dim= 128
    eval_loss = QErrorLoss()

    root_path = args.root_path
    groundtruth_path = args.groundtruth_path

    dataset = Graph_Pairs(dataset_name, query_size, root_path, groundtruth_path, batch_query_num=batch_query_num, mode='train')
    
    fold_results = []
    current_date = datetime.now().date()

    for fold in range(dataset.k_folds):

        model = Flow_Learner(out_dim = out_dim).to(device) 

        pretrained_model_path = 'pretrained_model.pt' #load pretrained parameters
        model.load_state_dict(torch.load(pretrained_model_path))

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)

        print('starting fold: ', fold)
        dataset.set_fold(fold)

        dataset.mode = 'train'
        model.train()

        training_start_time = time.time()

        for epoch in range(15): 
            print('starting the {} epoch training on {}'.format(epoch, dataset_name))
            
            train_data = DataLoader(dataset, shuffle = True, num_workers=4, collate_fn = _collate_)
            for b, (g_info, q_info, c_info, n_info) in enumerate(train_data):
                all_loss = 0

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

                    loss = eval_loss(counts, true_counts)

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    all_loss += loss
                    
                mean_loss = all_loss / (batch_query_num)
                print('loss: ', mean_loss.item())

            scheduler.step()    

        training_end_time = time.time()
        print('Training time: ', training_end_time - training_start_time)

        print('starting testing on {}'.format(dataset_name))
        dataset.mode = 'test'
        model.eval()

        test_data = DataLoader(dataset, shuffle=False, num_workers=4, collate_fn=_collate_)
        batch_loss = []

        start_time = time.time()
        test_result_dict = dict()
        test_sign_dict = dict()

        for b, (g_info, q_info, c_info, n_info) in enumerate(test_data):
            test_loss = []
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

                loss = eval_loss(counts, true_counts)

                if counts.item() > true_counts:
                    test_sign_dict[query_name] = 'overestimate'
                else:
                    test_sign_dict[query_name] = 'underestimate'

                test_result_dict[query_name] = loss.item()
                test_loss.append(loss.item())  

            batch_loss.append(np.mean(np.array(test_loss)))
        
        end_time = time.time()
        avg_time = (end_time - start_time) / dataset.test_query_num
        avg_loss = np.mean(np.array(batch_loss))

        print('average loss per dataset: ', avg_loss)
        print('average query time: ', avg_time)

        fold_results.append(avg_loss)
        
        if not os.path.exists('result_dir/{}_{}_result_{}'.format(dataset_name, query_size, current_date)):
            with open('result_dir/{}_{}_result_{}'.format(dataset_name, query_size, current_date), 'w') as file:
                pass #create file
        
        with open('result_dir/{}_{}_result_{}'.format(dataset_name, query_size, current_date), 'a') as file:
            file.write('fold: {}\n'.format(fold))
            file.write('traing time: {}\n'.format(training_end_time - training_start_time))
            for key, value in test_result_dict.items():
                file.write(f"{key}: {value} {test_sign_dict[key]}\n")
            file.write('average loss: {}\n'.format(np.mean(np.array(batch_loss))))
            file.write('average query time: {}\n'.format(avg_time))
          
    print('average loss of 5 folds: ', np.mean(np.array(fold_results)))
    print('done.')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--root_path', type=str, help='path to all datasets', default='dataset/')
    argparser.add_argument('--groundtruth_path', type=str, help='path to groundtruth folder', default='groundtruth/')
    argparser.add_argument('--batch_query_num', type=int, help='batch size', default=10)
    argparser.add_argument('--dataset_name', type=str, help='dataset name', default='yeast')
    argparser.add_argument('--query_size', type=int, help='query graph size', default=4)
    args = argparser.parse_args()
    main_counting()



    
    