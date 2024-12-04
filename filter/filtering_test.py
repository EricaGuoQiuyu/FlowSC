import os
import subprocess
import time
import re

# path to all datasets folder
datasets_path = '/data/qiuyug/matching_data/dataset/'

regex_patterns = {
    'candidate_nodes': r'num of candidate nodes: (\d+)',
    'candidate_edges': r'num of candidate edges: (\d+)',
    'elapsed_time': r'All elapsed time \(seconds\): ([\d.]+)'
}

TIMEOUT_SECONDS = 24 * 60 * 60 # 24 hours

# test for each dataset
for dataset_name in os.listdir(datasets_path):
    dataset_path = os.path.join(datasets_path, dataset_name)
    data_graph_path = os.path.join(dataset_path, 'data_graph', f'{dataset_name}.graph')
    query_graph_path = os.path.join(dataset_path, 'query_graph')

    if not os.path.isdir(dataset_path) or not os.path.isfile(data_graph_path) or not os.path.isdir(query_graph_path):
        continue

    start_time = time.time()
    print(f'Starting tests for dataset: {dataset_name} at {time.ctime(start_time)}')

    candidate_nodes_list = []
    candidate_edges_list = []
    elapsed_time_list = []

    for query_file in os.listdir(query_graph_path):
        current_time = time.time()
        if current_time - start_time > TIMEOUT_SECONDS:
            print(f"Timeout reached for dataset {dataset_name}. Stopping further tests.")
            break
        
        query_graph_file = os.path.join(query_graph_path, query_file)

        if not os.path.isfile(query_graph_file):
            continue

        # BipartitePlus command
        command = ['build1/matching/SubgraphMatching.out', '-d', data_graph_path, '-q', query_graph_file, '-filter', 'GQL'] #the filter used here is not `GQL` but `BipartitePlus`. Please do not modify the `-filter GQL` setting.
        #Fastest command
        #command = ['/home/qiuyug/Filter/FaSTest/build/Fastest', '-d', data_graph_path, '-q', query_graph_file, '--STRUCTURE', '3']
        #original GQL command
        #command = ['/home/qiuyug/Filter/OriginalGQL/SubgraphMatching/build/matching/SubgraphMatching.out', '-d', data_graph_path, '-q', query_graph_file, '-filter', 'GQL'] #orginal GQL command using GQL filter

        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            output = result.stdout

            candidate_nodes = int(re.search(regex_patterns['candidate_nodes'], output).group(1))
            candidate_edges = int(re.search(regex_patterns['candidate_edges'], output).group(1))
            elapsed_time = float(re.search(regex_patterns['elapsed_time'], output).group(1))

            candidate_nodes_list.append(candidate_nodes)
            candidate_edges_list.append(candidate_edges)
            elapsed_time_list.append(elapsed_time)

        except subprocess.CalledProcessError as e:
            print(f'Error executing command for {query_graph_file}: {e}')
        except AttributeError:
            print(f'Failed to parse output for {query_graph_file}.')

    end_time = time.time()
    print(f'Finished tests for dataset: {dataset_name} at {time.ctime(end_time)}')

    avg_candidate_nodes = sum(candidate_nodes_list) / len(candidate_nodes_list) if candidate_nodes_list else 0
    avg_candidate_edges = sum(candidate_edges_list) / len(candidate_edges_list) if candidate_edges_list else 0
    total_time = end_time - start_time

    results_file = os.path.join('/home/qiuyug/', f'{dataset_name}_results.txt') #path to save results
    with open(results_file, 'w') as f:
        f.write(f'Average candidate nodes: {avg_candidate_nodes}\n')
        f.write(f'Average candidate edges: {avg_candidate_edges}\n')
        f.write(f'Total batch test time (seconds): {total_time}\n')

    print(f'Results saved to {results_file}')