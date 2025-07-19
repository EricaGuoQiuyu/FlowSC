//
// Created by ssunah on 11/20/18.
//

#include "FilterVertices.h"
#include "GenerateFilteringPlan.h"
#include <memory.h>
#include <utility/graphoperations.h>
#include <vector>
#include <algorithm>
#define INVALID_VERTEX_ID 100000000

bool
FilterVertices::LDFFilter(const Graph *data_graph, const Graph *query_graph, ui **&candidates, ui *&candidates_count) {
    allocateBuffer(data_graph, query_graph, candidates, candidates_count);

    for (ui i = 0; i < query_graph->getVerticesCount(); ++i) {
        LabelID label = query_graph->getVertexLabel(i);
        ui degree = query_graph->getVertexDegree(i);

        ui data_vertex_num;
        const ui* data_vertices = data_graph->getVerticesByLabel(label, data_vertex_num);

        for (ui j = 0; j < data_vertex_num; ++j) {
            ui data_vertex = data_vertices[j];
            if (data_graph->getVertexDegree(data_vertex) >= degree) {
                candidates[i][candidates_count[i]++] = data_vertex;
            }
        }

        if (candidates_count[i] == 0) {
            return false;
        }
    }

    return true;
}

bool
FilterVertices::NLFFilter(const Graph *data_graph, const Graph *query_graph, ui **&candidates, ui *&candidates_count) {
    allocateBuffer(data_graph, query_graph, candidates, candidates_count);

    for (ui i = 0; i < query_graph->getVerticesCount(); ++i) {
        VertexID query_vertex = i;
        computeCandidateWithNLF(data_graph, query_graph, query_vertex, candidates_count[query_vertex], candidates[query_vertex]);

        if (candidates_count[query_vertex] == 0) {
            return false;
        }
    }

    return true;
}

bool
FilterVertices::GQLFilter(const Graph *data_graph, const Graph *query_graph, ui **&candidates, ui *&candidates_count) {

    // Local refinement.
    if (!NLFFilter(data_graph, query_graph, candidates, candidates_count))
        return false;

    // Allocate buffer.
    ui query_vertex_num = query_graph->getVerticesCount();
    ui data_vertex_num = data_graph->getVerticesCount();

    bool** valid_candidates = new bool*[query_vertex_num];
    for (ui i = 0; i < query_vertex_num; ++i) {
        valid_candidates[i] = new bool[data_vertex_num];
        memset(valid_candidates[i], 0, sizeof(bool) * data_vertex_num);
    }

    ui query_graph_max_degree = query_graph->getGraphMaxDegree();
    ui data_graph_max_degree = data_graph->getGraphMaxDegree();

    int* left_to_right_offset = new int[query_graph_max_degree + 1];
    int* left_to_right_edges = new int[query_graph_max_degree * data_graph_max_degree];
    int* left_to_right_match = new int[query_graph_max_degree];
    int* right_to_left_match = new int[data_graph_max_degree];
    int* match_visited = new int[data_graph_max_degree + 1];
    int* match_queue = new int[query_vertex_num];
    int* match_previous = new int[data_graph_max_degree + 1];
    
    // Record valid candidate vertices for each query vertex.
    for (ui i = 0; i < query_vertex_num; ++i) {
        VertexID query_vertex = i;
        for (ui j = 0; j < candidates_count[query_vertex]; ++j) {
            VertexID data_vertex = candidates[query_vertex][j];
            valid_candidates[query_vertex][data_vertex] = true;
        }
    }

    //triangle neighbors for each query_vertex
    std::unordered_map<std::pair<VertexID, VertexID>, std::unordered_set<VertexID>, pair_hash, pair_equal> NT_neighbors;

    VertexID max_degree_vertex = 0;
    ui max_degree = 0;
    
    for (ui i = 0; i < query_vertex_num; ++i) {
        VertexID query_vertex = i;
        ui query_neighbors_num;

        const VertexID* query_vertex_neighbors = query_graph->getVertexNeighbors(query_vertex, query_neighbors_num);

        for (int u =0; u < query_neighbors_num; ++u) {
            for (int v = u+1; v < query_neighbors_num; ++v){
                VertexID n1 = query_vertex_neighbors[u];
                VertexID n2 = query_vertex_neighbors[v];
                if (query_graph->checkEdgeExistence(n1, n2)) {
                    NT_neighbors[{query_vertex, n1}].insert(n2); 
                    NT_neighbors[{query_vertex, n2}].insert(n1);
                }
            }
        }

        if (query_graph->getVertexDegree(query_vertex) > max_degree) {
            max_degree = query_graph->getVertexDegree(query_vertex);
            max_degree_vertex = query_vertex;
        }        

    }

    //refinement order
    TreeNode *tree = new TreeNode[query_vertex_num];
    ui *query_vertex_order = new ui[query_vertex_num];
    GraphOperations::bfsTraversal(query_graph, max_degree_vertex, tree, query_vertex_order); 

    // Global refinement.
    std::unordered_map<std::pair<VertexID, VertexID>, std::unordered_set<VertexID>, pair_hash, pair_equal> candidate_neighbors; //valid candidate edges

    //const double stop_threshold = 0.9; 
    //We simply set the number of refinemnet to 5 here, the first refinement is original GQL, the rest are BipartitePlus.
    //You can algin with the settings in our paper by setting the activation threshold and stopping threshold.
    for (ui l = 0; l < 5; ++l) {
        ui previous_candidates_count = 0;
        ui current_candidates_count = 0;

        VertexID max_filtered_vertex = 0;
        double min_retained_ratio = 1.0;

        for (ui i = 0; i < query_vertex_num; ++i) {
            VertexID query_vertex = query_vertex_order[i];

            ui previous_candidates_per_query = 0;
            ui current_candidates_per_query = 0;
            double current_retain_ratio = 1.0;

            for (ui j = 0; j < candidates_count[query_vertex]; ++j) {
                VertexID data_vertex = candidates[query_vertex][j];

                if (data_vertex == INVALID_VERTEX_ID){
                    continue;
                }

                previous_candidates_count += 1;
                previous_candidates_per_query += 1;

                // GQL for first refinement
                if (l < 1) {
                    if (!verifyExactTwigIso(data_graph, query_graph, data_vertex, query_vertex, valid_candidates,
                                            left_to_right_offset, left_to_right_edges, left_to_right_match,
                                            right_to_left_match, match_visited, match_queue, match_previous, candidate_neighbors)) {
                        candidate_neighbors.erase({query_vertex, data_vertex});
                        candidates[query_vertex][j] = INVALID_VERTEX_ID;
                        valid_candidates[query_vertex][data_vertex] = false;
                    }
                    else {
                        current_candidates_count += 1;
                        current_candidates_per_query += 1;
                    }

                }
                // one can set threshold for activation of BipartitePlus by using current_retrain_ratio
                else {
                    if (!verifyExactTwigIso_Plus(data_graph, query_graph, data_vertex, query_vertex, valid_candidates,
                                            left_to_right_offset, left_to_right_edges, left_to_right_match,
                                            right_to_left_match, match_visited, match_queue, match_previous, candidate_neighbors, NT_neighbors)) {
                                        
                        candidate_neighbors.erase({query_vertex, data_vertex});//remove conditional neighbors of data_vertex
                        candidates[query_vertex][j] = INVALID_VERTEX_ID;
                        valid_candidates[query_vertex][data_vertex] = false; 

                    }
                    else {
                        current_candidates_count += 1;
                        current_candidates_per_query += 1;
                    }
                }

            }

            current_retain_ratio = (double)current_candidates_per_query / previous_candidates_per_query;
            if (current_retain_ratio < min_retained_ratio) {
                min_retained_ratio = current_retain_ratio;
                max_filtered_vertex = query_vertex;
            }
        }

        double refinement_ratio = current_candidates_count / previous_candidates_count;
        
        /*  //one can set threshold for stopping the refinement here
        if (refinement_ratio > stop_threshold) { 
            break;
        } 
        */ 

        if (l < 1) {
            if (max_filtered_vertex != max_degree_vertex) {
                GraphOperations::bfsTraversal(query_graph, max_filtered_vertex, tree, query_vertex_order);
            }

        }
        else {
            GraphOperations::bfsTraversal(query_graph, max_filtered_vertex, tree, query_vertex_order);

        }
         
    }

    delete[] tree;
    delete[] query_vertex_order;

    compactCandidates(candidates, candidates_count, query_vertex_num);
   
    std::unordered_map<VertexID, std::unordered_set<VertexID>> query_to_data_map;
    std::unordered_map<VertexID, std::unordered_set<VertexID>> data_to_query_map; 
    std::unordered_set<VertexID> all_valid_nodes; 
    for (ui i = 0; i < query_vertex_num; ++i) {
        VertexID query_vertex = i;
        for (ui j = 0; j < candidates_count[query_vertex]; ++j) {
            VertexID data_vertex = candidates[query_vertex][j];
            if (data_vertex != INVALID_VERTEX_ID) {
                query_to_data_map[query_vertex].insert(data_vertex);
                data_to_query_map[data_vertex].insert(query_vertex);
                all_valid_nodes.insert(data_vertex);
            }
        }
    }  
      
    std::unordered_map<VertexID, std::unordered_set<VertexID>> candidate_edges;
    for (const auto& pair : candidate_neighbors) {
        if (valid_candidates[pair.first.first][pair.first.second]){

            const std::unordered_set<VertexID>& neighbors_set = pair.second;
            for (VertexID neighbor : neighbors_set) { 
                if (all_valid_nodes.find(neighbor) != all_valid_nodes.end()){
                    candidate_edges[pair.first.second].insert(neighbor);
                    candidate_edges[neighbor].insert(pair.first.second);
                }
                else {
                    continue;
                }

            }  
      
        }
        
    }

    //reindexing and printing
    std::unordered_map<VertexID, VertexID> reindex_map;
    VertexID new_index = 0;
    for (const auto& node : all_valid_nodes) {
        reindex_map[node] = new_index++;
    }
    
    std::cout << "Candidate Sizes : ";
    for (ui u = 0; u < query_vertex_num; ++u) {
        VertexID query_vertex = u;
        std::cout << query_to_data_map[query_vertex].size() << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Candidate Nodes : " ;
    for (ui u = 0; u < query_vertex_num; ++u) {
        VertexID query_vertex = u;
        for (const auto& data_vertex : query_to_data_map[query_vertex]) {
            if (reindex_map.find(data_vertex) == reindex_map.end()) {
                std::cout<<"invaliddata_vertex" <<std::endl;
            }
            else {
                std::cout << reindex_map[data_vertex] << " "; 
            }
            
        }
        if (u < query_vertex_num -1 ) {
            std::cout << "| ";
        }
    }
    std::cout << std::endl;

    ui num_edges = 0;
    std::cout << "Candidate Edges : ";
    for (const auto& pair : candidate_edges) {
        num_edges += pair.second.size();
        VertexID vertex = pair.first;
        for (const auto& neighbor : pair.second) {
            if (reindex_map.find(vertex) == reindex_map.end()) {
                std::cout<< "invalid_vertex_in_edges" << std::endl;
            }
            else if (reindex_map.find(neighbor) == reindex_map.end()) {
                std::cout << "invalid_neighbor_in_edges" << std::endl;
            }
            else {
                std::cout << "(" << reindex_map[vertex] << "," << reindex_map[neighbor] << ") ";
            }
            
        }
    }
    std::cout << std::endl;
    
    num_edges = num_edges / 2;
    std::cout << "num of candidate nodes: " << all_valid_nodes.size() << std::endl;
    std::cout << "num of candidate edges: " << num_edges << std::endl; 
    
    // Release memory.
    candidate_neighbors.clear();   
    query_to_data_map.clear();
    candidate_edges.clear();
    all_valid_nodes.clear();

    for (ui i = 0; i < query_vertex_num; ++i) {
        delete[] valid_candidates[i];
    }
    delete[] valid_candidates;
    delete[] left_to_right_offset;
    delete[] left_to_right_edges;
    delete[] left_to_right_match;
    delete[] right_to_left_match;
    delete[] match_visited;
    delete[] match_queue;
    delete[] match_previous;

    return true;
    //return isCandidateSetValid(candidates, candidates_count, query_vertex_num);
}

void FilterVertices::allocateBuffer(const Graph *data_graph, const Graph *query_graph, ui **&candidates,
                                    ui *&candidates_count) {
    ui query_vertex_num = query_graph->getVerticesCount();
    ui candidates_max_num = data_graph->getGraphMaxLabelFrequency();

    candidates_count = new ui[query_vertex_num];
    memset(candidates_count, 0, sizeof(ui) * query_vertex_num);

    candidates = new ui*[query_vertex_num];

    for (ui i = 0; i < query_vertex_num; ++i) {
        candidates[i] = new ui[candidates_max_num];
    }
}

bool //original GQL with candidate edge pruning
FilterVertices::verifyExactTwigIso(const Graph *data_graph, const Graph *query_graph, ui data_vertex, ui query_vertex,
                                   bool **valid_candidates, int *left_to_right_offset, int *left_to_right_edges,
                                   int *left_to_right_match, int *right_to_left_match, int* match_visited,
                                   int* match_queue, int* match_previous, 
                                   std::unordered_map<std::pair<VertexID, VertexID>, std::unordered_set<VertexID>, pair_hash, pair_equal> &candidate_neighbors) {
    // Construct the bipartite graph between N(query_vertex) and N(data_vertex)
    ui left_partition_size;
    ui right_partition_size;
    const VertexID* query_vertex_neighbors = query_graph->getVertexNeighbors(query_vertex, left_partition_size);
    const VertexID* data_vertex_neighbors = data_graph->getVertexNeighbors(data_vertex, right_partition_size);

    std::unordered_set<VertexID> current_candidate_neighbors; 
    current_candidate_neighbors.reserve(right_partition_size);

    ui edge_count = 0;

    for (int i = 0; i < left_partition_size; ++i) {
        VertexID query_vertex_neighbor = query_vertex_neighbors[i];
        left_to_right_offset[i] = edge_count;

        for (int j = 0; j < right_partition_size; ++j) {
            VertexID data_vertex_neighbor = data_vertex_neighbors[j];

            if (valid_candidates[query_vertex_neighbor][data_vertex_neighbor]) {
                left_to_right_edges[edge_count++] = j;

                current_candidate_neighbors.insert(data_vertex_neighbor);
            }
        }

    }
    left_to_right_offset[left_partition_size] = edge_count;

    memset(left_to_right_match, -1, left_partition_size * sizeof(int));
    memset(right_to_left_match, -1, right_partition_size * sizeof(int));

    GraphOperations::match_bfs(left_to_right_offset, left_to_right_edges, left_to_right_match, right_to_left_match,
                               match_visited, match_queue, match_previous, left_partition_size, right_partition_size);
    for (int i = 0; i < left_partition_size; ++i) {
        if (left_to_right_match[i] == -1) {
            return false;
        }
    }

    candidate_neighbors[{query_vertex, data_vertex}] = std::move(current_candidate_neighbors);
    
    return true;
}

bool //BipartitePlus with candidate edge pruning and triangle check
FilterVertices::verifyExactTwigIso_Plus(const Graph *data_graph, const Graph *query_graph, ui data_vertex, ui query_vertex,
                                   bool **valid_candidates, int *left_to_right_offset, int *left_to_right_edges,
                                   int *left_to_right_match, int *right_to_left_match, int* match_visited,
                                   int* match_queue, int* match_previous, 
                                   std::unordered_map<std::pair<VertexID, VertexID>, std::unordered_set<VertexID>, pair_hash, pair_equal> &candidate_neighbors,
                                   std::unordered_map<std::pair<VertexID, VertexID>, std::unordered_set<VertexID>, pair_hash, pair_equal> NT_neighbors) {
    // Construct the bipartite graph between N(query_vertex) and N(data_vertex)
    ui left_partition_size;
    ui right_partition_size;
    const VertexID* query_vertex_neighbors = query_graph->getVertexNeighbors(query_vertex, left_partition_size);
    const VertexID* data_vertex_neighbors = data_graph->getVertexNeighbors(data_vertex, right_partition_size);

    std::unordered_set<VertexID> current_candidate_neighbors; 
    current_candidate_neighbors.reserve(right_partition_size);

    ui edge_count = 0;
    bool is_valid = true;
    for (ui i = 0; i < left_partition_size; ++i) {
        VertexID query_vertex_neighbor = query_vertex_neighbors[i];
        left_to_right_offset[i] = edge_count;

        bool any_valid_data_vertex_neighbor = false;
        for (ui j = 0; j < right_partition_size; ++j) {
            VertexID data_vertex_neighbor = data_vertex_neighbors[j];

            bool data_vertex_neighbor_all_cycle_check = false;
            if (valid_candidates[query_vertex_neighbor][data_vertex_neighbor]) {
                data_vertex_neighbor_all_cycle_check = true;

                std::unordered_set<VertexID> query_cycle_neighbors;
                if (!NT_neighbors[{query_vertex, query_vertex_neighbor}].empty()){ //need triangle matching check

                    const auto& query_cycle_neighbors = NT_neighbors[{query_vertex, query_vertex_neighbor}];
                    
                    bool q_cycle_neighbor_check; 
                    for (const auto& q_cycle_neighbor : query_cycle_neighbors) {  
                        q_cycle_neighbor_check = false;
                        for (ui k = 0; k < right_partition_size; ++k) {
                            VertexID g_cycle_neighbor = data_vertex_neighbors[k];
                            if (!valid_candidates[q_cycle_neighbor][g_cycle_neighbor]) {
                                continue;
                            }
                             
                            if (candidate_neighbors.find({query_vertex_neighbor, data_vertex_neighbor}) != candidate_neighbors.end() 
                                    && candidate_neighbors[{query_vertex_neighbor, data_vertex_neighbor}].find(g_cycle_neighbor) != candidate_neighbors[{query_vertex_neighbor, data_vertex_neighbor}].end()
                                    && candidate_neighbors[{query_vertex, data_vertex}].find(g_cycle_neighbor) != candidate_neighbors[{query_vertex, data_vertex}].end()
                                    && valid_candidates[q_cycle_neighbor][g_cycle_neighbor]) { 
                                
                                q_cycle_neighbor_check = true;
                                break; //one data triangle is enough for a single query triangle
                            }

                        }

                        if (!q_cycle_neighbor_check){
                            data_vertex_neighbor_all_cycle_check = false;
                            break; 
                        }
                        
                    }

                }

                if (data_vertex_neighbor_all_cycle_check) {
                    left_to_right_edges[edge_count++] = j;
                    current_candidate_neighbors.insert(data_vertex_neighbor); 
                    any_valid_data_vertex_neighbor = true;
                }

            }
        }

        if (!any_valid_data_vertex_neighbor) {
            is_valid = false;
            break;
        }

    }

    if (!is_valid) {
        return false; //no need for injective matching check
    }

    left_to_right_offset[left_partition_size] = edge_count;

    memset(left_to_right_match, -1, left_partition_size * sizeof(int));
    memset(right_to_left_match, -1, right_partition_size * sizeof(int));

    GraphOperations::match_bfs(left_to_right_offset, left_to_right_edges, left_to_right_match, right_to_left_match,
                               match_visited, match_queue, match_previous, left_partition_size, right_partition_size);
    for (int i = 0; i < left_partition_size; ++i) {
        if (left_to_right_match[i] == -1) {
            return false;
        }
    }

    candidate_neighbors[{query_vertex, data_vertex}] = std::move(current_candidate_neighbors);
    return true;
}

void FilterVertices::compactCandidates(ui **&candidates, ui *&candidates_count, ui query_vertex_num) {
    for (ui i = 0; i < query_vertex_num; ++i) {
        VertexID query_vertex = i;
        ui next_position = 0;
        for (ui j = 0; j < candidates_count[query_vertex]; ++j) {
            VertexID data_vertex = candidates[query_vertex][j];

            if (data_vertex != INVALID_VERTEX_ID) {
                candidates[query_vertex][next_position++] = data_vertex;
            }
        }
        candidates_count[query_vertex] = next_position;

    }
}


bool FilterVertices::isCandidateSetValid(ui **&candidates, ui *&candidates_count, ui query_vertex_num) {
    for (ui i = 0; i < query_vertex_num; ++i) {
        if (candidates_count[i] == 0)
            return false;
    }
    return true;
}

void
FilterVertices::computeCandidateWithNLF(const Graph *data_graph, const Graph *query_graph, VertexID query_vertex,
                                               ui &count, ui *buffer) {
    LabelID label = query_graph->getVertexLabel(query_vertex);
    ui degree = query_graph->getVertexDegree(query_vertex);
#if OPTIMIZED_LABELED_GRAPH == 1
    const std::unordered_map<LabelID, ui>* query_vertex_nlf = query_graph->getVertexNLF(query_vertex);
#endif
    ui data_vertex_num;
    const ui* data_vertices = data_graph->getVerticesByLabel(label, data_vertex_num);
    count = 0;
    for (ui j = 0; j < data_vertex_num; ++j) {
        ui data_vertex = data_vertices[j];
        if (data_graph->getVertexDegree(data_vertex) >= degree) {

            // NFL check
#if OPTIMIZED_LABELED_GRAPH == 1
            const std::unordered_map<LabelID, ui>* data_vertex_nlf = data_graph->getVertexNLF(data_vertex);

            if (data_vertex_nlf->size() >= query_vertex_nlf->size()) {
                bool is_valid = true;

                for (auto element : *query_vertex_nlf) {
                    auto iter = data_vertex_nlf->find(element.first);
                    if (iter == data_vertex_nlf->end() || iter->second < element.second) {
                        is_valid = false;
                        break;
                    }
                }

                if (is_valid) {
                    if (buffer != NULL) {
                        buffer[count] = data_vertex;
                    }
                    count += 1;
                }
            }
#else
            if (buffer != NULL) {
                buffer[count] = data_vertex;
            }
            count += 1;
#endif
        }
    }

}

void FilterVertices::computeCandidateWithLDF(const Graph *data_graph, const Graph *query_graph, VertexID query_vertex,
                                             ui &count, ui *buffer) {
    LabelID label = query_graph->getVertexLabel(query_vertex);
    ui degree = query_graph->getVertexDegree(query_vertex);
    count = 0;
    ui data_vertex_num;
    const ui* data_vertices = data_graph->getVerticesByLabel(label, data_vertex_num);

    if (buffer == NULL) {
        for (ui i = 0; i < data_vertex_num; ++i) {
            VertexID v = data_vertices[i];
            if (data_graph->getVertexDegree(v) >= degree) {
                count += 1;
            }
        }
    }
    else {
        for (ui i = 0; i < data_vertex_num; ++i) {
            VertexID v = data_vertices[i];
            if (data_graph->getVertexDegree(v) >= degree) {
                buffer[count++] = v;
            }
        }
    }
}

void FilterVertices::generateCandidates(const Graph *data_graph, const Graph *query_graph, VertexID query_vertex,
                                       VertexID *pivot_vertices, ui pivot_vertices_count, VertexID **candidates,
                                       ui *candidates_count, ui *flag, ui *updated_flag) {
    LabelID query_vertex_label = query_graph->getVertexLabel(query_vertex);
    ui query_vertex_degree = query_graph->getVertexDegree(query_vertex);
#if OPTIMIZED_LABELED_GRAPH == 1
    const std::unordered_map<LabelID , ui>* query_vertex_nlf = query_graph->getVertexNLF(query_vertex);
#endif
    ui count = 0;
    ui updated_flag_count = 0;
    for (ui i = 0; i < pivot_vertices_count; ++i) {
        VertexID pivot_vertex = pivot_vertices[i];

        for (ui j = 0; j < candidates_count[pivot_vertex]; ++j) {
            VertexID v = candidates[pivot_vertex][j];

            if (v == INVALID_VERTEX_ID)
                continue;
            ui v_nbrs_count;
            const VertexID* v_nbrs = data_graph->getVertexNeighbors(v, v_nbrs_count);

            for (ui k = 0; k < v_nbrs_count; ++k) {
                VertexID v_nbr = v_nbrs[k];
                LabelID v_nbr_label = data_graph->getVertexLabel(v_nbr);
                ui v_nbr_degree = data_graph->getVertexDegree(v_nbr);

                if (flag[v_nbr] == count && v_nbr_label == query_vertex_label && v_nbr_degree >= query_vertex_degree) {
                    flag[v_nbr] += 1;

                    if (count == 0) {
                        updated_flag[updated_flag_count++] = v_nbr;
                    }
                }
            }
        }

        count += 1;
    }

    for (ui i = 0; i < updated_flag_count; ++i) {
        VertexID v = updated_flag[i];
        if (flag[v] == count) {
            // NLF filter.
#if OPTIMIZED_LABELED_GRAPH == 1
            const std::unordered_map<LabelID, ui>* data_vertex_nlf = data_graph->getVertexNLF(v);

            if (data_vertex_nlf->size() >= query_vertex_nlf->size()) {
                bool is_valid = true;

                for (auto element : *query_vertex_nlf) {
                    auto iter = data_vertex_nlf->find(element.first);
                    if (iter == data_vertex_nlf->end() || iter->second < element.second) {
                        is_valid = false;
                        break;
                    }
                }

                if (is_valid) {
                    candidates[query_vertex][candidates_count[query_vertex]++] = v;
                }
            }
#else
            candidates[query_vertex][candidates_count[query_vertex]++] = v;
#endif
        }
    }

    for (ui i = 0; i < updated_flag_count; ++i) {
        ui v = updated_flag[i];
        flag[v] = 0;
    }
}

void FilterVertices::pruneCandidates(const Graph *data_graph, const Graph *query_graph, VertexID query_vertex,
                                    VertexID *pivot_vertices, ui pivot_vertices_count, VertexID **candidates,
                                    ui *candidates_count, ui *flag, ui *updated_flag) {
    LabelID query_vertex_label = query_graph->getVertexLabel(query_vertex);
    ui query_vertex_degree = query_graph->getVertexDegree(query_vertex);

    ui count = 0;
    ui updated_flag_count = 0;
    for (ui i = 0; i < pivot_vertices_count; ++i) {
        VertexID pivot_vertex = pivot_vertices[i];

        for (ui j = 0; j < candidates_count[pivot_vertex]; ++j) {
            VertexID v = candidates[pivot_vertex][j];

            if (v == INVALID_VERTEX_ID)
                continue;
            ui v_nbrs_count;
            const VertexID* v_nbrs = data_graph->getVertexNeighbors(v, v_nbrs_count);

            for (ui k = 0; k < v_nbrs_count; ++k) {
                VertexID v_nbr = v_nbrs[k];
                LabelID v_nbr_label = data_graph->getVertexLabel(v_nbr);
                ui v_nbr_degree = data_graph->getVertexDegree(v_nbr);

                if (flag[v_nbr] == count && v_nbr_label == query_vertex_label && v_nbr_degree >= query_vertex_degree) {
                    flag[v_nbr] += 1;

                    if (count == 0) {
                        updated_flag[updated_flag_count++] = v_nbr;
                    }
                }
            }
        }

        count += 1;
    }

    for (ui i = 0; i < candidates_count[query_vertex]; ++i) {
        ui v = candidates[query_vertex][i];
        if (v == INVALID_VERTEX_ID)
            continue;

        if (flag[v] != count) {
            candidates[query_vertex][i] = INVALID_VERTEX_ID;
        }
    }

    for (ui i = 0; i < updated_flag_count; ++i) {
        ui v = updated_flag[i];
        flag[v] = 0;
    }
}

void FilterVertices::printCandidatesInfo(const Graph *query_graph, ui *candidates_count, std::vector<ui> &optimal_candidates_count) {
    std::vector<std::pair<VertexID, ui>> core_vertices;
    std::vector<std::pair<VertexID, ui>> tree_vertices;
    std::vector<std::pair<VertexID, ui>> leaf_vertices;

    ui query_vertices_num = query_graph->getVerticesCount();
    double sum = 0;
    double optimal_sum = 0;
    for (ui i = 0; i < query_vertices_num; ++i) {
        VertexID cur_vertex = i;
        ui count = candidates_count[cur_vertex];
        sum += count;
        optimal_sum += optimal_candidates_count[cur_vertex];

        if (query_graph->getCoreValue(cur_vertex) > 1) {
            core_vertices.emplace_back(std::make_pair(cur_vertex, count));
        }
        else {
            if (query_graph->getVertexDegree(cur_vertex) > 1) {
                tree_vertices.emplace_back(std::make_pair(cur_vertex, count));
            }
            else {
                leaf_vertices.emplace_back(std::make_pair(cur_vertex, count));
            }
        }
    }

    printf("#Candidate Information: CoreVertex(%zu), TreeVertex(%zu), LeafVertex(%zu)\n", core_vertices.size(), tree_vertices.size(), leaf_vertices.size());

    for (auto candidate_info : core_vertices) {
        printf("CoreVertex %u: %u, %u \n", candidate_info.first, candidate_info.second, optimal_candidates_count[candidate_info.first]);
    }

    for (auto candidate_info : tree_vertices) {
        printf("TreeVertex %u: %u, %u\n", candidate_info.first, candidate_info.second, optimal_candidates_count[candidate_info.first]);
    }

    for (auto candidate_info : leaf_vertices) {
        printf("LeafVertex %u: %u, %u\n", candidate_info.first, candidate_info.second, optimal_candidates_count[candidate_info.first]);
    }

    printf("Total #Candidates: %.1lf, %.1lf\n", sum, optimal_sum);
}

void FilterVertices::sortCandidates(ui **candidates, ui *candidates_count, ui num) {
    for (ui i = 0; i < num; ++i) {
        std::sort(candidates[i], candidates[i] + candidates_count[i]);
    }
}

double
FilterVertices::computeCandidatesFalsePositiveRatio(const Graph *data_graph, const Graph *query_graph, ui **candidates,
                                                    ui *candidates_count, std::vector<ui> &optimal_candidates_count) {
    ui query_vertices_count = query_graph->getVerticesCount();
    ui data_vertices_count = data_graph->getVerticesCount();

    std::vector<std::vector<ui>> candidates_copy(query_vertices_count);
    for (ui i = 0; i < query_vertices_count; ++i) {
        candidates_copy[i].resize(candidates_count[i]);
        std::copy(candidates[i], candidates[i] + candidates_count[i], candidates_copy[i].begin());
    }

    std::vector<int> flag(data_vertices_count, 0);
    std::vector<ui> updated_flag;
    std::vector<double> per_query_vertex_false_positive_ratio(query_vertices_count);
    optimal_candidates_count.resize(query_vertices_count);

    bool is_steady = false;
    while (!is_steady) {
        is_steady = true;
        for (ui i = 0; i < query_vertices_count; ++i) {
            ui u = i;

            ui u_nbr_cnt;
            const ui *u_nbrs = query_graph->getVertexNeighbors(u, u_nbr_cnt);

            ui valid_flag = 0;
            for (ui j = 0; j < u_nbr_cnt; ++j) {
                ui u_nbr = u_nbrs[j];

                for (ui k = 0; k < candidates_count[u_nbr]; ++k) {
                    ui v = candidates_copy[u_nbr][k];

                    if (v == INVALID_VERTEX_ID)
                        continue;

                    ui v_nbr_cnt;
                    const ui *v_nbrs = data_graph->getVertexNeighbors(v, v_nbr_cnt);

                    for (ui l = 0; l < v_nbr_cnt; ++l) {
                        ui v_nbr = v_nbrs[l];

                        if (flag[v_nbr] == valid_flag) {
                            flag[v_nbr] += 1;

                            if (valid_flag == 0) {
                                updated_flag.push_back(v_nbr);
                            }
                        }
                    }
                }
                valid_flag += 1;
            }

            for (ui j = 0; j < candidates_count[u]; ++j) {
                ui v = candidates_copy[u][j];

                if (v == INVALID_VERTEX_ID)
                    continue;

                if (flag[v] != valid_flag) {
                    candidates_copy[u][j] = INVALID_VERTEX_ID;
                    is_steady = false;
                }
            }

            for (auto v : updated_flag) {
                flag[v] = 0;
            }
            updated_flag.clear();
        }
    }

    double sum = 0;
    for (ui i = 0; i < query_vertices_count; ++i) {
        ui u = i;
        ui negative_count = 0;
        for (ui j = 0; j < candidates_count[u]; ++j) {
            ui v = candidates_copy[u][j];

            if (v == INVALID_VERTEX_ID)
                negative_count += 1;
        }

        per_query_vertex_false_positive_ratio[u] =
                (negative_count) / (double) candidates_count[u];
        sum += per_query_vertex_false_positive_ratio[u];
        optimal_candidates_count[u] = candidates_count[u] - negative_count;
    }

    return sum / query_vertices_count;
}



