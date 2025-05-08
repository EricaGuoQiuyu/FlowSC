# FlowSC
Efficient and Accurate Subgraph Counting: A Bottom-up Flow-learning based Approach [VLDB 2025]

## Dependencies

python==3.11.7

networkx==3.3

numpy==2.1.3

scipy==1.14.1

torch==2.1.0

## Quick Start

```shell
python main.py
```

We provide the WordNet dataset, its groundtruth, and the model's initialization parameters. You can use `main.py` to test the 5 query graph sets in WordNet directly. Modify the `query_size` in `main.py` to test different query sets.

## Filtering Test

In the `filter` folder, `build1` provides the BipartitePlus filter used in our paper, which has been complied and is ready for direct use. The `SubgraphMatching` folder contains the source code of BipartitePlus, developed based on [SubgraphMatching](https://github.com/RapidsAtHKUST/SubgraphMatching/tree/master). We have  commented out the activation and termination conditions of Bipartite, and set the first iteration of refinement to use `GQL`, with the remaining refinements using BipattitePlus for a total of 5 refinement rounds. This setup makes it convenient for you to independently test the filtering performance of BipartitePlus. You can easily adjust the activation and termination conditions of BipartitePlus as needed. To compile SubgraphMatching:

```
cd filter/SubgraphMatching
mkdir build
cmake ..
make
```

Execute SubgraphMatching (5 refinements version without activation and termination threshold):

```
cd filter/SubgraphMatching/build/matching
./SubgraphMatching.out -d path_to_data_graph -q path_to_query_graph

```

You can use `filtering_test.py` to reproduce the filtering experiments in our paper. Since filtering tests do not require groundtruth, the filtering experiments use the complete datasets from [SubgraphMatching](https://github.com/RapidsAtHKUST/SubgraphMatching/tree/master), with each dataset containing 1,800 query graphs. Update `datasets_path` to point to your datasets path.

## Pretrain

We provide our pertaining code and the model initialization parameters used in our paper. If you want to test your own datasets, you can simply use:

```
python pretrain.py
```

to generate your own initialization parameters. We do not recommend directly using our initialization parameters for your dataset because the pretraining dataset and the evaluation dataset should not overlap. You can split your own pretraining dataset and generate your own pretraining parameters.  Update `--root_path` to point to your pretraining dataset, modify `--pretrain_groundtruth_path` to the corresponding groundtruth, and update the path to the pretrained parameters in `main.py`.

## Datasets

The datasets used in our paper is from [SubgraphMatching](https://github.com/RapidsAtHKUST/SubgraphMatching/tree/master). The exact count for each query graph is generated using the recommended method in [SubgraphMatching](https://github.com/RapidsAtHKUST/SubgraphMatching/tree/master).