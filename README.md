# MendMKG

This is the source code for MendMKG. 

## Running MendMKG

**STEP1: Environment requirement**

Install python 2.7, ptorch 1.1, PyG

**STEP2: EHR data preparation**

We used two EHR datasets for expriments in paper. 
1. MIMIC-III: a publicly available dataset whih requires a registration for usage (https://mimic.physionet.org/gettingstarted/access/)
An example of training/test samples is provided in /data ("train_data_100_cases.csv" and "test_data_100_cases.csv" with prepdiction label "data_label.csv".)

2. CLAIM: no permission to share the internal secret dataset. 

**STEP3: Medical knowledge graph preparation**

The medical knowledge graph used in our experiment: "/data/graph_edge_coo.npz", and the dictionary "/data/graph_nodes_dict.pkl". 

**STEP4: Self-supervised learning strategy**

Perform "code/run_pretrain.py" for the self-supervised learning strategy to achieve a mutual enhancement between knowledge graph and EHR data.

**STEP5: Disease prediction**

Perform "code/run_model.py" for the disease prediction through a fine-tuning. 
