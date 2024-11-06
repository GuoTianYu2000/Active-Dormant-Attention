#!/bin/bash

conda init bash
source activate nanogpt

# Assuming the Python file is named 'script.py' and accepts parameters
# params=(0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79)

# dormant copy task
layer_idx=1
head_idx=1
k=3
seed_list=(20 21 22 23 24 25 26 27 28 29)
mix_p_list=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
iter_num=5000

# The first try
    # python ihead_full_main.py max_iters=${iter_num} log_probes=True eval_delta=5  task_name=dormant_copy fine_grid_log=${iter_num} seperate_loss=True seed=42\
    # use_simple_model=True\
    # data_args.k=${k} data_args.fixed_special_toks=True data_args.bos_num=1 data_args.delimiter_p=0 data_args.delim_num=1\
    # wandb_args.name=simple_model\
    # optim_args.use_sgd=False optim_args.learning_rate=0.03 optim_args.weight_decay=1e-4 optim_args.batch_size=512 \
    # save_dir=/data/tianyu/birth/gens/pre-iclr/dynamics/Simplemodel
# to have diminishing value states, add O matrix in the structure
    python ihead_full_main.py max_iters=${iter_num} log_probes=True eval_delta=5  task_name=dormant_copy fine_grid_log=${iter_num} seperate_loss=True seed=42\
    use_simple_model=True\
    data_args.k=${k} data_args.fixed_special_toks=True data_args.bos_num=1 data_args.delimiter_p=0 data_args.delim_num=1\
    wandb_args.name=simple_model\
    optim_args.use_sgd=False optim_args.learning_rate=0.03 optim_args.weight_decay=1e-4 optim_args.batch_size=512 \
    save_dir=/data/tianyu/birth/gens/pre-iclr/dynamics/Simplemodel

