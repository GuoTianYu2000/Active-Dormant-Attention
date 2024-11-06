#!/bin/bash

conda init bash
source activate nanogpt

# Assuming the Python file is named 'script.py' and accepts parameters
# params=(0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79)

# dormant copy task
layer_values=(1 3)
k_values=(3 5)
seed_list=(20 21 22 23 24 25 26 27 28 29)
mix_p_list=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
iter_num=10000

for layer_idx in "${layer_values[@]}"
do
for k in "${k_values[@]}"
do
  python ihead_full_main.py max_iters=${iter_num} log_probes=True eval_delta=5  task_name=dormant_copy fine_grid_log=${iter_num} seperate_loss=True seed=42\
  model_args.n_layers=${layer_idx} model_args.n_heads=1 model_args.dim=256\
  data_args.k=${k} data_args.fixed_special_toks=True data_args.bos_num=1 data_args.delimiter_p=0 data_args.delim_num=1\
  wandb_args.name=dormant_copy\
  optim_args.use_sgd=False optim_args.learning_rate=0.0003 optim_args.weight_decay=1e-4 optim_args.batch_size=512 \
  save_dir=/data/tianyu/birth/gens/final/dormant_copy_k${k}
done
done

# ablation 1: Biette's task
for layer_idx in "${layer_values[@]}"
do
  python ihead_full_main.py max_iters=${iter_num} log_probes=True eval_delta=5  task_name=default fine_grid_log=0 seperate_loss=True seed=42\
  model_args.n_layers=${layer_idx} model_args.n_heads=1 model_args.dim=256\
  data_args.k=3 data_args.fixed_special_toks=True data_args.bos_num=1 data_args.delimiter_p=0 data_args.delim_num=1\
  wandb_args.name=Biette\
  optim_args.use_sgd=False optim_args.learning_rate=0.0003 optim_args.weight_decay=1e-4 optim_args.batch_size=512 \
  save_dir=/data/tianyu/birth/gens/final/Biette
done

# ablation 2: Dormant Markov
for layer_idx in "${layer_values[@]}"
do
  python ihead_full_main.py max_iters=${iter_num} log_probes=True eval_delta=5  task_name=dormant_markov fine_grid_log=0 seperate_loss=True seed=42\
  model_args.n_layers=${layer_idx} model_args.n_heads=1 model_args.dim=256\
  data_args.k=3 data_args.fixed_special_toks=True data_args.bos_num=1 data_args.delimiter_p=0 data_args.delim_num=1\
  wandb_args.name=dormant_markov\
  optim_args.use_sgd=False optim_args.learning_rate=0.0003 optim_args.weight_decay=1e-4 optim_args.batch_size=512 \
  save_dir=/data/tianyu/birth/gens/final/dormant_markov
done

# ablation 3: Markov
for layer_idx in "${layer_values[@]}"
do
  python ihead_full_main.py max_iters=${iter_num} log_probes=True eval_delta=5  task_name=markov fine_grid_log=0 seed=42\
  model_args.n_layers=${layer_idx} model_args.n_heads=1 model_args.dim=256\
  data_args.k=3 data_args.fixed_special_toks=True data_args.bos_num=1 data_args.delimiter_p=0 data_args.delim_num=1\
  wandb_args.name=markov\
  optim_args.use_sgd=False optim_args.learning_rate=0.0003 optim_args.weight_decay=1e-4 optim_args.batch_size=512 \
  save_dir=/data/tianyu/birth/gens/final/markov
done

# ablation 2: change of context delimiter
for layer_idx in "${layer_values[@]}"
do
  python ihead_full_main.py max_iters=${iter_num} log_probes=True eval_delta=5  task_name=dormant_double_tasks fine_grid_log=0 seed=42\
  model_args.n_layers=${layer_idx} model_args.n_heads=1 model_args.dim=256\
  data_args.k=3 data_args.fixed_special_toks=True data_args.bos_num=1 data_args.delimiter_p=0 data_args.delim_num=1\
  wandb_args.name=dormant_double_tasks\
  optim_args.use_sgd=False optim_args.learning_rate=0.0003 optim_args.weight_decay=1e-4 optim_args.batch_size=512 \
  save_dir=/data/tianyu/birth/gens/final/dormant_double_tasks
done
