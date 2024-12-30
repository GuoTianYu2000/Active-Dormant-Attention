# Usage examples

The main script is `train_main.py`, which trains on the BBM task with a generic Transformer, with arbitrary number of layers, including MLP feed-forward layers and layer-normalization.

The arguments can be provided in the command line as in the following example:
```
  python train_main.py max_iters=10000 eval_delta=5  task_name=bbm fine_grid_log=1000 seperate_loss=True seed=42\
  model_args.n_layers=1 model_args.n_heads=1 model_args.dim=256\
  data_args.k=3 data_args.fixed_special_toks=True data_args.bos_num=1 data_args.delimiter_p=0 data_args.delim_num=1\
  wandb_args.name=bbm wandb_args.entity=tianyu_guo\
  optim_args.use_sgd=False optim_args.learning_rate=0.0003 optim_args.weight_decay=1e-4 optim_args.batch_size=512 \
  save_dir=./gens/final/bbm_k3
```

Some comments on the above command line arguments:
- `max_iters` is the total number of training steps
- `task_name` indicates the data generating procedure, by default it is `bbm`
- `fine_grid_log` gives the phase boundary for the training logs. Under `fine_grid_log`, the model checkpoint is saved per 5 training steps. Above `fine_grid_log`, the model checkpoint is saved 5 times
- `data_args.k` is the number of trigger tokens in BBM
- `data_args.fixed_special_toks=True` indicates fixed triggers, chosen as the most frequent tokens (should be `True` by default, this argument is inherited from the previous codebase but is now outdated and should be revised or deprecated)
- `data_args.bos_num` is the number of beginning-of-sequence tokens
- `data_args.delimiter_p` is the probability of generating delimiter tokens (should be 0 by default, this argument is used for variations of the `BBM` task)
- `wandb_args.entity` is the entity name for using wandb

For more command line arguments, you can take a look at the classes `TrainerArgs` (arguments with no prefix), `OptimArgs`, `DataArgs` (in `data.py`) and `ModelArgs` (in `train_main.py`).



