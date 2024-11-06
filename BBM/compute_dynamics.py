from dataclasses import dataclass
import itertools
import logging
import random
import math
import numpy as np
import json
import pickle
import time
import torch
import sys
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
from probe_utils import *
from plot_utils import *
from tqdm import tqdm

from omegaconf import OmegaConf
from pathlib import Path
from torch import nn, Tensor
from torch.nn import functional as F
from typing import List, Optional, Tuple
import os
# os.chdir("/data/tianyu_guo/birth")
from data import DataArgs, Dataset, iterate_batches, make_dataset
from ihead_full_model import *


def compute_dynamic_traj(run_path_server, no_attn=(),):
    model_loader = ModelLoader(run_path_local="/Users/guotianyu/GitHub/birth/gens/special/markov", run_path_server=run_path_server, bos_num=1, train_steps=10000, delim=0, n_layers=3, n_heads=1, no_attn_norm=(), no_ffn_norm=(), no_attn=no_attn, no_ffn=(), linear_ffn=(), with_data=True, with_optim=True, data_path_local="/Users/guotianyu/GitHub/birth/data", data_path_server="~/data", device=device)
    model, cfg, x, y, ds, optim = model_loader(with_data=True)
    triggers_pos = ds.get_triggers_pos(x.to('cpu'))
    hook_dict = {"basic": forward_hook([], ''), "no_attn_0": check_embed(target_layers=[0, 1, 2], target_heads=[(0, 0)], target_mlp_layers=[]), "no_mlp_0": check_embed(target_layers=[0, 1, 2], target_heads=[], target_mlp_layers=[0]), "clean_attn": clean_attn(list(set([1, 2]) - set(no_attn)), torch.from_numpy(triggers_pos))}
    keys = ["icl_risk", "markov_risk", "bos_attn", "output_norm", "value_norm", "output_state", "value_state", "attn_logits", "grads_fr", "grads_l2", "adam_fr", "adam_l2", "norm_influence"]
    grads, params, updates, norm_grads = {}, {}, {}, {}
    optim.zero_grad()
    pred, outputs_list = model.modified_forward_with_hook(x, hook_dict['basic'])

    probs = get_oracle_predicts(x, ds)

    summary_dynamic_traj = {}
    if cfg.max_iters == 100000:
        steps_list = torch.arange(0, 10000, 20).tolist() + torch.arange(10000, 100001, 200).tolist()
    else:
        steps_list = torch.arange(0, 10000, 20).tolist()
    for step in tqdm(steps_list):
        model_loader.change_steps(step)
        summary_dynamic_traj[step] = get_dynamic_summary(ds, x, y, model_loader, hook_dict, keys, probs, triggers_pos)
    model_loader.save_dynamic_summary(summary_dynamic_traj)
    



if __name__ == "__main__":
    torch.cuda.set_device(2)
    device = 'cuda:2'
    run_path_list = ["~/gens/pre-iclr/dynamics/bbm_long_train_redo", "~/gens/pre-iclr/dynamics/bbm_long_train", "~/gens/pre-iclr/dynamics/bbm_simplified"]
    no_attn_list = [( ), ( ), (2, )]
    for run_path, no_attn in zip(run_path_list, no_attn_list):
        compute_dynamic_traj(run_path, no_attn)



