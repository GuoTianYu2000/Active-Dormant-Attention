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

from omegaconf import OmegaConf
from pathlib import Path
from torch import nn, Tensor
from torch.nn import functional as F
from typing import List, Optional, Tuple
import os
# os.chdir("/data/tianyu_guo/birth")
from data import DataArgs, Dataset, iterate_batches, make_dataset
from ihead_full_model import ModelArgs, Transformer, forward_hook, test_value, test_sink

fn = "/data/tianyu_guo/birth/figures"

run_path_server = "/data/tianyu_guo/birth/gens/pre_final/bbm"
# run_path_server2="/data/tianyu_guo/birth/gens/special/bbm_2"
model, cfg, x, y, ds = load_model(run_path_local="/Users/guotianyu/GitHub/birth/gens/special/markov", run_path_server=run_path_server, n_layers=3, n_heads=1, bos_num=1, train_steps=9980, delim=0, with_data=True, data_path_local="/Users/guotianyu/GitHub/birth/data", data_path_server="/data/tianyu_guo/birth/data")
hook = forward_hook(target_layers=[], target_name="")
predicts, outputs_list = model.modified_forward_with_hook(x, hook)
model.cuda()
trigger_toks, attns_to_0, markov_tok = get_triggers(ds, model, hook, cutoff=0.89)

# outputs_list_steps = {}
norms_list = {}
steps_list = []
n_layers, n_heads, bos_num = 3, 1, 1
# f"/data/tianyu_guo/birth/gens/special/bbm_2/model_L{n_layers}_H{n_heads}_bos{bos_num}_delim0/state_{train_steps}.pt"
dir_name = f"/data/tianyu_guo/birth/gens/pre_final/bbm/model_L{n_layers}_H{n_heads}_bos{bos_num}_delim0"
model = model.cuda()
x = x.cuda()
y = y.cuda()
x_cpu = x.cpu()
i = 0
sub_steps = np.arange(100, 1020, 40).tolist()
with torch.no_grad():
    for fn in os.listdir(dir_name):
        if 'state' not in fn:
            continue
        step = int(fn.split('.')[0].split('_')[1])
        steps_list.append(step)
        state = torch.load(os.path.join(dir_name, fn), map_location="cuda")
        model.load_state_dict(state["model_state_dict"], strict=False, )
        trigger_toks, attns_to_0, markov_tok = get_triggers(ds, model, hook, cutoff=0.89)
        predicts, outputs_list = model.modified_forward_with_hook(x, hook)
        triggers_pos = ds.get_triggers_pos(x_cpu)
        trigger_pos = torch.from_numpy(triggers_pos).cuda()
        loss = F.cross_entropy(predicts.flatten(0, 1), y.flatten(0, 1))
        icl_loss = F.cross_entropy(predicts[triggers_pos, :], y[triggers_pos])
        markov_loss = F.cross_entropy(predicts[~triggers_pos, :], y[~triggers_pos])
        outputs_list = move_device(outputs_list)
        all_norms = torch.zeros(3, 512, 256)
        # pdb.set_trace()
        for layer_idx in range(3):
            all_norms[layer_idx, :, :] = outputs_list[layer_idx]['output'][:, :, :].norm(dim=-1).cpu().detach()
        norms_list[step] = [all_norms[layer_idx, :, 0].mean() for layer_idx in range(3)]
        i += 1
        print(i)
steps_list.sort()

results = {"steps_list":steps_list, "norms_list":norms_list}

with open("/data/tianyu_guo/birth/gens/final/dynamics_massive.pkl", "wb") as f:
    pickle.dump(results, f)

