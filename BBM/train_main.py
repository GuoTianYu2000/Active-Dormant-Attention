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
import os
import pdb
import wandb

from omegaconf import OmegaConf
from pathlib import Path
from torch import nn, Tensor
from torch.nn import functional as F
from typing import List, Optional, Tuple

from data import *
from model import *
from tqdm import tqdm
from probe_utils import get_model_name

logging.getLogger().setLevel(logging.INFO)

def set_random_seed(seed):
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed) # RZ: Sets the seed for generating random numbers for all devices (both CPU and CUDA).
    torch.manual_seed(seed) # RZ: Specifically sets the seed for all CUDA GPUs for generating random numbers. This is necessary for ensuring that all GPUs have the same initial random state.
    random.seed(seed)
    torch.backends.cudnn.deterministic = True # RZ: Some operations in CUDA are non-deterministic by default. To ensure determinism, you might need to set additional flags in PyTorch. However, this could potentially impact performance.
    torch.backends.cudnn.benchmark = False


@dataclass
class OptimArgs:
    learning_rate: float = 0.01
    weight_decay: float = 0.01
    momentum: float = 0.9  # for SGD
    batch_size: int = 64
    use_sgd: bool = False  # otherwise use AdamW
    seperate_lr: bool = False
    massiv_lr: float = 0.0003
    copy_lr: float = 0.03

@dataclass
class WandbArgs:
    project: str = 'dormant'
    entity: str = 'tianyu_guo' # change to your entity name
    name: str = 'bbm'

@dataclass
class TrainerArgs:
    optim_args: OptimArgs
    data_args: DataArgs
    model_args: ModelArgs
    simple_model_args: SimpleModelArgs
    wandb_args: WandbArgs
    use_simple_model: Optional[bool] = False
    max_iters: Optional[int] = None
    eval_delta: int = 5
    log_norms: bool = False
    log_probes: bool = False
    num_data_workers: int = 24
    save_dir: Optional[str] = None
    fine_grid_log: int = 0
    root_dir: str = ''
    task_name: str = ''
    seperate_loss: bool = False
    device_num: int = 2
    seed: int = 42


if __name__ == '__main__':
    args = TrainerArgs(
           optim_args=OptimArgs(),
           data_args=DataArgs(),
           model_args=ModelArgs(),
           wandb_args=WandbArgs(),
           simple_model_args=SimpleModelArgs(),
        )
    cli_args = OmegaConf.from_cli()
    cfg = OmegaConf.merge(OmegaConf.structured(args), cli_args)
    # cfg.model_args.bos_num = cfg.data_args.bos_num
    set_random_seed(cfg.seed)
    torch.cuda.set_device(cfg.device_num)
    d_name = float_to_str(cfg.data_args.delimiter_p)
    meta_path = f"~/data/bos{cfg.data_args.bos_num}_d" + d_name +"/meta.pickle" if cfg.data_args.delim_num == 1 else f"~/data/bos{cfg.data_args.bos_num}_d" + d_name + "_delim2" +"/meta.pickle"
    with open(meta_path, "rb") as f:
        meta_info = pickle.load(f)
    model_name = get_model_name(bos_num=cfg.data_args.bos_num, train_steps=cfg.max_iters, delim=cfg.data_args.delimiter_p, mix_p=cfg.data_args.mix_p, lr=cfg.optim_args.learning_rate, use_simple_model=cfg.use_simple_model, seed=cfg.seed, **(cfg.model_args if not cfg.use_simple_model else cfg.simple_model_args))
    cfg.save_dir = os.path.join(cfg.save_dir, model_name)
    # pdb.set_trace()
    ds = make_dataset(cfg, meta_info)
    idxs_in_torch = torch.Tensor(ds.idxs).cuda()
    cfg.model_args.vocab_size = ds.num_tokens  
    cfg.simple_model_args.vocab_size = ds.num_tokens   
    if cfg.save_dir is not None:
        outdir = Path(cfg.root_dir) / Path(cfg.save_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        # save params
        dict_cfg = dict(cfg)
        print(dict_cfg)
        with open(outdir / 'params.yaml', 'w') as f:
            OmegaConf.save(dict_cfg, f,)
    # wandb.init()
    wandb.init(
            dir=str(outdir),
            project=cfg.wandb_args.project,
            entity=cfg.wandb_args.entity,
            name=cfg.wandb_args.name,
        )
    model = Transformer(cfg.model_args) if not cfg.use_simple_model else SimpleModel(cfg.simple_model_args)
    model.cuda()

    # optim
    if cfg.optim_args.use_sgd:
        optimizer = torch.optim.SGD(model.parameters(),
                lr=cfg.optim_args.learning_rate,
                weight_decay=cfg.optim_args.weight_decay,
                momentum=cfg.optim_args.momentum)
    else:
        if cfg.use_simple_model and cfg.optim_args.seperate_lr:
            params_except_copy = [param for name, param in model.named_parameters() if 'layers.1' not in name]
            params_copy = [param for name, param in model.named_parameters() if 'layers.1' in name]
            optimizer = torch.optim.AdamW(
                [
                    {"params": params_except_copy, "lr": cfg.optim_args.massiv_lr, "weight_decay": cfg.optim_args.weight_decay,
                    "betas": (0.9, 0.99), "eps": 1e-8},
                    {"params": params_copy, "lr": cfg.optim_args.copy_lr, "weight_decay": cfg.optim_args.weight_decay,
                    "betas": (0.9, 0.99), "eps": 1e-8},
                ]
            )
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg.optim_args.learning_rate,
                weight_decay=cfg.optim_args.weight_decay,
                betas=(0.9, 0.99),
                eps=1e-8)
        
    # pdb.set_trace()
    # a test batch for experimentation
    x_exp = ds.gen_batch(np.random.default_rng(0), 128)
    x_exp = x_exp[:,:ds.seq_length]


    if not cfg.use_simple_model:
        if cfg.fine_grid_log == 0:
            log_steps = np.linspace(cfg.fine_grid_log, cfg.max_iters, 5).tolist()
        elif cfg.fine_grid_log < cfg.max_iters:
            log_steps = np.arange(0, cfg.fine_grid_log, 5).tolist() + np.linspace(cfg.fine_grid_log, cfg.max_iters, 5).tolist()
        else:
            log_steps = np.arange(0, 10000, 20).tolist()
            log_steps = log_steps + np.arange(10000, cfg.max_iters+1, 200).tolist()
    else:
        log_steps = np.arange(0, 1000, 5).tolist()
        log_steps = log_steps + np.arange(1000, cfg.max_iters+1, 200).tolist()
    
    for i, (x, y) in enumerate(iterate_batches(ds, batch_size=cfg.optim_args.batch_size, num_workers=cfg.num_data_workers)):
        if i in log_steps:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, os.path.join(cfg.save_dir, f"state_{i}.pt"))
        if cfg.max_iters is not None and i >= cfg.max_iters:
            sys.exit(0)
        x = torch.from_numpy(x).cuda()
        y = torch.from_numpy(y).cuda()
        triggers_pos = torch.isin(x, idxs_in_torch)
        exist_icl = torch.sum(triggers_pos) > 0

        optimizer.zero_grad()
        if cfg.use_simple_model:
            pred = model(x, triggers_pos)
        else:
            pred = model(x)

        loss = F.cross_entropy(pred.flatten(0, 1), y.flatten(0, 1))
        loss.backward()

        optimizer.step()
        if cfg.seperate_loss:
            icl_loss = F.cross_entropy(pred[triggers_pos, :], y[triggers_pos]) if exist_icl else 0
            markov_loss = F.cross_entropy(pred[~triggers_pos, :], y[~triggers_pos])
            log_dict = {
                            f"{cfg.task_name}/overall_loss": loss,
                            f"{cfg.task_name}/markov_loss": markov_loss,
                            f"{cfg.task_name}/icl_loss": icl_loss,
                        }
            if cfg.use_simple_model:
                layer = model.layers[0] if cfg.simple_model_args.n_layers == 1 else model.layers[1]
                attn_on_bos = layer.attention.qk_bos.detach().clone().cpu().tolist()
                value_state_norm = layer.attention.wv_bos.detach().clone().norm().item()
                log_dict["value_state_norm"] = value_state_norm
                for idx, value in enumerate(attn_on_bos[:10]):
                    log_dict[f'attn_on_bos_{idx}'] = value
            wandb.log(log_dict, step=i, )
        else:
            wandb.log({f"{cfg.task_name}/overall_loss": loss}, step=i, )

    # save the last state
    training_state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_step": i+1,
    }
    torch.save(training_state, os.path.join(cfg.save_dir, f"state_{i+1}.pt"))
