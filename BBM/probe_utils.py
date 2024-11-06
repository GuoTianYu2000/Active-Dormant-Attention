import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
import os
from omegaconf import OmegaConf
from data import DataArgs, Dataset, iterate_batches, make_dataset
from model import *
import pickle
import numpy as np
from torch.nn import functional as F
from data import *

def plot_attns(cfg, ax, seq_idx, head_idx, layer_idx, seq_len, outputs_list, text):
    attns = outputs_list[layer_idx]['attn_weights'][seq_idx, head_idx, :seq_len, :seq_len].detach().numpy()
    # attns = scores[0, i, :20, :20]
    mask = 1 - np.tril(np.ones_like(attns)) # manually impose causal mask for better-looking plots
    if head_idx == 0:
        yticklabels = text
    else:
        yticklabels = False
    if layer_idx == cfg.model_args['n_layers'] - 1:
        xticklabels = text
    else:
        xticklabels = False
    sns.heatmap(
            attns, mask=mask,
            cmap="coolwarm", xticklabels=xticklabels, yticklabels=yticklabels,
            ax=ax,
        )
    ax.set_title(f"Layer{layer_idx} Head{head_idx}")
    return ax

def get_model_name(bos_num=1, train_steps=4999, delim=0, mix_p=None, n_layers=1, n_heads=1, no_attn_norm=(), no_ffn_norm=(), no_attn=(), no_ffn=(), linear_ffn=(), lr=0.0003, use_simple_model=False, use_vo=False, use_read_out=False, seed=42, **kwargs):
    d_name = float_to_str(delim)
    if mix_p is not None:
        mix_p_name = float_to_str(mix_p, digits=4)
        model_name = f"model_L{n_layers}_H{n_heads}_bos{bos_num}_delim" + d_name + "_mix_p" + mix_p_name
    else:
        model_name = f"model_L{n_layers}_H{n_heads}_bos{bos_num}_delim" + d_name
    model_name = add_missed_structures(model_name, "no_attn_norm", no_attn_norm)
    model_name = add_missed_structures(model_name, "no_ffn_norm", no_ffn_norm)
    model_name = add_missed_structures(model_name, "no_attn", no_attn)
    model_name = add_missed_structures(model_name, "no_ffn", no_ffn)
    model_name = add_missed_structures(model_name, "linear_ffn", linear_ffn)
    if use_simple_model:
        if use_vo:
            model_name = model_name + "_vo"
        if n_layers == 3:
            if use_read_out:
                model_name = model_name + "_ro"
    if lr != 0.0003:
        model_name = add_training_info(model_name, "lr", lr)
    if seed != 42:
        model_name = model_name + f"_seed{seed}"
    return model_name


def add_training_info(model_name, name, value):
    model_name = model_name + name + "_" + float_to_str(value, digits=4)
    return model_name


def add_missed_structures(model_name, struct, layers):
    if layers:
        model_name = model_name + struct
        for layer_idx in layers:
            model_name = model_name + f"_{layer_idx}"
        model_name = model_name + "_"
    return model_name


def load_model(run_path_local="/Users/guotianyu/GitHub/birth/gens/special/bbm", run_path_server="/data/tianyu_guo/birth/gens/special/bbm_2", bos_num=1, train_steps=4999, delim=0, mix_p=None, n_layers=1, n_heads=1, no_attn_norm=(), no_ffn_norm=(), no_attn=(), no_ffn=(), linear_ffn=(), lr=0.0003, use_simple_model=False, use_vo=False, use_read_out=False, seed=42, with_data=True, with_optim=False, model_name=None, data_path_local="/Users/guotianyu/GitHub/birth/data", data_path_server="/data/tianyu_guo/birth/data", seeds=[42, 27], device='cpu', state_name=None):
    if model_name is None:
        model_name = get_model_name(bos_num=bos_num, train_steps=train_steps, delim=delim, mix_p=mix_p, n_layers=n_layers, n_heads=n_heads, no_attn_norm=no_attn_norm, no_ffn_norm=no_ffn_norm, no_attn=no_attn, no_ffn=no_ffn, linear_ffn=linear_ffn, lr=lr, use_simple_model=use_simple_model, use_vo=use_vo, use_read_out=use_read_out, seed=seed)
    try:
        path_local = os.path.join(run_path_local, model_name, "params.yaml")
        cfg = OmegaConf.load(path_local)
    except:
        path_server = os.path.join(run_path_server, model_name, "params.yaml")
        cfg = OmegaConf.load(path_server)

    try:
        getattr(cfg.model_args, 'no_attn_norm')
    except:
        cfg.model_args.no_attn_norm = ()
        cfg.model_args.no_ffn_norm = ()
        cfg.model_args.no_attn = ()
        cfg.model_args.no_ffn = ()
        cfg.model_args.linear_ffn = ()
    try: 
        getattr(cfg.model_args, 'attn_use_relu')
    except:
        cfg.model_args.attn_use_relu = False

    if use_simple_model:
        try:
            getattr(cfg.simple_model_args, 'use_read_out')
        except:
            cfg.simple_model_args.use_read_out = False
        try:
            getattr(cfg.simple_model_args, 'use_scalar')
        except:
            cfg.simple_model_args.use_scalar = []
        model = SimpleModel(cfg.simple_model_args)
    else:
        model = Transformer(cfg.model_args)
    model.to(device)
    # model.eval()
    
    if state_name is None:
        state_name = f"state_{train_steps}.pt"
    else:
        state_name = state_name+f"_{train_steps}.pt"
    try:
        state_path_local = os.path.join(run_path_local, model_name, state_name)
        state = torch.load(state_path_local, map_location=device)
    except:
        state_path_server = os.path.join(run_path_server, model_name, state_name)
        state = torch.load(state_path_server, map_location=device)
    model.load_state_dict(state["model_state_dict"], strict=False, )

    if with_optim:
        optim = torch.optim.AdamW(
                model.parameters(),
                lr=cfg.optim_args.learning_rate,
                weight_decay=cfg.optim_args.weight_decay,
                betas=(0.9, 0.99),
                eps=1e-8)
        optim.load_state_dict(state["optimizer_state_dict"],)
    else:
        optim = None
        
    if not with_data:
        return model, cfg, optim
    else:
        d_name = float_to_str(delim)
        data_name = f"bos{bos_num}_d" + d_name
        try:
            if cfg.data_args.delim_num > 1:
                data_name = data_name + "_delim2"
        except:
            data_name = data_name

        try:
            data_path_local = os.path.join(data_path_local, data_name, "meta.pickle")
            with open(data_path_local, "rb") as f:
                meta_info = pickle.load(f)
        except:
            data_path_server = os.path.join(data_path_server, data_name, "meta.pickle")
            with open(data_path_server, "rb") as f:
                meta_info = pickle.load(f)

        # data_cfg = OmegaConf.structured(meta_info)


        ds = make_dataset(cfg, meta_info)
        x = ds.gen_batch(rng=np.random.default_rng(seeds), batch_size=cfg.optim_args.batch_size)
        x = torch.from_numpy(x)
        # y = torch.from_numpy(y)
        y = x[:, 1:]
        x = x[:, :-1]
        x = x.to(device)
        y = y.to(device)
        return model, cfg, x, y, ds, optim
    
class ModelLoader():
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model_name_from_args = get_model_name(**self.kwargs)

    def __call__(self, with_data=False, with_optim=True, state_name=None):
        self.kwargs['with_data'] = with_data
        self.kwargs['with_optim'] = with_optim
        self.kwargs['state_name'] = state_name
        return load_model(**self.kwargs)
    
    def change_steps(self, train_steps):
        self.kwargs['train_steps'] = train_steps

    def save_dynamic_summary(self, summary):
        save_path = os.path.join(self.kwargs['run_path_server'], self.model_name_from_args, f"dynamic_summary.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(summary, f)
    
    def load_dynamic_summary(self):
        save_path = os.path.join(self.kwargs['run_path_server'], self.model_name_from_args, f"dynamic_summary.pkl")
        with open(save_path, "rb") as f:
            summary = pickle.load(f)
        return summary

def move_device(outputs_list):
    for i, outputs in enumerate(outputs_list):
        for key, value in outputs.items():
            outputs[key] = value.cpu()

    return outputs_list

def summarise_attns(attns, seqs, unique_seqs):
    average_attns = torch.zeros_like(unique_seqs, dtype=attns.dtype).to(attns.device)
    for i, seq in enumerate(unique_seqs):
        mask = (seqs == seq)
        average_attns[i] = torch.nanmean(attns[mask])
    return average_attns

def summarise_logits_difference(full_attns, seqs, unique_seqs):
    B, N, N = full_attns.shape
    average_attns = torch.zeros((B, N), dtype=full_attns.dtype).to(full_attns.device)
    for i in range(N):
        average_attns[:, i] = torch.mean(full_attns[:, i, 1:i], dim=-1)
    non_bos_logits = summarise_attns(average_attns, seqs, unique_seqs)
    bos_logit_differece = full_attns[:, :, 0] - average_attns
    bos_logit_differece = summarise_attns(bos_logit_differece, seqs, unique_seqs)
    return non_bos_logits, bos_logit_differece

def get_triggers(ds, model, hook, cutoff=0.89):
    markov_tok = [i for i in ds.tok_range if i not in ds.idxs and i not in ds.bos and i != ds.delimiter]
    # and i not in (-ds.marginal).argsort()[-15:]
    x_dormant = torch.LongTensor([ds.bos + markov_tok + [i] for i in ds.tok_range]).cuda()
    _, outputs_list_dormant = model.modified_forward_with_hook(x_dormant, hook)
    outputs_list_dormant = move_device(outputs_list_dormant)
    trigger_toks = [i for i in ds.tok_range if outputs_list_dormant[0]['attn_weights'][i, 0, -1, 0]<cutoff]
    attns_to_0 = outputs_list_dormant[0]['attn_weights'][:, 0, -1, 0].detach().cpu().numpy()
    return trigger_toks, attns_to_0, markov_tok

def get_oracle_predicts(x, ds):
    device = x.device
    x = x.cpu()
    B, N, V = x.shape[0], x.shape[1], ds.num_tokens
    predicts_oracle = torch.zeros((B, N, V))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            idx = x[i, j].item()
            if idx in ds.idxs:
                predicts_oracle[i, j, x[i, j-1].item()] = 1
            else:
                predicts_oracle[i, j, :] = torch.from_numpy(ds.cond[idx])
    return predicts_oracle.to(device)

def get_risk(probs, predicts, predict_in_logits, triggers_pos):
    if predict_in_logits:
        predicts = torch.nn.functional.softmax(predicts, dim=-1)
    loss = - torch.log(predicts)
    loss[torch.where(probs == 0)] = 0
    risk = torch.einsum("ikj,ikj->ik", probs, loss)
    risk_icl = risk[triggers_pos].mean()
    risk_markov = risk[~triggers_pos].mean()
    return risk, risk_icl, risk_markov

def get_dynamic_summary(ds, x, y, model_loader, hook_dict, keys, probs=None, triggers_pos=None, ):
    summary = {}
    if probs is None:
        probs = get_oracle_predicts(x, ds)
    if triggers_pos is None:
        triggers_pos = ds.get_triggers_pos(x)
    for hook_name, hook in hook_dict.items():
        model, cfg, optimizer = model_loader()
        attn_layers = list(set(range(len(model.layers))) - set(cfg.model_args.no_attn))
        optimizer.zero_grad()
        pred, outputs_list = model.modified_forward_with_hook(x, hook)
        if "norm_influence" in keys:
            grad_weight= torch.autograd.grad(outputs=outputs_list[0]['output'][0, 0, :].norm(), inputs=model.parameters(), create_graph=True, allow_unused=True)
            norm_grads = {}
            for (name, param), grad in zip(model.named_parameters(), grad_weight):
                if grad is not None:
                    norm_grads[name] = grad
        loss = F.cross_entropy(pred.flatten(0, 1), y.flatten(0, 1))
        loss.backward()
        grads, params, summary[hook_name] = {}, {}, {}
        for name, param in model.named_parameters():
            if param is not None:
                params[name] = param.detach().clone()
            if param.grad is not None:
                grads[name] = param.grad.detach().clone()
        if "adam_fr" in keys or "adam_l2" in keys:
            optimizer.step()
            update = {}
            for name, param in model.named_parameters():
                if param is not None:
                    update[name] = param.detach().clone() - params[name]
        _, icl_risk, markov_risk = get_risk(probs, pred, predict_in_logits=True, triggers_pos=triggers_pos)
        for key in keys:
            if key == "icl_risk":
                summary[hook_name][key] = icl_risk.item()
            elif key == "markov_risk":
                summary[hook_name][key] = markov_risk.item()
            elif key == "bos_attn":
                summary[hook_name][key] = [outputs_list[idx]['attn_weights'][:, 0, :, 0][~triggers_pos].mean().item() for idx in attn_layers]
            elif key == "output_norm":
                summary[hook_name][key] = [outputs_list[idx]['output'][0, 0, :].detach().norm(dim=-1).cpu().item() for idx in range(len(model.layers))]
            elif key == "value_norm":
                summary[hook_name][key] = [outputs_list[idx]['value_states'][0, 0, 0, :].detach().norm(dim=-1).cpu().item() for idx in attn_layers]
            elif key == "output_state":
                summary[hook_name][key] = [outputs_list[idx]['output'][0, 0, :].detach().cpu().tolist() for idx in range(len(model.layers))]
            elif key == "value_state":
                summary[hook_name][key] = [outputs_list[idx]['value_states'][0, 0, 0, :].detach().cpu().tolist() for idx in attn_layers]
            elif key == "attn_logits":
                summary[hook_name][key] = [outputs_list[idx]['attn_logits'][0, 0, :, 0].detach().cpu().tolist() for idx in attn_layers]
            elif key == "grads_fr":
                grads_fr = dict([(n, k.norm().item()) for n, k in grads.items()])
                summary[hook_name][key] = grads_fr
            elif key == "grads_l2":
                grads_l2 = dict([(n, torch.linalg.norm(k, ord=2).item()) for n, k in grads.items()])
                summary[hook_name][key] = grads_l2
            elif key == "adam_fr":
                udpate_fr = dict([(n, k.norm().item()) for n, k in update.items()])
                summary[hook_name][key] = udpate_fr
            elif key == "adam_l2":
                udpate_l2 = dict([(n, torch.linalg.norm(k, ord=2).item()) for n, k in update.items()])
                summary[hook_name][key] = udpate_l2
            elif key == "norm_influence":
                norm_influence = dict([(n, (grad * update[n]).sum().cpu().detach().item()) for n, grad in norm_grads.items()])
                norm_influence_grad = dict([(n, ((grad * update[n]).sum().cpu().detach()/update[n].norm().cpu().detach()).item()) for n, grad in norm_grads.items()])
                norm_influence_update = dict([(n, ((grad * update[n]).sum().cpu().detach()/grad.norm().cpu().detach()).item()) for n, grad in norm_grads.items()])
                summary[hook_name][key] = norm_influence
                summary[hook_name][key + "_grad"] = norm_influence_grad
                summary[hook_name][key + "_update"] = norm_influence_update
            elif key == "norm_influence_taylor":
                norm_influence_taylor = dict([(n, (grad * grads[n]).sum().cpu().detach().item()) for n, grad in norm_grads.items()])
                summary[hook_name][key] = norm_influence_taylor
            elif key == "dnorm_norm":
                dnorm_norm = dict([(n, grad.norm().cpu().detach().item()) for n, grad in norm_grads.items()])
                summary[hook_name][key] = dnorm_norm
    return summary

def concat_summary(summary, hook_dict, keys=["icl_risk", "markov_risk", "bos_attn", "output_norm", "value_norm"], ):
    summary = dict(sorted(summary.items(), key=lambda x: x[0]))
    key_in_dict = ["grads_fr", "grads_l2", "adam_fr", "adam_l2", "norm_influence", "norm_influence_grad", "norm_influence_update"]
    summary_sample = summary[list(summary.keys())[0]]
    summary_concat = dict([(k, dict([(k1, dict([(k2, []) for k2 in summary_sample[k][k1].keys()])) if k1 in key_in_dict else (k1, []) for k1 in keys])) for k in hook_dict.keys()])
    step_list = []
    for step, step_summary in summary.items():
        step_list.append(step)
        for hook_name in hook_dict.keys():
            for key in keys:
                if key in key_in_dict:
                    for k in summary_concat[hook_name][key].keys():
                        summary_concat[hook_name][key][k].append(step_summary[hook_name][key][k])
                else:
                    summary_concat[hook_name][key].append(step_summary[hook_name][key])
    for keys, values in summary_concat.items():
        for key, value in values.items():
            if key in key_in_dict:
                for k, v in value.items():
                    summary_concat[keys][key][k] = np.array(v)
            else:
                summary_concat[keys][key] = np.array(value)
    return step_list, summary_concat



def get_risk_by_token(ds, risk, triggers_pos, x, ):
    risk_by_token = dict([(idx, []) for idx in ds.tok_range])
    for seq_idx in range(256):
        ids = x[seq_idx, [idx-1 for idx, f in enumerate(triggers_pos[seq_idx, :]) if f]].tolist()
        risks = risk[0][seq_idx, [idx for idx, f in enumerate(triggers_pos[seq_idx, :]) if f]].tolist()
        for id, r in zip(ids, risks):
            risk_by_token[id].append(r)
    for idx, r in risk_by_token.items():
        if len(r)>0:
            risk_by_token[idx] = np.mean(risk_by_token[idx])
        else:
            risk_by_token[idx] = 0
    return risk_by_token
