import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from transformers import OlmoForCausalLM, AutoTokenizer, GPTNeoXForCausalLM, AutoModelForCausalLM
from src.to_numpy import to_numpy
from src.model_wrapper import ModelWrapper
from src.hook import Hook, RecordHook, ZeroOutAttentionHeadHook, CompositeHook
from src.model_wrapper import ModelWrapper
torch.set_printoptions(sci_mode=False)
device = "cuda:1"
model_name_olmo = "allenai/OLMo-7B-0424-hf"

revisions_olmo = [
    (0, "step0-tokens0B"),
    (500, "step500-tokens2B"),
    (1000, "step1000-tokens4B"),
    (1500, "step1500-tokens6B"),
    (2000, "step2000-tokens8B"),
    (2500, "step2500-tokens10B"),
    (3000, "step3000-tokens12B"),
    (3500, "step3500-tokens14B"),
    (4000, "step4000-tokens16B"),
    (4500, "step4500-tokens18B"),
    (5000, "step5000-tokens20B"),
    (5500, "step5500-tokens23B"),
    (6000, "step6000-tokens25B"),
    (6500, "step6500-tokens27B"),
    (7000, "step7000-tokens29B"),
    (7500, "step7500-tokens31B"),
    (8000, "step8000-tokens33B"),
    (8500, "step8500-tokens35B"),
    (9000, "step9000-tokens37B"),
    (9500, "step9500-tokens39B"),
    (10000, "step10000-tokens41B"),
    (25000, "step25000-tokens104B"),
    (50000, "step50000-tokens209B"),
    (100000, "step100000-tokens419B"),
    (147500, "step147500-tokens618B"),
    (200000, "step200000-tokens838B"),
    (250000, "step250000-tokens1048B"),
    (300000, "step300000-tokens1257B"),
    (350000, "step350000-tokens1467B"),
    (400000, "step410000-tokens1719B"),
    (450000, "step449000-tokens1882B")
]

def load_olmo_model(revision):
    model = AutoModelForCausalLM.from_pretrained(model_name_olmo, revision=revision[1], attn_implementation="eager", trust_remote_code=True, device_map=device)
    model.eval()
    model.requires_grad_(False)
    return ModelWrapper(model)

tokenizer = AutoTokenizer.from_pretrained(model_name_olmo, trust_remote_code=True)




def collect_statistics_at_layer(model_wrapper, input_tokens, layer_probe):
    layer_input_prenorm_hook = RecordHook(target_name="layer_input_prenorm", record_buffer=[], target_layers=[layer_probe])
    layer_input_postnorm_hook = RecordHook(target_name="layer_input_postnorm", record_buffer=[], target_layers=[layer_probe])
    q_hook = RecordHook(target_name="q_proj", record_buffer=[], target_layers=[layer_probe])
    k_hook = RecordHook(target_name="k_proj", record_buffer=[], target_layers=[layer_probe])
    v_hook = RecordHook(target_name="v_proj", record_buffer=[], target_layers=[layer_probe])
    q_postrope_hook = RecordHook(target_name="q_proj_postrope", record_buffer=[], target_layers=[layer_probe])
    k_postrope_hook = RecordHook(target_name="k_proj_postrope", record_buffer=[], target_layers=[layer_probe])
    v_postrope_hook = RecordHook(target_name="v_proj_postrope", record_buffer=[], target_layers=[layer_probe])
    attn_logits_hook = RecordHook(target_name="attn_logits", record_buffer=[], target_layers=[layer_probe])
    attn_weights_hook = RecordHook(target_name="attn_weights", record_buffer=[], target_layers=[layer_probe])
    attn_softmax_qkt_v_hook = RecordHook(target_name="attn_softmax_qkt_v", record_buffer=[], target_layers=[layer_probe])
    attn_output_hook = RecordHook(target_name="attn_output", record_buffer=[], target_layers=[layer_probe])
    post_attn_residual_hook = RecordHook(target_name="post_attn_residual", record_buffer=[], target_layers=[layer_probe])
    post_attn_postnorm_hook = RecordHook(target_name="post_attn_postnorm", record_buffer=[], target_layers=[layer_probe])
    post_mlp_residual_hook = RecordHook(target_name="post_mlp_residual", record_buffer=[], target_layers=[layer_probe])
    hook = CompositeHook(hooks=[layer_input_prenorm_hook, layer_input_postnorm_hook, q_hook, k_hook, v_hook, q_postrope_hook, k_postrope_hook, v_postrope_hook, attn_logits_hook, attn_weights_hook, attn_softmax_qkt_v_hook, attn_output_hook, post_attn_residual_hook, post_attn_postnorm_hook, post_mlp_residual_hook])
    model_wrapper.forward(input_tokens, hook=hook)
    return {
        "layer_input_prenorm": layer_input_prenorm_hook.record_buffer[0],
        "layer_input_postnorm": layer_input_postnorm_hook.record_buffer[0],
        "q_proj": q_hook.record_buffer[0],
        "k_proj": k_hook.record_buffer[0],
        "v_proj": v_hook.record_buffer[0],
        "q_proj_postrope": q_postrope_hook.record_buffer[0],
        "k_proj_postrope": k_postrope_hook.record_buffer[0],
        "v_proj_postrope": v_postrope_hook.record_buffer[0],
        "attn_logits": attn_logits_hook.record_buffer[0],
        "attn_weights": attn_weights_hook.record_buffer[0],
        "attn_softmax_qkt_v": attn_softmax_qkt_v_hook.record_buffer[0],
        "attn_output": attn_output_hook.record_buffer[0],
        "post_attn_residual": post_attn_residual_hook.record_buffer[0],
        "post_attn_postnorm": post_attn_postnorm_hook.record_buffer[0],
        "post_mlp_residual": post_mlp_residual_hook.record_buffer[0],
    }
def plot_heatmap_per_head(scores, layer_probe, title, xlabel, ylabel, use_01_colorscheme=False):
    sns.set_theme(style="white")
    scores = to_numpy(scores)

    # Visualize each slice in the index 1 as a separate seaborn heatmap
    num_heads = scores.shape[0]
    assert num_heads == 32
    rows, cols = 4, 8
    fig, axes = plt.subplots(rows, cols, figsize=(20, 8))
    axes = axes.flatten()
    fig.suptitle(title)

    for head in range(num_heads):
        if use_01_colorscheme:
            sns.heatmap(scores[head], cmap="Blues", ax=axes[head], mask=scores[head] == 0, vmin=0, vmax=1)
        else:
            sns.heatmap(scores[head], cmap="Blues",  ax=axes[head], mask=scores[head] == 0)
        axes[head].set_title(f"L{layer_probe}H{head}")
        axes[head].set_xlabel(xlabel)
        axes[head].set_ylabel(ylabel)

    plt.tight_layout()
    plt.show()
    #sns.set_theme()


batch_size = 1
num_extra_tokens = 1
input_text = "Summer is warm. Winter is cold."
input_tokens = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
print(input_tokens.shape)
print(tokenizer.decode(input_tokens.cpu().tolist()[0], clean_up_tokenization_spaces=False))
torch.save(input_tokens, "input_tokens.pt")
input_tokens = torch.load("input_tokens.pt").to(device)
layer = 24
model = load_olmo_model(revisions_olmo[-1])
stats = collect_statistics_at_layer(model, input_tokens, layer_probe=layer)
print(stats["attn_weights"].shape)
plot_heatmap_per_head(stats["attn_weights"][0], layer, f"Attention Weights, Layer {layer}, Step {revisions_olmo[-1][0]}", "K", "Q", use_01_colorscheme=True)

del model 
del stats 
torch.cuda.empty_cache()
attn_weights = {}
attn_logits = {}
qs = {}
ks = {}
vs = {}
layer_outputs = {}

for revision in revisions_olmo:
    model = load_olmo_model(revision)
    stats = collect_statistics_at_layer(model, input_tokens, layer_probe=layer)
    attn_weights[revision] = stats["attn_weights"]
    attn_logits[revision] = stats["attn_logits"]
    qs[revision] = stats["q_proj_postrope"]
    ks[revision] = stats["k_proj_postrope"]
    vs[revision] = stats["v_proj_postrope"]
    layer_outputs[revision] = stats["post_mlp_residual"]
    del model
    del stats
    torch.cuda.empty_cache()
torch.save(attn_weights, "attn_weights.pt")
torch.save(attn_logits, "attn_logits.pt")
torch.save(qs, "qs.pt")
torch.save(ks, "ks.pt")
torch.save(vs, "vs.pt")
torch.save(layer_outputs, "layer_outputs.pt")
