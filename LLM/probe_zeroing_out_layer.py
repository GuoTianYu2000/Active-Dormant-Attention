import torch 
from pathlib import Path 
from matplotlib import pyplot as plt
import seaborn as sns
from src.hook import Hook, RecordHook, ZeroOutAttentionHeadHook, CompositeHook
from src.models import load_model_tokenizer, get_layers_heads
from src.model_wrapper import ModelWrapper

device = "cuda:2"

model_name = "llama3.1-8B"
model, tokenizer = load_model_tokenizer(model_name, device_map=device)
num_layers, num_heads = get_layers_heads(model_name)
wrapper = ModelWrapper(model)

text = "Summer. Summer. Summer. Summer."
inputs = tokenizer(text, return_tensors="pt").input_ids.to(device)
hook_results = []
layer_2_post_attn_residual_record_hook = RecordHook(
    target_name="post_attn_residual",
    record_buffer=hook_results,
    target_layers=[2],
)
outputs1 = wrapper.forward(inputs, hook=layer_2_post_attn_residual_record_hook)

zero_out_layer_hook = ZeroOutAttentionHeadHook({2: list(range(num_heads))})
composite_hook = CompositeHook([layer_2_post_attn_residual_record_hook, zero_out_layer_hook])
outputs2 = wrapper.forward(inputs, hook=composite_hook)