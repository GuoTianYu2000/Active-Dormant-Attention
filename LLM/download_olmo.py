import torch
from transformers import OlmoForCausalLM, AutoTokenizer, GPTNeoXForCausalLM, GPTNeoXTokenizerFast


for revision in [
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
    (150000, "step150000-tokens628B"),
    (200000, "step200000-tokens838B"),
    (250000, "step250000-tokens1048B"),
    (300000, "step300000-tokens1257B"),
    (350000, "step350000-tokens1467B"),
    (400000, "step410000-tokens1719B"),
    (450000, "step449000-tokens1882B")
]:
    model = OlmoForCausalLM.from_pretrained("allenai/OLMo-7B-0424-hf", revision=revision[1], trust_remote_code=True, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B-0424-hf", trust_remote_code=True)

# for revision in [
#     "step1000",
#     "step10000",
#     "step25000",
#     "step50000",
#     "step75000",
#     "step100000",
#     "step125000",
#     "step143000",
# ]:
#     model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-6.9b", revision=revision, trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-6.9b", trust_remote_code=True)

