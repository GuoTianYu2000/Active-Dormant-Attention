# Active-Dormant-Attention
This repository contains the code for the experiments in [Active-Dormant Attention Heads:  Mechanistically Demystifying Extreme-Token Phenomena in LLMs](https://arxiv.org/pdf/2410.13835?)


## Getting started

### Environments
The BBM and LLM require different environments.

```
conda env create -f BBM/environment.yml
conda env create -f LLM/environment.yml
```

### BBM Training
Train the model on the BB tasks:
```
cd ./BBM
conda activate bbm
chmod +x ./scripts/1L-dynamics.sh
./scripts/1L-dynamics.sh
```
Run `BBM/scripts/1L-dynamics.sh` for recording the training dynamics of 1-layer transformer on the BB task. (cf. Figure 4(b) in [our paper](https://arxiv.org/pdf/2410.13835?))


### Tracking training with wandb
Change the `wandb_args.entity` arguments in any scripts.

### Evaluation and plotting
See `BBM/final-plots.ipynb` and `BBM/appendix-plot.ipynb` for details.

### Active-dormant mechanism in LLMs
See `LLM/OLMo_pretraining.ipynb` and `LLM/find_attn_sink_circuit.ipynb` for details.



