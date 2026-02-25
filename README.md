# Unveiling Scaling Behaviors in Molecular Language Models: Effects of Model Size, Data, and Representation

<!-- badges (place right under the title) -->
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-required-red)
![Transformers](https://img.shields.io/badge/Transformers-required-orange)
![datasets](https://img.shields.io/badge/datasets-required-yellow)
![TensorBoard](https://img.shields.io/badge/TensorBoard-required-brightgreen)
![RDKit](https://img.shields.io/badge/RDKit-optional-lightgrey)

<sub>Paper: https://arxiv.org/abs/2601.22757</sub>

This repository is used to reproduce the core pretraining experiments in **Unveiling Scaling Behaviors in Molecular Language Models: Effects of Model Size, Data, and Representation**. The code is built on the training implementation from the **Trio** codebase, keeping the same GPT-style autoregressive modeling and dataset interface, while sweeping **model size** and **training token budget** to obtain the scaling results reported in the paper.


<p align="center">
  <img src="image/fig2_framework.png" width="900">
</p>


The diagram illustrates the entire research framework, which is divided into three parts:  
1) **Data and representations**: starting from ZINC and UniChem, each molecule is converted into five string representations: **DeepSMILES**, **FragLink**, **FragSeq**, **SAFE**, and **SMILES**.  
2) **Model and training**: a unified GPT architecture is pretrained with next-token prediction, and controlled sweeps are performed across model sizes and token budgets; in the paper, **LoRA** is used in a downstream stage to adapt the pretrained model to regression and classification tasks.  
3) **Evaluation**: the main axis is the minimum validation loss along a compute-optimal frontier, and transfer performance is further assessed on molecular property prediction benchmarks.

The main contributions can be summarized as:  
- Scaling is analyzed not only across model size and data size, but also with **molecular string representation** treated as a first-order factor in the same comparison framework.  
- Compute is handled explicitly by using compute-controlled training grids to characterize loss trends and compute-optimal behavior.  
- In the paper, downstream LoRA transfer experiments are used to relate pretraining differences to property-prediction performance.


## Installation

### Step 1. Create a Python environment

```bash
conda create -n mlm_scaling python=3.10 -y
conda activate mlm_scaling
```

### Step 2. Install dependencies

Install PyTorch for your CUDA version first, then install the rest:

```bash
pip install -r requirements.txt
```

### Step 3. Prepare the vocabulary file

Training needs a SMILES vocabulary file at `vocabs/vocab.txt`.

This tokenizer is a standard SMILES regex tokenizer derived from the rxnfp style setup. A practical way is to copy an existing vocab file used by rxnfp and place it at the path above.

After that, the repository should contain:

```text
vocabs/
  vocab.txt
```

## Data format

Training uses HuggingFace datasets `load_from_disk`.

Each dataset folder must contain:

- split `train`
- split `validation`
- column `input` in both splits, one sequence string per row


## Code map

### Used for the paper scaling experiments

- `train.py`  
  Distributed pretraining entry point. Loads a dataset from `--dataset_path`, builds the model from `--model_size`, trains for `--epochs`, and saves the best checkpoint to `./weights/<run_name>.pt`.

- `trainer.py`  
  Training loop and validation loop. It also handles the token based learning rate schedule and tensorboard logging.

- `model.py`  
  Decoder only Transformer and loss computation. Includes a `generate` method.

- `dataset.py`  
  `SmileDataset` reads the `input` column, adds BOS and EOS, tokenizes, and returns token ids. `SmileCollator` pads and creates shifted labels.

- `tokenizer.py`  
  `SmilesTokenizer` based on HuggingFace `BertTokenizer`, with SMILES regex tokenization.

- `test.py`  
  Recomputes validation loss for a directory of checkpoints and writes a CSV for plotting and scaling law fitting. Two places usually need editing:
  - `DATASET_PATHS` must point to your local datasets
  - `MODEL_SPECS` must match your training configs

## Run commands


### 1. Train one run

Example, SMILES, token budget 100M, model size 1M, 1 epoch:

```bash
python train.py \
  --dataset_path /path/to/SMILES-100M \
  --run_name SMILES-1M-100M-Epoch1 \
  --model_size 1M \
  --epochs 1 \
  --batch_size 128 \
  --world_size 1
```

Model size presets in `train.py`:

- 1M
- 4M
- 16M
- 56M
- 85M
- 152M
- 278M
- 650M

Note: `train.py` sets `CUDA_VISIBLE_DEVICES` inside the script. If you want a different GPU set, edit the line near the start of `train_worker`.

### 2. Duration controlled run

Same setting, trained for 2 epochs from scratch:

```bash
python train.py \
  --dataset_path /path/to/SMILES-100M \
  --run_name SMILES-1M-100M-Epoch2 \
  --model_size 1M \
  --epochs 2 \
  --batch_size 128 \
  --world_size 1
```

### 3. Export validation losses

Step 1, edit `DATASET_PATHS` and `MODEL_SPECS` in `test.py`.

Step 2, run:

```bash
python test.py \
  --ckpt_dir ./weights \
  --out_csv ./results/valid_loss.csv \
  --batch_size 32
```


## Acknowledgements

This repository is built on top of the Trio codebase, and parts of Trio are kept here for completeness.

- Trio: https://github.com/SZU-ADDG/Trio
