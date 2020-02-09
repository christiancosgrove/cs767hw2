# Deep Learning for Automated Discourse - Homework 2

**Note: you need Git LFS to download the model files**

Our fork of [ParlAI](https://github.com/facebookresearch/ParlAI) with included model files for homework 2

## Our Model
### Details
1. Trained on a Google Cloud instance with 13 GB RAM, 2 cores, 1 NVIDIA K80
1. Seq2Seq model without pretrained word embeddings.
1. Trained on ParlAI Twitter dataset
1. Trained for ~ 12 hours.

### Results



## Instructions
### Train

`nohup python examples/train_model.py -t twitter -m seq2seq/seq2seq -mf hello_seq2seq -bs 10 -stim 3600 --max-train-time 59800 &`

### Evaluate

`python -m parlai.scripts.eval_model -m seq2seq/seq2seq -mf hello_seq2seq.checkpoint -t twitter -bs 50`

### Chat

`python -m parlai.scripts.interactive -m seq2seq/seq2seq -mf hello_seq2seq.checkpoint`
