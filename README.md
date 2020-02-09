# Deep Learning for Automated Discourse - Homework 2

**Note: you need [Git LFS](https://git-lfs.github.com/) to download the model files**

Our fork of [ParlAI](https://github.com/facebookresearch/ParlAI) with included model files for homework 2

## Our Model
### Details
1. Trained on a Google Cloud instance with 13 GB RAM, 2 cores, 1 NVIDIA K80
1. Seq2Seq model without pretrained word embeddings.
1. Trained on ParlAI Twitter dataset
1. Trained for ~ 12 hours.

### Results

Evaluated on validation set:
`[ Finished evaluating tasks ['twitter'] using datatype valid ]
{'exs': 10405, 'accuracy': 9.610764055742432e-05, 'f1': 0.055454937351312183, 'bleu-4': 0.0001680273494616013, 'lr': 1, 'total_train_updates': 176783, 'gpu_mem_percent': 0.29, 'loss': 6075.0, 'token_acc': 0.2306, 'nll_loss': 5.832, 'ppl': 340.9}`


#### Example outputs
A few example outputs (in Forever format) have been recorded in `test_outputs`.

**Perplexity:** 340.9

## Instructions
### Train

`nohup python examples/train_model.py -t twitter -m seq2seq/seq2seq -mf hello_seq2seq -bs 10 -stim 3600 --max-train-time 59800 &`

### Evaluate

`python -m parlai.scripts.eval_model -m seq2seq/seq2seq -mf hello_seq2seq.checkpoint -t twitter -bs 50`

### Chat

**Note**: We modified `world_logging.py` to include an option to print to the [Forever chat specification](https://github.com/jkeen/forever-chat-format).

`python -m parlai.scripts.interactive -m seq2seq/seq2seq -mf hello_seq2seq.checkpoint --log-keep-fields all --report_filename test.json --save_world_logs True`

The JSON output will be recorded in `test_replies.json`.