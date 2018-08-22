## Reinforced Mnemonic Reader in tensorflow
RMR: https://arxiv.org/abs/1705.02798

## Pipline
1. Run the ``preprocess.ipynb`` to get the input datasets.
2. Run ``train.py`` to start training.

## Updates
- [x] Init for the RMR model (without dynamic-critical reinforcement learning (DCRL))
- [ ] Add the DCRL (comming soon...)

## Results
Result on dev set of squad

EM:70.82% F1:79.51% (21800 steps without EMA)