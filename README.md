## Reinforced Mnemonic Reader in tensorflow
RMR: https://arxiv.org/abs/1705.02798

## Pipline
1. Run the ``preprocess.ipynb`` to get the input datasets.
2. Run ``train_h5py.py`` to start training. Now elmo and cove is not useable.

### notes
1. `conv1d` in `tensor2tensor` is used to instead of the matrix matmul (full connection) operation in RMR model.
2. Welcome to discuss any problem about this project (especially the RL loss).
3. The reinforcement loss should be used after the convergence of cross-entropy.

## Updates
- [x] Init for the RMR model (without dynamic-critical reinforcement learning DCRL)
- [x] Add the self-critical sequence training (SCST) (no test)
- [x] Update cudnn LSTM and squad 2.0
- [ ] Test the RL loss

## Results
Result on dev set of squad 1.1
EM:71.17% F1:79.56% (no elmo, no cove)

Result on dev set of squad 2.0
EM:64.89% F1:67.81% (+elmo+cove)