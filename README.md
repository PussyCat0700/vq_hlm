# Installation
- install dependencies with `pip install -r requirements.txt`
- initialize submodule with `git submodule update --init --recursive`
- `cd vector-quantize-pytorch`, and install the submodule with `pip install -e .`

# Guidelines to run a new baseline

- 在`models.py`中查询或修改自己需要训练的VQ-VAE模型
- 在`conf/models`中添加自己的模型超参
- 以residualvq为例，可用以下命令训练模型：

`python train_vq.py --ckpt_dir ./runs/residualvq --model_config conf/models/residualvq.yaml`
- 训练完以后测试看下结果:

`python train_vq.py --ckpt_dir ./runs/residualvq --model_config conf/models/residualvq.yaml --test`
- 此外，`example_usage.ipynb`中有数据相关的可视化供大家参考

# Train a hlm model (on gpt2)
### Prepare token for training
1. You'd better have a folder `exp/$RUNNAME` with all model_config.yaml, data_config.yaml, best_checkpoint.pt in it.
2. Run `bash prepare_data.sh $RUNNAME` to prepare token for training.
3. Copy `datasets/htokens.py` to `path/to/your dataset`. Now you should have `htoken.py` with `{train, val, test}_tokens.pt` in it.
4. If you have a `htoken` folder in your dataset path before training, please remove it to avoid huggingface using cached data.

### Train a hlm model
1. Copy training config into `exp/$RUNNAME`. We recommend to customize it in `exp/$RUNNAME`, not in `conf/hlm_train`
2. Adjust number of gpu in `train_hlm_gpt2.sh`
3. Run `bash train_hlm_gpt2.sh $RUNNAME` to train a hlm model.
4. Recommended: use `sbatch runner_hlm_gpt2.slurm $RUNNAME` to submit a slurm job. **Remember to change your slurm config!**


# Change Log


- [24.12.22]
    - 新增训练自动续点续传功能
    - 新增patience参数用于切换到early stopping模式，需要指定patience>=1
    - 恢复默认的1 epoch设置
    - test set evaluation会保存`codebook_utilization.png`，更加方便报告可视化。

- [24.12.20]
    - 新增[TruthX](https://github.com/ictnlp/TruthX/blob/a41093a6ae3bcbcb523759da782de0f329d03d91/truthx.py#L261)自定义模型，参数量7.36M，比原先toy setting的0.05M大不少。
    > In TruthX, the truthful encoder and semantic encoder consist of 2-layer MLPs with dimensions [4096 → 2048, 2048 → 1024], and the decoder consists of 2-layer MLPs with dimensions [1024 → 2048, 2048 → 4096].
    The specific structure is presented in Appendix A. Following Li et al. (2023b) and Chen et al. (2024), we employ a 2-fold validation on TruthfulQA to ensure no overlap between training and testing.
    For training, TruthX is optimized using Adam optimizer with a learning rate of 1e-4.
    Based on the performance on validation set, we set the number of editing layers k = 10 and the editing strength α = 1.0 and α = 4.5 for the open-ended generation and multiple-choice task.
    - 功能更新
        - 为适应不同模型训练量大小，将1 epoch设置取消，换为patience动态决定训练何时停止。
        - `get_model`加入参数量统计功能

- [24.12.18] 修复reconstruction loss，去除clamp；监测指标变为reconstruction loss。

- [24.12.08] 新增模型集合文件`models.py`，供同学们参考和替换为自己的模型。

- [24.12.06] 在`train_vq.py`中新增VectorQuantize的例子

- [24.12.05] 新增vector-quantize-pytorch作为submodule
    - 后续开发可以继承submodule中的类，避免复制粘贴一堆代码

- [24.12.05] 新增`dataloading.py`
    - `ChunkedDataset`可用于训练VQ-VAE读取数据

- [24.12.02] 导出脚本汇总于`exporter`