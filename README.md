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

# Change Log

- [24.12.02] 导出脚本汇总于`exporter`

- [24.12.05] 新增`dataloading.py`
    - `ChunkedDataset`可用于训练VQ-VAE读取数据

- [24.12.05] 新增vector-quantize-pytorch作为submodule
    - 后续开发可以继承submodule中的类，避免复制粘贴一堆代码

- [24.12.06] 在`train_vq.py`中新增VectorQuantize的例子

- [24.12.08] 新增模型集合文件`models.py`，供同学们参考和替换为自己的模型。

- [24.12.18] 修复reconstruction loss，去除clamp；监测指标变为reconstruction loss。