Please put `htoken.py` into your dataset folder with `{train, val, test}_tokens.pt`.

**Attention!!** Since huggingface has dataset cache for same dataset name, you need to clear the cache before you change your tokens.
The cache dir is typically `~/.cache/huggingface/datasets/htokens`.

**How to get tokens?**
1. Prepare the vq model folder with `data_config.yaml`, `model_config.yaml` and `best_checkpoint.pt` in `your_vq_model` folder.
2. Run `bash prepare_data.sh your_vq_model` in main folder.
3. If everything is fine, you will get `{train, val, test}_tokens.pt` in `your_vq_model` folder.
4. Remember to copy `htoken.py` into your dataset folder.
5. You can change the path to save tokens `save_dir` in `prepare_hlm_token.py` after `save_dir = data_config['h5_file_path']`. You can also run `python prepare_hlm_token.py --data_config {path/to/your/data_config.yaml} --model_config {path/to/your/model_config.yaml} --ckpt_dir {path/to/your/ckpt_dir}` to generate tokens.