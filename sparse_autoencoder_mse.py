import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Any
import os
import glob
import tqdm


# 定义LN函数
def LN(x: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = x.mean(dim=-1, keepdim=True)
    x = x - mu
    std = x.std(dim=-1, keepdim=True)
    x = x / (std + eps)
    return x, mu, std

# TopK激活函数
class TopK(nn.Module):
    def __init__(self, k: int, postact_fn: Callable = nn.ReLU()) -> None:
        super().__init__()
        self.k = k
        self.postact_fn = postact_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        topk = torch.topk(x, k=self.k, dim=-1)
        values = self.postact_fn(topk.values)
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        return result

class Autoencoder(nn.Module):
    def __init__(
        self, n_latents: int, n_inputs: int, activation: Callable = nn.ReLU(), tied: bool = False,
        normalize: bool = True  # 默认启用归一化
    ) -> None:
        super().__init__()

        self.pre_bias = nn.Parameter(torch.zeros(n_inputs))
        self.encoder = nn.Linear(n_inputs, n_latents, bias=False)
        self.latent_bias = nn.Parameter(torch.zeros(n_latents))
        self.activation = activation
        self.decoder = nn.Linear(n_latents, n_inputs, bias=False)
        self.normalize = normalize

    def preprocess(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        if not self.normalize:
            return x, dict()
        x, mu, std = LN(x)
        return x, dict(mu=mu, std=std)

    def encode_pre_act(self, x: torch.Tensor) -> torch.Tensor:
        x = x - self.pre_bias
        return F.linear(x, self.encoder.weight, self.latent_bias)

    def decode(self, latents: torch.Tensor, info: dict[str, Any] | None = None) -> torch.Tensor:
        ret = self.decoder(latents) + self.pre_bias
        if self.normalize and info:
            ret = ret * info["std"] + info["mu"]
        return ret

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, info = self.preprocess(x)
        latents_pre_act = self.encode_pre_act(x)
        latents = self.activation(latents_pre_act)
        recons = self.decode(latents, info)
        return latents_pre_act, latents, recons
    
    @classmethod
    def from_state_dict(cls, state_dict: dict[str, torch.Tensor]) -> "Autoencoder":
        # 从权重文件确定维度
        if "W_enc" in state_dict:
            w_enc = state_dict["W_enc"]
            if w_enc.shape[0] < w_enc.shape[1]:
                n_inputs, n_latents = w_enc.shape
            else:
                n_latents, n_inputs = w_enc.shape
        else:
            raise ValueError("权重文件中缺少 W_enc")
    
        # 创建模型实例 - 启用归一化
        k = 32
        activation = TopK(k=k)
        autoencoder = cls(n_latents, n_inputs, activation=activation, normalize=True)
    
        # 准备权重映射
        new_state_dict = {}
        
        # 编码器权重
        if "W_enc" in state_dict:
            if w_enc.shape[0] < w_enc.shape[1]:
                new_state_dict["encoder.weight"] = w_enc.t()
            else:
                new_state_dict["encoder.weight"] = w_enc
                
        # 解码器权重
        if "W_dec" in state_dict:
            w_dec = state_dict["W_dec"]
            expected_shape = autoencoder.decoder.weight.shape
            
            if w_dec.shape == expected_shape:
                new_state_dict["decoder.weight"] = w_dec
            elif w_dec.t().shape == expected_shape:
                new_state_dict["decoder.weight"] = w_dec.t()
            else:
                new_state_dict["decoder.weight"] = w_dec
                
        # 偏置
        if "b_enc" in state_dict:
            new_state_dict["latent_bias"] = state_dict["b_enc"]
        if "b_dec" in state_dict:
            new_state_dict["pre_bias"] = state_dict["b_dec"]
    
        # 加载处理后的权重
        try:
            autoencoder.load_state_dict(new_state_dict, strict=False)
        except RuntimeError as e:
            if "decoder.weight" in new_state_dict:
                with torch.no_grad():
                    if new_state_dict["decoder.weight"].shape[0] == autoencoder.decoder.weight.shape[1] and \
                       new_state_dict["decoder.weight"].shape[1] == autoencoder.decoder.weight.shape[0]:
                        autoencoder.decoder.weight.copy_(new_state_dict["decoder.weight"].t())
                    else:
                        raise e
        
        return autoencoder

def evaluate_normalized_mse(model_path,hidden_states_dir,layer_num=5):
    # 加载模型权重
    
    state_dict = torch.load(f'{model_path}/sae_weights.pt')
    print(f"已加载层{layer_num}的SAE权重")
    
    # 创建归一化的SAE模型
    model = Autoencoder.from_state_dict(state_dict)
    print("已创建归一化SAE模型")
    
    # 找到所有layer_5.pt隐藏状态文件
    
    layer_files = glob.glob(f"{hidden_states_dir}*layer_{layer_num}.pt")
    
    if not layer_files:
        print("未找到任何layer_5.pt文件")
        return 0
    
    print(f"找到{len(layer_files)}个文件待处理")
    
    # 计算总MSE和样本数
    total_mse = 0.0
    total_samples = 0
    
    # 使用tqdm创建进度条
    with torch.no_grad():
        for file_path in tqdm(layer_files, desc="处理文件进度", unit="文件"):
            # 加载隐藏状态
            hidden_states = torch.load(file_path)
            
            # 前向传播
            _, _, reconstructed = model(hidden_states)
            
            # 计算MSE
            mse = F.mse_loss(reconstructed, hidden_states)
            
            # 累计MSE和样本数
            total_mse += mse.item() * hidden_states.shape[0]
            total_samples += hidden_states.shape[0]
    
    # 计算平均MSE
    avg_mse = total_mse / total_samples if total_samples > 0 else 0
    print(f"层{layer_num}的归一化SAE平均MSE: {avg_mse:.6f}")
    
    return avg_mse

if __name__ == "__main__":
    model_path = f'/root/filesystem/modelGPT2/sparse_autoencoder/sae_pt/v5_32k_layer_5.pt'
    hidden_states_dir = "/root/filesystem/modelGPT2/sparse_autoencoder/hidden_states/"
    evaluate_normalized_mse(layer_num=5,hidden_states_dir=hidden_states_dir)
