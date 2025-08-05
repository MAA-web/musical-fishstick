import os
import re
import time
import math
import json
import torch
import socket
import pickle
import requests
import threading
from glob import iglob
from torch import Tensor
from pathlib import Path
from typing import Callable
from torch import Tensor, nn
from PIL import ExifTags, Image
from dataclasses import dataclass
from safetensors.torch import load_file as load_sft
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer


from einops import rearrange, repeat

from transformers import pipeline

# from flux.util import (
#     check_onnx_access_for_trt,
#     configs,
#     load_ae,
#     load_clip,
#     load_flow_model,
#     load_t5,
#     save_image,
# )


# from flux.modules.layers import (
#     DoubleStreamBlock,
#     EmbedND,
#     LastLayer,
#     MLPEmbedder,
#     SingleStreamBlock,
#     timestep_embedding,
# )
# from flux.modules.lora import LinearLora, replace_linear_with_lora


@dataclass
class AutoEncoderParams:
    resolution: int
    in_channels: int
    ch: int
    out_ch: int
    ch_mult: list[int]
    num_res_blocks: int
    z_channels: int
    scale_factor: float
    shift_factor: float




@dataclass
class FluxParams:
    in_channels: int
    out_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool


@dataclass
class ModelSpec:
    params: FluxParams
    ae_params: AutoEncoderParams
    repo_id: str
    repo_flow: str
    repo_ae: str
    lora_repo_id: str | None = None
    lora_filename: str | None = None


configs = {
    "flux-dev": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        params=FluxParams(
            in_channels=64,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-schnell": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-schnell",
        repo_flow="flux1-schnell.safetensors",
        repo_ae="ae.safetensors",
        params=FluxParams(
            in_channels=64,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
        ),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-canny": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-Canny-dev",
        repo_flow="flux1-canny-dev.safetensors",
        repo_ae="ae.safetensors",
        params=FluxParams(
            in_channels=128,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-canny-lora": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        lora_repo_id="black-forest-labs/FLUX.1-Canny-dev-lora",
        lora_filename="flux1-canny-dev-lora.safetensors",
        params=FluxParams(
            in_channels=128,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-depth": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-Depth-dev",
        repo_flow="flux1-depth-dev.safetensors",
        repo_ae="ae.safetensors",
        params=FluxParams(
            in_channels=128,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-depth-lora": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        lora_repo_id="black-forest-labs/FLUX.1-Depth-dev-lora",
        lora_filename="flux1-depth-dev-lora.safetensors",
        params=FluxParams(
            in_channels=128,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-redux": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-Redux-dev",
        repo_flow="flux1-redux-dev.safetensors",
        repo_ae="ae.safetensors",
        params=FluxParams(
            in_channels=64,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-fill": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-Fill-dev",
        repo_flow="flux1-fill-dev.safetensors",
        repo_ae="ae.safetensors",
        params=FluxParams(
            in_channels=384,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-kontext": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-Kontext-dev",
        repo_flow="flux1-kontext-dev.safetensors",
        repo_ae="ae.safetensors",
        params=FluxParams(
            in_channels=64,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
}




def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe)

    x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    x = rearrange(x, "B H L D -> B L (H D)")

    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)



class HFEmbedder(nn.Module):
    def __init__(self, version: str, max_length: int, **hf_kwargs):
        super().__init__()
        self.is_clip = version.startswith("openai")
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        if self.is_clip:
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(version, max_length=max_length)
            self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(version, **hf_kwargs)
        else:
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(version, max_length=max_length)
            self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(version, **hf_kwargs)

        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, text: list[str]) -> Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key].bfloat16()

class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))
    

@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x
    

class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor) -> tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img blocks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        # calculate the txt blocks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        return img, txt


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        # compute attention
        attn = attention(q, k, v, pe=pe)
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod.gate * output


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding






class DualStreamSender:
    def __init__(self, double_stream_server_address, single_stream_server_address):
        """
        Initializes the DualStreamSender.

        :param double_stream_server_address: Tuple containing the IP and port of the double stream server.
        :param single_stream_server_address: Tuple containing the IP and port of the single stream server.
        """
        self.double_stream_server_address = double_stream_server_address
        self.single_stream_server_address = single_stream_server_address

    def send_to_double_stream_server(self, data):
        """Sends data to the double stream server and returns the response."""
        return self._send_data(self.double_stream_server_address, data)

    def send_to_single_stream_server(self, data):
        """Sends data to the single stream server and returns the response."""
        return self._send_data(self.single_stream_server_address, data)

    def _send_data(self, server_address, data):
        """Helper method to send data to a server and receive response."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(server_address)
                s.sendall(json.dumps(data).encode())
                response = s.recv(1024)
                return json.loads(response.decode())
        except Exception as e:
            return {"error": str(e)}




class DoubleStreamServer:
    def __init__(self, host='127.0.0.1', port=65432):
        self.host = host
        self.port = port

    def handle_client(self, conn):
        with conn:
            data = conn.recv(1024)
            if not data:
                return
            try:
                data_dict = json.loads(data.decode())
                streams = data_dict.get("streams", [])
                # Compute double streams
                double_streams = [stream * 2 for stream in streams]
                conn.sendall(json.dumps(double_streams).encode())
            except Exception as e:
                conn.sendall(json.dumps({"error": str(e)}).encode())

    def start(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            s.listen()
            print(f"Server listening on {self.host}:{self.port}")
            while True:
                conn, addr = s.accept()
                thread = threading.Thread(target=self.handle_client, args=(conn,))
                thread.start()








class WorkerServer:
    def __init__(self, mode='none', host='localhost', port=80085):
        self.host = host
        self.port = port
        self.mode = mode
        self.model = None
    
    def establish_socket_stream_connection(self):
        """
        Establishes a socket stream connection to the master server.
        """
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
        print(f"Connected to master server at {self.host}:{self.port}")

    def load_model(self, model_spec: ModelSpec):
        if self.mode == 'double':
            self.model = load_flow_model_DoubleStream()
        if self.mode == 'single':
            self.model = load_flow_model_SingleStream()

    def listen_for_tasks(self, mode='single'):
        """
        Keeps the connection alive, listens for tasks from the master server,
        and processes them.
        """
        try:
            while True:
                # 1. Receive length of incoming message
                length_bytes = self.socket.recv(8)
                print(f"length of the 'length_bytes':{length_bytes}")


                if not length_bytes:
                    print("Master didn't send anything yet.")
                    continue
                    # break
                
                msg_len = int.from_bytes(length_bytes, 'big')

                # 2. Receive full data
                data = self.recvall(msg_len)
                request = pickle.loads(data)

                print("Received request:", request)

                # 3. Process based on mode
                if request["type"] == self.mode:    # single
                    img = torch.tensor(request["img"])
                    vec = torch.tensor(request["vec"])
                    pe = torch.tensor(request["pe"])

                    # Replace with your actual model logic
                    result = {
                        "img": (img + 1).numpy()  # Dummy operation
                    }


                elif request["type"] == self.mode:  # double
                    img = torch.tensor(request["img"])
                    txt = torch.tensor(request["txt"])
                    vec = torch.tensor(request["vec"])
                    pe = torch.tensor(request["pe"])

                    # Replace with your actual model logic
                    result = {
                        "img": (img + 1).numpy(),
                        "txt": (txt + 1).numpy()
                    }


                # 3. Process based on mode
                elif request["type"] == "ping":
                    result = {"status": "alive"}
                    # response = pickle.dumps(result)
                    # self.socket.sendall(len(response).to_bytes(8, 'big'))
                    # self.socket.sendall(response)
                    # continue
                                
                else:
                    print("Unknown task type:")
                    print(request)
                    result = {"status": "unknown task"}

                # 4. Send back response
                print(result)
                response = pickle.dumps(result)
                self.socket.send(len(response).to_bytes(8, 'big'))
                print(len(response))
                self.socket.sendall(response)
                print("Length bytes sent:", len(response).to_bytes(8, 'big').hex())


                continue  # Continue to listen for more tasks

        except Exception as e:
            print("Worker crashed with error:", e)

    def recvall(self, length: int) -> bytes:
        """
        Helper function to receive exactly `length` bytes from the socket.
        """
        data = b''
        while len(data) < length:
            more = self.socket.recv(length - len(data))
            if not more:
                raise EOFError("Socket closed unexpectedly")
            data += more
        return data



import flask
class MasterServer:
    def __init__(self, host='localhost', port=80085, params: FluxParams | None = None):
        self.host = host
        self.port = port
        self.servers :socket = []
        self.model = None
        self.MAX_MSG_SIZE = 1024 * 1024 * 10  # 10 MB


    # I'll replace a lot of the functionality with the denoise_over_network function
    
    def denoise_over_network(
        self,
        server_list: list,
        # model input
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        vec: Tensor,
        # sampling parameters
        timesteps: list[float],
        guidance: float = 4.0,
        # extra img tokens (channel-wise)
        img_cond: Tensor | None = None,
        # extra img tokens (sequence-wise)
        img_cond_seq: Tensor | None = None,
        img_cond_seq_ids: Tensor | None = None,
    ):
        # this is ignored for schnell
        guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
        for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
            img_input = img
            img_input_ids = img_ids
            if img_cond is not None:
                img_input = torch.cat((img, img_cond), dim=-1)
            if img_cond_seq is not None:
                assert (
                    img_cond_seq_ids is not None
                ), "You need to provide either both or neither of the sequence conditioning"
                img_input = torch.cat((img_input, img_cond_seq), dim=1)
                img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)
            
            pred = self.model(
                img=img_input,
                img_ids=img_input_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec,
                guidance=guidance_vec,
            )

            if img_input_ids is not None:
                pred = pred[:, : img.shape[1]]

            img = img + (t_prev - t_curr) * pred

        return img


    def test(self, conn :socket):
        # Send ping
        try:
            ping_msg = pickle.dumps({"type": "ping"})
            conn.sendall(len(ping_msg).to_bytes(8, 'big'))
            conn.sendall(ping_msg)

            # time.sleep(1)

            # Expect pong
            length_bytes = conn.recv(8)
            # length_bytes = self.recv_exact(conn, 8)
            # length_bytes = recv_exact(conn, 8)
            
            print(f"length of the 'length_bytes':{length_bytes}")
            
            if not length_bytes:
                print("No response from worker.")
                # dead_connections.append(conn)
                # continue

            msg_len = int.from_bytes(length_bytes, 'big')
            # if msg_len <= 0 or msg_len > self.MAX_MSG_SIZE:
            #     raise ValueError(f"Suspicious msg_len: {msg_len}")
            
            data = recv_exact(conn, msg_len)
            # data = self.recvall(conn, msg_len)

            response = pickle.loads(data)
            if response.get("status") == "alive":
                print("Worker is alive.")
            else:
                print("Unexpected response:", response)

        except Exception as e:
            print(f"Worker test failed: {e}")


    def start_to_accept_connections(self):
        """
        Accepts persistent connections from workers and stores their socket objects.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            s.listen()
            print(f"Master server listening on {self.host}:{self.port}")

            while True:
                conn, addr = s.accept()
                print(f"Accepted connection from {addr}")
                self.servers.append(conn)

                # Optionally handle messages from workers in separate threads
                # threading.Thread(target=self.worker_heartbeat_or_listener, args=(conn, addr), daemon=True).start()

                if len(self.servers) >= 2:
                    print("Two servers connected, ready to dispatch.")
                    break  # Break after accepting two connections for simplicity
                    


    # the function to run the master server
    def run(self):
        """
        Starts the master server to accept connections and handle tasks.
        """
        self.start_to_accept_connections()
        self.test_connections(interval=2)  # Check every 10 seconds
        
        # threading.Thread(target=self.start_to_accept_connections(), daemon=True).start()

        print("Loading model...")
        # self.model = Flux_master(self.servers, params=self.params)
        self.app = flask.Flask(__name__)

        
        # user front facing API
        @self.app.route('/', methods=['GET'])
        def slash():
            """Handles POST requests from the user and then sends it to be denoised."""
            # return "Master server is running."
            # data = flask.request.json()
            data = flask.request.get_json(silent=True)  # ✅ Safely attempts to parse JSON without throwing 415
            
            for conn in self.servers:
                self.test(conn)
            
            # here we will return the currently maintained connections with all the other servers
            return f"<h3>Number of connections being maintained: {len(self.servers)}</h3>"




            # img = self.denoise_over_network(
            #     server_list=self.servers,
            #     img=torch.tensor(data["img"]),
            #     img_ids=torch.tensor(data["img_input_ids"]),
            #     txt=torch.tensor(data["txt"]),
            #     txt_ids=torch.tensor(data["txt_ids"]),
            #     vec=torch.tensor(data["vec"]),
            #     timesteps=data["t_vec"],
            #     guidance=data.get("guidance_vec", 4.0),
            #     # extra img tokens (channel-wise)
            #     img_cond = None,
            #     # extra img tokens (sequence-wise)
            #     img_cond_seq = None,
            #     img_cond_seq_ids = None,
            # )

            # pred = self.model(
            #     img=data["img_input"],
            #     img_ids=data["img_input_ids"],
            #     txt=data["txt"],
            #     txt_ids=data["txt_ids"],
            #     y=data["vec"],
            #     timesteps=data["t_vec"],
            #     guidance=data["guidance_vec"],
            # )

            # return flask.jsonify({img})
        
        self.app.run(host=self.host, port=self.port)


    def worker_heartbeat_or_listener(self, conn, addr):
        # Optional: You can implement receiving logs, heartbeats, or stats here.
        try:
            while True:
                data = conn.recv(1024)
                if not data:
                    print(f"Connection closed by {addr}")
                    break
                try:
                    msg = pickle.loads(data)
                    print(f"Received message from {addr}: {msg}")
                except:
                    print(f"Received raw data from {addr}")
        except Exception as e:
            print(f"Error with worker {addr}: {e}")

    def recvall(self, conn, length: int) -> bytes:
        """
        Receives exactly `length` bytes from the socket.
        """
        data = b''
        while len(data) < length:
            more = conn.recv(length - len(data))
            if not more:
                raise EOFError("Socket closed unexpectedly")
            data += more
        return data

    def recv_exact(self, sock, n):
        data = b''
        while len(data) < n:
            more = sock.recv(n - len(data))
            print("receiving bytes")
            if not more:
                raise EOFError('Socket closed prematurely')
            data += more
        return data

    def test_connections(self, interval=1):
        """
        Periodically tests if worker connections are alive by sending a ping message.
        """
        def test_loop():
            while True:
                dead_connections = []
                for conn in self.servers:
                    try:
                        # Send ping
                        ping_msg = pickle.dumps({"type": "ping"})
                        conn.sendall(len(ping_msg).to_bytes(8, 'big'))
                        conn.sendall(ping_msg)

                        # time.sleep(1)

                        # Expect pong
                        length_bytes = conn.recv(8)
                        # length_bytes = self.recv_exact(conn, 8)
                        # length_bytes = recv_exact(conn, 8)
                        
                        print(f"length of the 'length_bytes':{length_bytes}")
                        
                        if not length_bytes:
                            print("No response from worker.")
                            dead_connections.append(conn)
                            continue

                        msg_len = int.from_bytes(length_bytes, 'big')
                        if msg_len <= 0 or msg_len > self.MAX_MSG_SIZE:
                            raise ValueError(f"Suspicious msg_len: {msg_len}")
                        
                        data = recv_exact(conn, msg_len)
                        # data = self.recvall(conn, msg_len)

                        response = pickle.loads(data)
                        if response.get("status") == "alive":
                            print("Worker is alive.")
                        else:
                            print("Unexpected response:", response)

                    except Exception as e:
                        print(f"Worker test failed: {e}")
                        # dead_connections.append(conn)
                    # print("...................")
                # Remove dead connections
                for dead in dead_connections:
                    if dead in self.servers:
                        # self.servers.remove(dead)
                        print("Removed dead connection.")
                
                # continue

                time.sleep(interval)

        # Run test loop in background
        threading.Thread(target=test_loop, daemon=True).start()




def recv_exact(sock, n):
    data = b''
    while len(data) < n:
        more = sock.recv(n - len(data))
        if not more:
            raise ConnectionError("Socket connection lost")
        data += more
    return data




# TODO make 2 classes one for DoubleStreamBlock and one for SingleStreamBlock

class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: FluxParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )

        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)

        vec = self.time_in(timestep_embedding(timesteps, 256))

        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))

        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        for block in self.double_blocks:        # double stream server
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        img = torch.cat((txt, img), 1)          # double stream server

        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)    # single stream server
            
        img = img[:, txt.shape[1] :, ...]       # master server

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img




class Flux_master(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, servers, params: FluxParams):
        super().__init__()
        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )
        self.servers = servers,
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        # self.double_blocks = nn.ModuleList(
        #     [
        #         DoubleStreamBlock(
        #             self.hidden_size,
        #             self.num_heads,
        #             mlp_ratio=params.mlp_ratio,
        #             qkv_bias=params.qkv_bias,
        #         )
        #         for _ in range(params.depth)
        #     ]
        # )

        # self.single_blocks = nn.ModuleList(
        #     [
        #         SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
        #         for _ in range(params.depth_single_blocks)
        #     ]
        # )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)

        vec = self.time_in(timestep_embedding(timesteps, 256))

        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))

        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        # self.servers[0]
        img, txt = self.dispatch_to_doublestream_server(self.servers[0], img, txt, vec, pe)

        # for block in self.double_blocks:        # double stream server
        #     img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        img = torch.cat((txt, img), 1)          # double stream server

        # self.servers[1]
        img = self.dispatch_to_singlestream_server(self.servers[1], img, vec, pe)
        
        # for block in self.single_blocks:
        #     img = block(img, vec=vec, pe=pe)    # single stream server
            
        img = img[:, txt.shape[1] :, ...]       # master server

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img

    def dispatch_to_doublestream_server(self, server_socket: socket.socket, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor) -> list[Tensor]:
        data = {
            "type": "double",
            "img": img.cpu().numpy(),
            "txt": txt.cpu().numpy(),
            "vec": vec.cpu().numpy(),
            "pe": pe.cpu().numpy(),
        }

        serialized = pickle.dumps(data)
        server_socket.sendall(len(serialized).to_bytes(8, 'big'))
        server_socket.sendall(serialized)

        length_bytes = self.recvall(server_socket, 8)
        response_length = int.from_bytes(length_bytes, 'big')
        response_data = self.recvall(server_socket, response_length)
        response = pickle.loads(response_data)

        return [
            torch.tensor(response["img"]).to(img.device),
            torch.tensor(response["txt"]).to(txt.device),
        ]

    def dispatch_to_singlestream_server(self, server_socket: socket.socket, img: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        data = {
            "type": "single",
            "img": img.cpu().numpy(),
            "vec": vec.cpu().numpy(),
            "pe": pe.cpu().numpy(),
        }

        serialized = pickle.dumps(data)
        server_socket.sendall(len(serialized).to_bytes(8, 'big'))
        server_socket.sendall(serialized)

        length_bytes = self.recvall(server_socket, 8)
        response_length = int.from_bytes(length_bytes, 'big')
        response_data = self.recvall(server_socket, response_length)
        response = pickle.loads(response_data)

        return torch.tensor(response["img"]).to(img.device)

    def recvall(self, sock: socket.socket, length: int) -> bytes:
        data = b''
        while len(data) < length:
            more = sock.recv(length - len(data))
            if not more:
                raise EOFError("Socket connection broken")
            data += more
        return data






# class FluxLoraWrapper(Flux):
#     def __init__(
#         self,
#         lora_rank: int = 128,
#         lora_scale: float = 1.0,
#         *args,
#         **kwargs,
#     ) -> None:
#         super().__init__(*args, **kwargs)

#         self.lora_rank = lora_rank

#         replace_linear_with_lora(
#             self,
#             max_rank=lora_rank,
#             scale=lora_scale,
#         )

#     def set_lora_scale(self, scale: float) -> None:
#         for module in self.modules():
#             if isinstance(module, LinearLora):
#                 module.set_scale(scale=scale)
# this must go as well



class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x




class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        # get dtype for proper tracing
        upscale_dtype = next(self.up.parameters()).dtype

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # cast to proper dtype
        h = h.to(upscale_dtype)
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h
    



def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h

class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor):
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        # downsampling
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h



class AutoEncoder(nn.Module):
    def __init__(self, params: AutoEncoderParams, sample_z: bool = False):
        
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("Initializing AutoEncoder with parameters:")
        print("Resolution:", params.resolution)
        print(params)
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

        super().__init__()
        self.params = params
        self.encoder = Encoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.decoder = Decoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            out_ch=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.reg = DiagonalGaussian(sample=sample_z)

        self.scale_factor = params.scale_factor
        self.shift_factor = params.shift_factor

    def encode(self, x: Tensor) -> Tensor:
        z = self.reg(self.encoder(x))
        z = self.scale_factor * (z - self.shift_factor)
        return z

    def decode(self, z: Tensor) -> Tensor:
        z = z / self.scale_factor + self.shift_factor
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))
    



class DiagonalGaussian(nn.Module):
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim

    def forward(self, z: Tensor) -> Tensor:
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        if self.sample:
            std = torch.exp(0.5 * logvar)
            return mean + std * torch.randn_like(mean)
        else:
            return mean












CHECKPOINTS_DIR = Path("checkpoints")
CHECKPOINTS_DIR.mkdir(exist_ok=True)
BFL_API_KEY = os.getenv("BFL_API_KEY")

os.environ.setdefault("TRT_ENGINE_DIR", str(CHECKPOINTS_DIR / "trt_engines"))
(CHECKPOINTS_DIR / "trt_engines").mkdir(exist_ok=True)




def optionally_expand_state_dict(model: torch.nn.Module, state_dict: dict) -> dict:
    """
    Optionally expand the state dict to match the model's parameters shapes.
    """
    for name, param in model.named_parameters():
        if name in state_dict:
            if state_dict[name].shape != param.shape:
                print(
                    f"Expanding '{name}' with shape {state_dict[name].shape} to model parameter with shape {param.shape}."
                )
                # expand with zeros:
                expanded_state_dict_weight = torch.zeros_like(param, device=state_dict[name].device)
                slices = tuple(slice(0, dim) for dim in state_dict[name].shape)
                expanded_state_dict_weight[slices] = state_dict[name]
                state_dict[name] = expanded_state_dict_weight

    return state_dict


def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))


def get_checkpoint_path(repo_id: str, filename: str, env_var: str) -> Path:
    """Get the local path for a checkpoint file, downloading if necessary."""
    # THERE WILL BE NO DOWNLOAD UNDER MY WATCH MUWAHAHAHAHAHAHA
    
    if os.environ.get(env_var) is not None:
        local_path = os.environ[env_var]
        if os.path.exists(local_path):
            return Path(local_path)

        print(
            f"Trying to load model {repo_id}, {filename} from environment "
            f"variable {env_var}. But file {local_path} does not exist. "
            "Falling back to default location."
        )

    # Create a safe directory name from repo_id
    safe_repo_name = repo_id.replace("/", "_")
    checkpoint_dir = CHECKPOINTS_DIR / safe_repo_name
    checkpoint_dir.mkdir(exist_ok=True)

    local_path = checkpoint_dir / filename
    print(f"Checking for {filename} in {local_path}")
    

    # I have decreed that the local path will exist


    # if not local_path.exists():
    #     print(f"Downloading {filename} from {repo_id} to {local_path}")
    #     try:
    #         ensure_hf_auth()
    #         hf_hub_download(repo_id=repo_id, filename=filename, local_dir=checkpoint_dir)
    #     except Exception as e:
    #         if "gated repo" in str(e).lower() or "restricted" in str(e).lower():
    #             print(f"\nError: Cannot access {repo_id} -- this is a gated repository.")

    #             # Try one more time to authenticate
    #             if prompt_for_hf_auth():
    #                 # Retry the download after authentication
    #                 print("Retrying download...")
    #                 hf_hub_download(repo_id=repo_id, filename=filename, local_dir=checkpoint_dir)
    #             else:
    #                 print("Authentication failed or cancelled.")
    #                 print("You can also run 'huggingface-cli login' or set HF_TOKEN environment variable")
    #                 raise RuntimeError(f"Authentication required for {repo_id}")
    #         else:
    #             raise e

    return local_path


def load_flow_model(name: str, device: str | torch.device = "cuda", verbose: bool = True, path: str = "./FLUX-research/FLUX-1.dev/transformer/diffusion_pytorch_model-00001-of-00003.safetensors") -> Flux:
    # Loading Flux
    print("Init model")
    config = configs[name]
    
    print(f"Loading model {name} with parameters: {config.params}")


    # checkpoint_path = str(get_checkpoint_path(config.repo_id, config.repo_flow, "FLUX_MODEL"))

    checkpoint_path = str(path)

    # with torch.device("meta"):
    #     if config.lora_repo_id is not None and config.lora_filename is not None:
    #         # model = FluxLoraWrapper(params=config.params).to(torch.bfloat16)
    #         pass
    #     else:
    #         model = Flux(config.params).to(torch.bfloat16)

    model = Flux(config.params).to(torch.bfloat16)


    print(f"Loading checkpoint: {checkpoint_path}")
    # load_sft doesn't support torch.device
    state_dict = load_sft(checkpoint_path, device=str(device))      # state dict has all the parameters: weights of the model
    # state_dict1 = load_sft("./FLUX-research/FLUX-1.dev/transformer/diffusion_pytorch_model-00002-of-00003.safetensors", device=str(device))
    # state_dict2 = load_sft("./FLUX-research/FLUX-1.dev/transformer/diffusion_pytorch_model-00003-of-00003.safetensors", device=str(device))

    # state_dict.update(state_dict1)
    # state_dict.update(state_dict2)

    # del state_dict1, state_dict2

    print(f"Loaded state dict with {len(state_dict)} keys.")
    # state_dict = optionally_expand_state_dict(model, state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    
    if verbose:
        print_load_warning(missing, unexpected)

    # if config.lora_repo_id is not None and config.lora_filename is not None:
    #     print("Loading LoRA")
    #     lora_path = str(get_checkpoint_path(config.lora_repo_id, config.lora_filename, "FLUX_LORA"))
    #     lora_sd = load_sft(lora_path, device=str(device))
    #     # loading the lora params + overwriting scale values in the norms
    #     missing, unexpected = model.load_state_dict(lora_sd, strict=False, assign=True)
    #     if verbose:
    #         print_load_warning(missing, unexpected)

    return model











def load_flow_model_master(name: str, device: str | torch.device = "cuda", verbose: bool = True, path: str = "./FLUX-research/FLUX-1.dev/transformer/diffusion_pytorch_model-00001-of-00003.safetensors") -> Flux:
    # Loading Flux
    print("Init model")
    config = configs[name]
    
    print(f"Loading model {name} with parameters: {config.params}")


    # checkpoint_path = str(get_checkpoint_path(config.repo_id, config.repo_flow, "FLUX_MODEL"))

    checkpoint_path = str(path)

    with torch.device("meta"):
        if config.lora_repo_id is not None and config.lora_filename is not None:
            # model = FluxLoraWrapper(params=config.params).to(torch.bfloat16)
            pass
        else:
            model = Flux_master(config.params).to(torch.bfloat16)

    # print(f"Loading checkpoint: {checkpoint_path}")
    # # load_sft doesn't support torch.device
    state_dict = load_sft(checkpoint_path, device=str(device))      # state dict has all the parameters: weights of the model
    # state_dict1 = load_sft("./FLUX-research/FLUX-1.dev/transformer/diffusion_pytorch_model-00002-of-00003.safetensors", device=str(device))
    # state_dict2 = load_sft("./FLUX-research/FLUX-1.dev/transformer/diffusion_pytorch_model-00003-of-00003.safetensors", device=str(device))

    # state_dict.update(state_dict1)
    # state_dict.update(state_dict2)

    # del state_dict1, state_dict2

    print(f"Loaded state dict with {len(state_dict)} keys.")
    # state_dict = optionally_expand_state_dict(model, state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    
    if verbose:
        print_load_warning(missing, unexpected)

    # if config.lora_repo_id is not None and config.lora_filename is not None:
    #     print("Loading LoRA")
    #     lora_path = str(get_checkpoint_path(config.lora_repo_id, config.lora_filename, "FLUX_LORA"))
    #     lora_sd = load_sft(lora_path, device=str(device))
    #     # loading the lora params + overwriting scale values in the norms
    #     missing, unexpected = model.load_state_dict(lora_sd, strict=False, assign=True)
    #     if verbose:
    #         print_load_warning(missing, unexpected)

    return model







def load_flow_model_DoubleStream(name: str, device: str | torch.device = "cuda", verbose: bool = True, path: str = "./FLUX-research/FLUX-1.dev/transformer/diffusion_pytorch_model-00001-of-00003.safetensors") -> Flux:
    # Loading Flux
    print("Init model")
    config = configs[name]
    
    print(f"Loading model {name} with parameters: {config.params}")


    # checkpoint_path = str(get_checkpoint_path(config.repo_id, config.repo_flow, "FLUX_MODEL"))

    checkpoint_path = str(path)

    with torch.device("meta"):
        if config.lora_repo_id is not None and config.lora_filename is not None:
            # model = FluxLoraWrapper(params=config.params).to(torch.bfloat16)
            pass
        else:
            model = Flux(config.params).to(torch.bfloat16)

    print(f"Loading checkpoint: {checkpoint_path}")
    # load_sft doesn't support torch.device

    state_dict = load_sft(checkpoint_path, device=str(device))      # state dict has all the parameters: weights of the model
    # state_dict1 = load_sft("./FLUX-research/FLUX-1.dev/transformer/diffusion_pytorch_model-00002-of-00003.safetensors", device=str(device))
    # state_dict2 = load_sft("./FLUX-research/FLUX-1.dev/transformer/diffusion_pytorch_model-00003-of-00003.safetensors", device=str(device))

    # state_dict.update(state_dict1)
    # state_dict.update(state_dict2)

    # del state_dict1, state_dict2

    print(f"Loaded state dict with {len(state_dict)} keys.")
    # state_dict = optionally_expand_state_dict(model, state_dict)

    # only keep enteries that have double in them

    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    
    if verbose:
        print_load_warning(missing, unexpected)

    # if config.lora_repo_id is not None and config.lora_filename is not None:
    #     print("Loading LoRA")
    #     lora_path = str(get_checkpoint_path(config.lora_repo_id, config.lora_filename, "FLUX_LORA"))
    #     lora_sd = load_sft(lora_path, device=str(device))
    #     # loading the lora params + overwriting scale values in the norms
    #     missing, unexpected = model.load_state_dict(lora_sd, strict=False, assign=True)
    #     if verbose:
    #         print_load_warning(missing, unexpected)

    return model




def load_flow_model_SingleStream(name: str, device: str | torch.device = "cuda", verbose: bool = True, path: str = "./FLUX-research/FLUX-1.dev/transformer/diffusion_pytorch_model-00001-of-00003.safetensors") -> Flux:
    # Loading Flux
    print("Init model")
    config = configs[name]
    
    print(f"Loading model {name} with parameters: {config.params}")


    # checkpoint_path = str(get_checkpoint_path(config.repo_id, config.repo_flow, "FLUX_MODEL"))

    checkpoint_path = str(path)

    with torch.device("meta"):
        if config.lora_repo_id is not None and config.lora_filename is not None:
            # model = FluxLoraWrapper(params=config.params).to(torch.bfloat16)
            pass
        else:
            model = Flux(config.params).to(torch.bfloat16)

    print(f"Loading checkpoint: {checkpoint_path}")
    # load_sft doesn't support torch.device
    state_dict = load_sft(checkpoint_path, device=str(device))      # state dict has all the parameters: weights of the model
    # state_dict1 = load_sft("./FLUX-research/FLUX-1.dev/transformer/diffusion_pytorch_model-00002-of-00003.safetensors", device=str(device))
    # state_dict2 = load_sft("./FLUX-research/FLUX-1.dev/transformer/diffusion_pytorch_model-00003-of-00003.safetensors", device=str(device))

    # state_dict.update(state_dict1)
    # state_dict.update(state_dict2)

    # del state_dict1, state_dict2

    # keep only the keys that have single_block in them
    state_dict = {k: v for k, v in state_dict.items() if "single_blocks" in k}

    print(f"Loaded state dict with {len(state_dict)} keys.")
    state_dict = optionally_expand_state_dict(model, state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    
    if verbose:
        print_load_warning(missing, unexpected)

    if config.lora_repo_id is not None and config.lora_filename is not None:
        print("Loading LoRA")
        lora_path = str(get_checkpoint_path(config.lora_repo_id, config.lora_filename, "FLUX_LORA"))
        lora_sd = load_sft(lora_path, device=str(device))
        # loading the lora params + overwriting scale values in the norms
        missing, unexpected = model.load_state_dict(lora_sd, strict=False, assign=True)
        if verbose:
            print_load_warning(missing, unexpected)

    return model












def load_t5(device: str | torch.device = "cuda", max_length: int = 512) -> HFEmbedder:
    # max length 64, 128, 256 and 512 should work (if your sequence is short enough)
    return HFEmbedder("google/t5-v1_1-xxl", max_length=max_length, torch_dtype=torch.bfloat16).to(device)


def load_clip(device: str | torch.device = "cuda") -> HFEmbedder:
    return HFEmbedder("openai/clip-vit-large-patch14", max_length=77, torch_dtype=torch.bfloat16).to(device)


def load_ae(name: str, path: str, device: str | torch.device = "cuda") -> AutoEncoder:
    config = configs[name]
    ckpt_path = str(get_checkpoint_path(config.repo_id, config.repo_ae, "FLUX_AE"))
    ckpt_path = path if path is not None else ckpt_path

    # Loading the autoencoder
    print("Init AE")
    with torch.device("meta"):
        ae = AutoEncoder(config.ae_params)

    print(f"Loading AE checkpoint: {ckpt_path}")
    sd = load_sft(ckpt_path, device=str(device))
    missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)
    print_load_warning(missing, unexpected)
    return ae



























# NSFW_THRESHOLD = 0.85


# TODO for now we will make 2 denoise functions, one for DoubleStream and one for SingleStream
# but later we can merge them into one with a flag for the type of model
# or later a new network class can be introduced to load balance the denoising between the two streams
# Maybe we need seperate implimentations of classes in a way that no other things are instanciated when the server has assumed the role
# should we use api calls or stream the data to the servers?

def denoise(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    # extra img tokens (channel-wise)
    img_cond: Tensor | None = None,
    # extra img tokens (sequence-wise)
    img_cond_seq: Tensor | None = None,
    img_cond_seq_ids: Tensor | None = None,
):
    # this is ignored for schnell
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        img_input = img
        img_input_ids = img_ids
        if img_cond is not None:
            img_input = torch.cat((img, img_cond), dim=-1)
        if img_cond_seq is not None:
            assert (
                img_cond_seq_ids is not None
            ), "You need to provide either both or neither of the sequence conditioning"
            img_input = torch.cat((img_input, img_cond_seq), dim=1)
            img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)
        
        pred = model(
            img=img_input,
            img_ids=img_input_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
        )

        if img_input_ids is not None:
            pred = pred[:, : img.shape[1]]

        img = img + (t_prev - t_curr) * pred

    return img




def denoise_over_network(
    server_list: list,
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    # extra img tokens (channel-wise)
    img_cond: Tensor | None = None,
    # extra img tokens (sequence-wise)
    img_cond_seq: Tensor | None = None,
    img_cond_seq_ids: Tensor | None = None,
):
    # this is ignored for schnell
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        img_input = img
        img_input_ids = img_ids
        if img_cond is not None:
            img_input = torch.cat((img, img_cond), dim=-1)
        if img_cond_seq is not None:
            assert (
                img_cond_seq_ids is not None
            ), "You need to provide either both or neither of the sequence conditioning"
            img_input = torch.cat((img_input, img_cond_seq), dim=1)
            img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)
        
        pred = model(
            img=img_input,
            img_ids=img_input_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
        )

        if img_input_ids is not None:
            pred = pred[:, : img.shape[1]]

        img = img + (t_prev - t_curr) * pred

    return img





def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        num_samples,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        dtype=dtype,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    ).to(device)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()

def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str]) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )





@dataclass
class SamplingOptions:
    prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None


def parse_prompt(options: SamplingOptions) -> SamplingOptions | None:
    user_question = "Next prompt (write /h for help, /q to quit and leave empty to repeat):\n"
    usage = (
        "Usage: Either write your prompt directly, leave this field empty "
        "to repeat the prompt or write a command starting with a slash:\n"
        "- '/w <width>' will set the width of the generated image\n"
        "- '/h <height>' will set the height of the generated image\n"
        "- '/s <seed>' sets the next seed\n"
        "- '/g <guidance>' sets the guidance (flux-dev only)\n"
        "- '/n <steps>' sets the number of steps\n"
        "- '/q' to quit"
    )

    while (prompt := input(user_question)).startswith("/"):
        if prompt.startswith("/w"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, width = prompt.split()
            options.width = 16 * (int(width) // 16)
            print(
                f"Setting resolution to {options.width} x {options.height} "
                f"({options.height * options.width / 1e6:.2f}MP)"
            )
        elif prompt.startswith("/h"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, height = prompt.split()
            options.height = 16 * (int(height) // 16)
            print(
                f"Setting resolution to {options.width} x {options.height} "
                f"({options.height * options.width / 1e6:.2f}MP)"
            )
        elif prompt.startswith("/g"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, guidance = prompt.split()
            options.guidance = float(guidance)
            print(f"Setting guidance to {options.guidance}")
        elif prompt.startswith("/s"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, seed = prompt.split()
            options.seed = int(seed)
            print(f"Setting seed to {options.seed}")
        elif prompt.startswith("/n"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, steps = prompt.split()
            options.num_steps = int(steps)
            print(f"Setting number of steps to {options.num_steps}")
        elif prompt.startswith("/q"):
            print("Quitting")
            return None
        else:
            if not prompt.startswith("/h"):
                print(f"Got invalid command '{prompt}'\n{usage}")
            print(usage)
    if prompt != "":
        options.prompt = prompt
    return options

def save_image(    # removed nsfw checks to simplify and also removed the watermark embedding
    # nsfw_classifier,
    name: str,
    output_name: str,
    idx: int,
    x: torch.Tensor,
    add_sampling_metadata: bool,
    prompt: str,
    nsfw_threshold: float = 0.85,
    track_usage: bool = False,
) -> int:
    fn = output_name.format(idx=idx)
    print(f"Saving {fn}")
    # bring into PIL format and save
    x = x.clamp(-1, 1)
    # x = embed_watermark(x.float())
    x = rearrange(x[0], "c h w -> h w c")

    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
    # if nsfw_classifier is not None:
    #     nsfw_score = [x["score"] for x in nsfw_classifier(img) if x["label"] == "nsfw"][0]
    # else:
    #     nsfw_score = nsfw_threshold - 1.0

    # if nsfw_score < nsfw_threshold:
    #     exif_data = Image.Exif()
    #     if name in ["flux-dev", "flux-schnell"]:
    #         exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
    #     else:
    #         exif_data[ExifTags.Base.Software] = "AI generated;img2img;flux"
    #     exif_data[ExifTags.Base.Make] = "Black Forest Labs"
    #     exif_data[ExifTags.Base.Model] = name
    #     if add_sampling_metadata:
    #         exif_data[ExifTags.Base.ImageDescription] = prompt
    #     img.save(fn, exif=exif_data, quality=95, subsampling=0)
    #     if track_usage:
    #         track_usage_via_api(name, 1)
    #     idx += 1
    # else:
    #     print("Your generated image may contain NSFW content.")
    img.save(fn, quality=95, subsampling=0)

    return idx


@torch.inference_mode()
def main(
    server_mode: bool = False, # false for DoubleStream true for SingleStream
    masterORworker: int = 1, # 1 for master, 2 for worker
    name: str = "flux-dev",
    # name: str = "flux-schnell",
    width: int = 1360,
    height: int = 768,
    seed: int | None = None,
    prompt: str = (
        "a photo of a forest with mist swirling around the tree trunks. The word "
        '"FLUX" is painted over it in big, red brush strokes with visible texture'
    ),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_steps: int | None = None,
    loop: bool = False,
    guidance: float = 2.5,
    offload: bool = False,
    output_dir: str = "output",
    add_sampling_metadata: bool = True,
    trt: bool = False,
    trt_transformer_precision: str = "bf16",
    track_usage: bool = False,
):
    """
    Sample the flux model. Either interactively (set `--loop`) or run for a
    single image.

    Args:
        name: Name of the model to load
        height: height of the sample in pixels (should be a multiple of 16)
        width: width of the sample in pixels (should be a multiple of 16)
        seed: Set a seed for sampling
        output_name: where to save the output image, `{idx}` will be replaced
            by the index of the sample
        prompt: Prompt used for sampling
        device: Pytorch device
        num_steps: number of sampling steps (default 4 for schnell, 50 for guidance distilled)
        loop: start an interactive session and sample multiple times
        guidance: guidance value used for guidance distillation
        add_sampling_metadata: Add the prompt to the image Exif metadata
        trt: use TensorRT backend for optimized inference
        trt_transformer_precision: specify transformer precision for inference
        track_usage: track usage of the model for licensing purposes
    """
    if masterORworker == 1:
        print("Running as master")
    else:
        if server_mode:
            print("Running SingleStream server mode")

        if not server_mode:
            print("Running DoubleStream serve mode")


    if masterORworker == 1:

        prompt = prompt.split("|")
        if len(prompt) == 1:
            prompt = prompt[0]
            additional_prompts = None
        else:
            additional_prompts = prompt[1:]
            prompt = prompt[0]

        assert not (
            (additional_prompts is not None) and loop
        ), "Do not provide additional prompts and set loop to True"

        # nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=device)

        if name not in configs:
            available = ", ".join(configs.keys())
            raise ValueError(f"Got unknown model name: {name}, chose from {available}")

        torch_device = torch.device(device)
        if num_steps is None:
            num_steps = 4 if name == "flux-schnell" else 50

        # allow for packing and conversion to latent space
        height = 16 * (height // 16)
        width = 16 * (width // 16)

        output_name = os.path.join(output_dir, "img_{idx}.jpg")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            idx = 0
        else:
            fns = [fn for fn in iglob(output_name.format(idx="*")) if re.search(r"img_[0-9]+\.jpg$", fn)]
            if len(fns) > 0:
                idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
            else:
                idx = 0


    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    print(f"Using device: {torch_device}")

        # t5 = load_t5(torch_device, max_length=256 if name == "flux-schnell" else 512)
        # print(f"Using T5 model: {t5.name_or_path}")
        # clip = load_clip(torch_device)
        # print(f"Using CLIP model: {clip.name_or_path}")
    
    torch_device = torch.device(device)
    model = None

    if masterORworker == 1:
        model = load_flow_model(name, device=torch_device if offload else torch_device)
    else:
        if not server_mode:
            model = load_flow_model_DoubleStream(name, device=torch_device, path="./FLUX-research/FLUX-1.dev/transformer/diffusion_pytorch_model-00001-of-00003.safetensors")
        if server_mode:
            model = load_flow_model_SingleStream(name, device=torch_device, path="./FLUX-research/FLUX-1.dev/transformer/diffusion_pytorch_model-00001-of-00003.safetensors")

    print(model._buffers.items())
    
    # print(f"Using flow model: {model.name_or_path}")
    # ae = load_ae(name, path="./FLUX-research/FLUX-1.dev/vae/diffusion_pytorch_model.safetensors", device="cpu" if offload else torch_device)
    # print(f"Using autoencoder: {ae.name_or_path}")

    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")


    if masterORworker == 1:


        # assulan aur ikhlakan server yahan workers ke connection ka intezar karega
        # lekin ye Pakistan ha to .........

        # we need atleast 2 workers to run the model or infact only 2

        server_list = ["localhost:8000", "localhost:8001"]



        rng = torch.Generator(device="cpu")
        opts = SamplingOptions(
            prompt=prompt,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance=guidance,
            seed=seed,
        )

        if loop:
            opts = parse_prompt(opts)

        while opts is not None:
            if opts.seed is None:
                opts.seed = rng.seed()
            print(f"Generating with seed {opts.seed}:\n{opts.prompt}")
            t0 = time.perf_counter()

            # prepare input
            x = get_noise(
                1,
                opts.height,
                opts.width,
                device=torch_device,
                dtype=torch.bfloat16,
                seed=opts.seed,
            )

            opts.seed = 0
            
            if offload:
                ae = ae.cpu()
                torch.cuda.empty_cache()
                t5, clip = t5.to(torch_device), clip.to(torch_device)
            inp = prepare(t5, clip, x, prompt=opts.prompt)
            timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(name != "flux-schnell"))

            # offload TEs to CPU, load model to gpu
            if offload:
                t5, clip = t5.cpu(), clip.cpu()
                torch.cuda.empty_cache()
                model = model.to(torch_device)

            # denoise initial noise # this function will be the one that sends the workers the data to denoise
            x = denoise_over_network(server_list, model=model, **inp, timesteps=timesteps, guidance=opts.guidance)

            # since we are making it run on a sigle machine we will keep the model on the gpu for faster compute
            # offload model, load autoencoder to gpu
            # if offload:
            #     model.cpu()
            #     torch.cuda.empty_cache()
            #     ae.decoder.to(x.device)

            # decode latents to pixel space
            x = unpack(x.float(), opts.height, opts.width)
            with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
                x = ae.decode(x)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            fn = output_name.format(idx=idx)
            print(f"Done in {t1 - t0:.1f}s. Saving {fn}")

            idx = save_image(
                name, output_name, idx, x, add_sampling_metadata, prompt, track_usage=track_usage
            )

            if loop:
                print("-" * 80)
                opts = parse_prompt(opts)
            elif additional_prompts:
                next_prompt = additional_prompts.pop(0)
                opts.prompt = next_prompt
            else:
                opts = None


def main1(
    input_mode: int = 0,  # 0 for DoubleStream, 1 for SingleStream
    masterORworker: int = 1,  # True for master, False for worker
    offload: bool = False,

):
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None

    if masterORworker == 1:

        print("Running as master")
        
        rng = torch.Generator(device="cpu")
        
        opts = SamplingOptions(
            prompt=prompt,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance=guidance,
            seed=seed,
        )
        
        height = 16 * (height // 16)
        width = 16 * (width // 16)

        model = load_flow_model(path="", device=torch_device if offload else torch_device)
        
        server = MasterServer()
        server.host = "localhost"
        server.port = 8000
        # server.start_to_accept_connections()
        server.run()

    else:
        if input_mode == 0:
            worker = WorkerServer()
            worker.host = "d06d7ae0858e.ngrok-free.app"
            worker.port = 443
            worker.establish_socket_stream_connection()
            worker.listen_for_tasks()
            print("Running in Double Stream mode")
            # main(server_mode=False, masterORworker=masterORworker)
        else:
            worker = WorkerServer()
            worker.host = "d06d7ae0858e.ngrok-free.app"
            worker.port = 443
            worker.establish_socket_stream_connection()
            worker.listen_for_tasks()
            print("Running in Single Stream mode")
            # main(server_mode=True, masterORworker=masterORworker)


if __name__ == "__main__":
    print("Welcome to the FLUX sampling script!")
    print("1. Master\n 2. Worker")
    masterORworker = int(input("|\n -> ").strip())
    input_mode = 0

    if masterORworker == 1:
        print("Running as master")
    else:
        print("Running as worker")
        print("select mode:\n\t0. Double Stream\n\t1. Single Stream")
        input_mode = int(input("|\n -> ").strip())

    # main(input_mode, masterORworker == 1)
    main1(input_mode, masterORworker)
