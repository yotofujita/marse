import torch
from torch import nn 
from einops import  repeat, rearrange
import math
import numpy as np
from .architectures import Conformer
from .io_representations import DACRepresentation
import tqdm
from .modules.supervised_single_channel_se_module import SupervisedSingleChannelSEModule, SingleChannelSEOutput
from .modules.datasets.labeled_single_channel_se_dataset import LabeledSingleChannelSESample


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


def remainder_to_mult(num, mult):
    return (mult - num % mult) % mult

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs


def FeedForward(*, dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )

class C_NAR_Model(SupervisedSingleChannelSEModule):
    def __init__(
        self,
        *,
        input_dim,
        dim,
        max_seq_len,
        dac_cfg: dict,
        conformer_cfg: dict,
        device="cuda"
    ):
        super().__init__()

        self.dac_representation = DACRepresentation(**dac_cfg, device=device)
        self.device = device

        self.max_seq_len = max_seq_len
        self.dim = dim
        self.input_dim=input_dim

        self.spatial_pos_emb = nn.Embedding(max_seq_len + 1,self.dim)
        self.noise_transformer = Conformer(**conformer_cfg, causal=False)

        self.input_layer  = nn.Linear(self.input_dim, self.dim)
        self.output_layer = nn.Linear(self.dim, self.input_dim)

    def _forward(
        self, 
        noisy_embeds: torch.Tensor,
        clean_embeds: torch.Tensor = None,
        return_loss: bool = False
    ):
        tokens_noisy = self.input_layer(noisy_embeds)
        
        out = self.noise_transformer(tokens_noisy)
        out_embeds = self.output_layer(out)
        
        if not return_loss :
            return out_embeds
        
        preds = out_embeds
        mse_loss = torch.nn.functional.mse_loss(preds, clean_embeds)
        return mse_loss, {"mse_loss": mse_loss}

    # ===== SingleChannelSEModel interface =====

    def forward(self, batch: LabeledSingleChannelSESample):
        noisy_wav = batch['noisy_wav'].to(self.device)
        clean_wav = batch['clean_wav'].to(self.device)

        noisy_embeds = self.dac_representation.encode(noisy_wav)   # (B, D, T)
        clean_embeds = self.dac_representation.encode(clean_wav)   # (B, D, T)
        noisy_embeds = rearrange(noisy_embeds, "b d t -> b t d")
        clean_embeds = rearrange(clean_embeds, "b d t -> b t d")

        loss, loss_dict = self._forward(
            noisy_embeds=noisy_embeds,
            clean_embeds=clean_embeds,
            return_loss=True,
        )
        return loss, loss_dict

    @torch.no_grad()
    def separate_batch(self, batch: LabeledSingleChannelSESample) -> SingleChannelSEOutput:
        noisy_wav = batch['noisy_wav'].to(self.device)
        clean_wav = batch['clean_wav'].to(self.device)
        name = batch['metadata']['name']

        noisy_embeds = self.dac_representation.encode(noisy_wav)
        clean_embeds = self.dac_representation.encode(clean_wav)
        noisy_embeds = rearrange(noisy_embeds, "b d t -> b t d")
        clean_embeds = rearrange(clean_embeds, "b d t -> b t d")

        generated = self._forward(
            noisy_embeds=noisy_embeds,
            clean_embeds=None,
            return_loss=False,
        )  # (B, T, D)

        generated = rearrange(generated, "b t d -> b d t")
        noisy_embeds = rearrange(noisy_embeds, "b t d -> b d t")
        clean_embeds = rearrange(clean_embeds, "b t d -> b d t")

        reconstructed = self.dac_representation.decode(generated)   # (B, T)
        noisy = self.dac_representation.decode(noisy_embeds)   # (B, T)
        clean = self.dac_representation.decode(clean_embeds)   # (B, T)

        return {
            'reconstructed': reconstructed,
            'noisy': noisy,
            'clean': clean,
            'metadata': {
                'name': name
            }
        }


if __name__ == "__main__":

    import soundfile as sf
    import torchaudio

    module = C_NAR_Model(
        input_dim=1024,
        dim=384,
        max_seq_len=50,
        N_layers=16,
        dim_head=32,
        heads=12,
        attn_dropout=0.0,
        ff_mult=4,
        ff_dropout=0.0,
        device="cuda"
    )

    module.to("cuda")
    module.eval()

    batch = {
        'noisy_wav': torchaudio.load("io_representations/waveform.wav")[0].unsqueeze(0),
        'clean_wav': torchaudio.load("io_representations/waveform.wav")[0].unsqueeze(0),
        'name': 'waveform'
    }

    loss = module.compute_loss_from_batch(batch)
    print(loss)
    output = module.separate_batch(batch)
    print(output['reconstructed'].shape)
    print(output['noisy'].shape)
    print(output['clean'].shape)
    sf.write("reconstructed.wav", output['reconstructed'].detach().cpu().numpy().squeeze(), 16000)
    sf.write("noisy.wav", output['noisy'].detach().cpu().numpy().squeeze(), 16000)
    sf.write("clean.wav", output['clean'].detach().cpu().numpy().squeeze(), 16000)