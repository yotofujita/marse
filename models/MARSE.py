import math
import torch
from torch import nn
from einops import rearrange, repeat

try:
    from .architectures import Conformer
    from .io_representations import DACRepresentation
    from .modules.supervised_single_channel_se_module import SupervisedSingleChannelSEModule, SingleChannelSEOutput
    from .modules.datasets.labeled_single_channel_se_dataset import LabeledSingleChannelSESample
except ImportError:
    from architectures import Conformer
    from io_representations import DACRepresentation
    from modules.supervised_single_channel_se_module import SupervisedSingleChannelSEModule, SingleChannelSEOutput
    from modules.datasets.labeled_single_channel_se_dataset import LabeledSingleChannelSESample


def plot_mask_and_masked_clean(mask, masked_clean, mask_embeds):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.title(f"Batch {i}, mask ratio {mask[i].sum().item() / mask[i].numel()}")
        plt.plot(mask[i].cpu().numpy())
    plt.tight_layout()
    plt.savefig("models/mask.png")
    plt.close()
    plt.figure(figsize=(10, 5))
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.title(f"Batch {i}, masked clean")
        plt.plot((masked_clean == mask_embeds).cpu().numpy()[i])
    plt.tight_layout()
    plt.savefig("models/masked_clean.png")
    plt.close()


def plot_unmasking_process(mask_list, score_list, err_list, mean_errs, policy):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(mask_list) // 2 + 1, len(mask_list[0]), figsize=(30, 20))
    for i in range(len(mask_list)):
        if (i+1) % 2 == 0 or i == 0:
            for j in range(len(mask_list[i])):
                axes[(i+1) // 2, j].set_title(f"b={j}, i={i}, mean_err={mean_errs[i][j]:.2f}")
                axes[(i+1) // 2, j].plot(mask_list[i][j].cpu().numpy())
                axes[(i+1) // 2, j].plot(err_list[i][j].cpu().numpy())
                axes[(i+1) // 2, j].plot(score_list[i][j].cpu().numpy())
                axes[(i+1) // 2, j].set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(f"models/unmasking_process_{policy}.png")
    plt.close()

def env_flag(name: str, default: bool = False) -> bool:
    import os
    value = os.environ.get(name, None)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


class MARSE_Model(SupervisedSingleChannelSEModule):
    def __init__(
        self,
        *,
        input_dim,
        dim,
        max_seq_len,
        dac_cfg: dict,
        conformer_cfg: dict,
        num_steps: int = 4,
        decoding_policy: str = "random",
        device: str = "cuda",
        debug: bool = False,
        pos_emb: bool = True
    ):
        super().__init__()

        self.dac_representation = DACRepresentation(**dac_cfg, device=device)
        self.device = device
        self.max_seq_len = max_seq_len
        self.dim = dim
        self.input_dim = input_dim
        self.num_steps = num_steps
        self.decoding_policy = decoding_policy
        self.debug = env_flag("DEBUG", default=debug)
        if self.decoding_policy not in {"random", "causal", "oracle"}:
            raise ValueError(
                "decoding_policy must be one of: random, causal, oracle; "
                f"got {self.decoding_policy!r}"
            )

        self.mask_embeds = nn.Parameter(torch.randn(self.input_dim))

        self.noise_transformer = Conformer(**conformer_cfg, causal=False)
        self.input_layer = nn.Linear(self.input_dim, self.dim)
        self.output_layer = nn.Linear(self.dim, self.input_dim)

        self.pos_emb = pos_emb
        if pos_emb:
            self.spatial_pos_emb = nn.Embedding(int(2 * max_seq_len) + 1, self.dim)
            self.spatial_start_token = nn.Parameter(torch.randn(self.dim)) # marker for the start of clean tokens

    def _cosine_mask_ratio(self, batch_size: int, device: torch.device) -> torch.Tensor:
        u = torch.rand(batch_size, device=device)  # [0,1) uniformly distributed
        return torch.cos(u * math.pi / 2)  # (0,1] cosine function

    def _sample_mask(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        mask_ratio = self._cosine_mask_ratio(batch_size, device)  # (0,1] cosine function
        mask_num = torch.clamp((mask_ratio * seq_len).int(), min=1)  # number of masked frames (at least 1 frame)
        mask = torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)
        for b in range(batch_size):
            mask[b, torch.randperm(seq_len)[:mask_num[b]]] = 1  # randomly mask the mask_num[b] frames
        return mask

    def _apply_mask(self, clean_embeds: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked = clean_embeds.clone()
        masked[mask] = self.mask_embeds.to(clean_embeds.dtype)
        return masked

    def _forward(
        self,
        noisy_embeds,
        masked_clean,
        return_loss: bool = False,
        clean_embeds_gt=None,
        mask=None,
    ):
        tokens_noisy = self.input_layer(noisy_embeds)
        tokens_clean = self.input_layer(masked_clean)

        if self.pos_emb:
            spatial_pos = self.spatial_pos_emb(
                torch.arange(tokens_noisy.shape[1]+tokens_clean.shape[1]+1, device=self.device)
            )
            spatial_tokens = torch.cat((
                tokens_noisy,
                repeat(self.spatial_start_token, 'f -> b 1 f', b = noisy_embeds.shape[0]),
                tokens_clean
            ), dim = 1)
            spatial_tokens = spatial_tokens + spatial_pos
        else:
            spatial_tokens = torch.cat((tokens_noisy, tokens_clean), dim = 1)
        
        out = self.noise_transformer(spatial_tokens)

        pred_clean = self.output_layer(out[:, -masked_clean.shape[1]:, :])

        if not return_loss:
            return pred_clean

        mse_loss = torch.nn.functional.mse_loss(pred_clean[mask], clean_embeds_gt[mask])
        loss = 0.5 * mse_loss
        return loss, {"mse_loss": mse_loss}

    def _compute_policy_scores(
        self,
        *,
        policy: str,
        mask: torch.Tensor,
        pred_clean_quantized: torch.Tensor,
        clean_embeds_quantized: torch.Tensor,
        current: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = mask.shape
        device = mask.device

        err = torch.empty([batch_size, seq_len], device=device)
        err[mask] = ((pred_clean_quantized[mask] - clean_embeds_quantized[mask]) ** 2).mean(dim=-1)
        err[~mask] = ((current[~mask] - clean_embeds_quantized[~mask]) ** 2).mean(dim=-1)
        err_normalized = torch.softmax(err, dim=-1)

        if policy == "random":
            scores = torch.rand(mask.shape, device=device)
        elif policy == "causal":
            scores = torch.arange(seq_len, device=device, dtype=pred_clean_quantized.dtype)
            scores = scores.unsqueeze(0).expand(batch_size, -1)
        elif policy == "oracle":
            scores = err
        else:
            raise ValueError(f"Unknown decoding policy: {policy}")

        scores = scores.masked_fill(~mask, float("-inf"))
        score_normalized = torch.softmax(scores, dim=-1)
        return scores, score_normalized, err_normalized

    def _iterative_unmask(self, noisy_embeds: torch.Tensor, clean_embeds_quantized: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = noisy_embeds.shape
        device = noisy_embeds.device
        current = repeat(self.mask_embeds, "d -> b t d", b=batch_size, t=seq_len).to(device).clone()
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        remaining = seq_len

        if self.debug:
            current_list, mask_list, score_list, err_list, mean_errs = [], [], [], [], []

        for step in range(self.num_steps):
            pred_clean = self._forward(noisy_embeds, current, return_loss=False)
            pred_clean_quantized = self.dac_representation.quantize(
                pred_clean.permute((0, 2, 1))
            ).permute((0, 2, 1))

            if step == self.num_steps - 1:
                remaining = 0
            else:
                ratio = math.cos(math.pi / 2 * ((step + 1) / self.num_steps))
                _remaining = int(round(ratio * seq_len))
                if _remaining >= remaining:
                    remaining = remaining - 1
                else:
                    remaining = _remaining
                remaining = max(0, min(seq_len, remaining))

            if remaining == 0:
                new_mask = torch.zeros_like(mask)
                s_normalized = torch.zeros(mask.shape, device=device)
                err_normalized = torch.zeros(mask.shape, device=device)
                err_for_debug = torch.zeros(mask.shape, device=device)
            else:
                scores, s_normalized, err_normalized = self._compute_policy_scores(
                    policy=self.decoding_policy,
                    mask=mask,
                    pred_clean_quantized=pred_clean_quantized,
                    clean_embeds_quantized=clean_embeds_quantized,
                    current=current,
                )
                err_for_debug = err_normalized

                # Keep the top scoring masked positions for the next iteration.
                topk_idx = scores.topk(k=min(remaining, mask.size(1)), dim=1).indices

                new_mask = torch.zeros_like(mask)
                new_mask.scatter_(1, topk_idx, True)
                new_mask &= mask

            unmask_idx = mask & ~new_mask
            
            if unmask_idx.any():
                current[unmask_idx] = pred_clean_quantized[unmask_idx]

            mask = new_mask

            if self.debug:
                current_list.append(current.clone())
                mask_list.append(new_mask.clone())
                score_list.append(s_normalized.clone())
                err_list.append(err_normalized.clone())
                mean_errs.append(err_for_debug.mean(dim=-1).clone())

        if self.debug:
            return current, current_list, mask_list, score_list, err_list, mean_errs
        else:
            return current

    # ===== SingleChannelSEModel interface =====

    def forward(self, batch: LabeledSingleChannelSESample):
        noisy_wav = batch["noisy_wav"].to(self.device)
        clean_wav = batch["clean_wav"].to(self.device)

        noisy_embeds = self.dac_representation.encode(noisy_wav)
        clean_embeds = self.dac_representation.encode(clean_wav)
        clean_embeds_quantized = self.dac_representation.quantize(clean_embeds)
        noisy_embeds = rearrange(noisy_embeds, "b d t -> b t d")
        clean_embeds = rearrange(clean_embeds, "b d t -> b t d")
        clean_embeds_quantized = rearrange(clean_embeds_quantized, "b d t -> b t d")

        mask = self._sample_mask(clean_embeds_quantized.shape[0], clean_embeds_quantized.shape[1], clean_embeds_quantized.device)
        masked_clean_quantized = self._apply_mask(clean_embeds_quantized, mask)

        if self.debug:
            plot_mask_and_masked_clean(mask, masked_clean_quantized, self.mask_embeds)

        loss, loss_dict = self._forward(
            noisy_embeds,
            masked_clean_quantized,
            return_loss=True,
            clean_embeds_gt=clean_embeds,
            mask=mask,
        )

        if self.debug:
            print(f"Loss: {loss}")
            for key, value in loss_dict.items():
                print(f"{key}: {value}")

        return loss, loss_dict

    @torch.no_grad()
    def separate_batch(self, batch: LabeledSingleChannelSESample) -> SingleChannelSEOutput:
        noisy_wav = batch["noisy_wav"].to(self.device)
        clean_wav = batch["clean_wav"].to(self.device)
        name = batch["metadata"].get("name", "unknown")

        noisy_embeds = self.dac_representation.encode(noisy_wav)
        clean_embeds = self.dac_representation.encode(clean_wav)
        clean_embeds_quantized = self.dac_representation.quantize(clean_embeds)
        noisy_embeds = rearrange(noisy_embeds, "b d t -> b t d")
        clean_embeds = rearrange(clean_embeds, "b d t -> b t d")
        clean_embeds_quantized = rearrange(clean_embeds_quantized, "b d t -> b t d")

        generated = self._iterative_unmask(noisy_embeds, clean_embeds_quantized)
        if self.debug:
            generated, _, mask_list, score_list, err_list, mean_errs = generated
            plot_unmasking_process(mask_list, score_list, err_list, mean_errs, self.decoding_policy)

        generated = rearrange(generated, "b t d -> b d t")
        noisy_embeds = rearrange(noisy_embeds, "b t d -> b d t")
        clean_embeds = rearrange(clean_embeds, "b t d -> b d t")

        reconstructed = self.dac_representation.decode(generated, quantized=True)
        noisy = self.dac_representation.decode(noisy_embeds)
        clean = self.dac_representation.decode(clean_embeds)

        return {
            "reconstructed": reconstructed,
            "noisy": noisy,
            "clean": clean,
            "metadata": {"name": name},
        }


if __name__ == "__main__":
    model = MARSE_Model(
        input_dim=1024,
        dim=384,
        max_seq_len=50,
        num_steps=4,
        device="cuda",
        dac_cfg={
            "dac_model_type": "16khz",
            "n_quantizers": 12,
            "sample_rate": 16000
        },
        conformer_cfg={
            "dim": 384,
            "layers": 16,
        },
        debug=True,
    )
    model.to("cuda")
    model.eval()
    
    batch = {
        "noisy_wav": torch.randn(4, 1, 16000),
        "clean_wav": torch.randn(4, 1, 16000),
        "metadata": {"name": "waveform"}
    }
    loss = model.forward(batch)
    print(loss)
    output = model.separate_batch(batch)
    print(output['reconstructed'].shape)
    print(output['noisy'].shape)
    print(output['clean'].shape)
