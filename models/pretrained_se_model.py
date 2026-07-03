import torch
import torchaudio
import soundfile as sf
# from asteroid.models import BaseModel
from .modules.supervised_single_channel_se_module import SingleChannelSEOutput
from .modules.datasets.labeled_single_channel_se_dataset import LabeledSingleChannelSESample


class Pretrained_Model(torch.nn.Module):
    def __init__(self, *, model_id, device="cuda", **kwargs):
        super().__init__()

        self.device = device
        
        self.model = BaseModel.from_pretrained(model_id).to(device)
        self.model.eval()

    @torch.no_grad()
    def separate_batch_long(self, batch: LabeledSingleChannelSESample) -> SingleChannelSEOutput:
        noisy_wav = batch['noisy_wav'].to(self.device)
        clean_wav = batch['clean_wav'].to(self.device)
        name = batch['metadata']['name']

        with torch.no_grad():
            generated_wav = self.model(noisy_wav)
        
        rms_in = torch.sqrt(torch.mean( ** 2) + EPS)
        rms_out = torch.sqrt(torch.mean(y ** 2) + EPS)
        y = y * (rms_in / rms_out)

        return {
            'reconstructed': generated_wav,
            'noisy': noisy_wav,
            'clean': clean_wav,
            'metadata': {
                'name': name
            }
        }