"""Abstract dataset classes for source separation."""

from torch.utils.data import Dataset

from typing import Tuple
import torch

import torchaudio
import random
import librosa
import tqdm


class CustomDatasetClass(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def transfer_to_ssd(self, data_list, dataset_name):
        import os
        import shutil

        tmpdir = os.environ.get("TMPDIR", None)
        if tmpdir is None:
            return data_list
        
        print(f"Transferring {dataset_name} to SSD...")
        
        new_data_list = []
        for i in range(len(data_list[0])):
            new_data_list_i = []

            if not os.path.exists(data_list[0][i]):
                for data in data_list:
                    new_data_list_i.append(data[i])
                new_data_list.append(new_data_list_i)
                continue

            new_key_dir = os.path.join(tmpdir, f"data_{i}")
            os.makedirs(new_key_dir, exist_ok=True)

            for data in tqdm.tqdm(data_list, desc=f"Transferring data_{i} to SSD"):
                new_data_path = os.path.join(new_key_dir, os.path.basename(data[i]))
                if not os.path.exists(new_data_path):
                    shutil.copy2(data[i], new_data_path)
                new_data_list_i.append(new_data_path)
            new_data_list.append(new_data_list_i)

        new_data_list = list(zip(*new_data_list))
        print(f"Transferred {dataset_name} to SSD ({tmpdir})")
        return new_data_list


LabeledSingleChannelSESample = {
    'noisy_wav': torch.Tensor,
    'clean_wav': torch.Tensor,
    'metadata': dict
}

class LabeledSingleChannelSEDataset(CustomDatasetClass):
    """
    Dataset for labeled single-channel single-source speech enhancement
    """
    
    def __init__(self, labeled_data_dirs, seq_len, random_start=True):
        """
        Args:
            labeled_data_dirs: List of paths to labeled data directories
            seq_len: Sequence length
            random_start: Whether to use random start
        """
        self.labeled_data_list = []
        for labeled_data_dir in labeled_data_dirs:
            noisy_wav_list = librosa.util.find_files(labeled_data_dir['noisy_dir'], ext='wav')
            clean_wav_list = librosa.util.find_files(labeled_data_dir['clean_dir'], ext='wav')
            self.labeled_data_list.extend(list(zip(
                noisy_wav_list, 
                clean_wav_list, 
                [f"{labeled_data_dir['name']}_{filename.split('/')[-1].split('.')[0]}" for filename in noisy_wav_list])))
        self.seq_len = seq_len
        self.random_start = random_start

        self.labeled_data_list = self.transfer_to_ssd(self.labeled_data_list, "Labeled SE Dataset")
    
    def __len__(self):
        return len(self.labeled_data_list)
    
    def __getitem__(self, idx) -> LabeledSingleChannelSESample:
        """
        Returns:
            LabeledSingleChannelSESample: Sample for labeled single-channel single-source speech enhancement
            {
                'noisy_wav': torch.Tensor,
                'clean_wav': torch.Tensor,
                'name': str
            }
        """
        noisy_wav_path, clean_wav_path, name = self.labeled_data_list[idx]

        noisy_wav, _ = torchaudio.load(noisy_wav_path)
        clean_wav, _ = torchaudio.load(clean_wav_path)

        if self.seq_len is not None:
            if self.random_start:
                start = random.randint(0, noisy_wav.shape[1] - self.seq_len)
            else:
                start = 0
            stop = start + self.seq_len
            
        else:
            start = 0
            stop = None
        
        noisy_wav = noisy_wav[:,start:stop]
        clean_wav = clean_wav[:,start:stop]
        
        return {
            'noisy_wav': noisy_wav,
            'clean_wav': clean_wav,
            'metadata': {
                'name': name
            }
        }


if __name__ == "__main__":
    labeled_data_dirs = [
        {
            'noisy_dir': '/home/yfujita/datasets/Libri2Mix/wav16k/min/train-360/mix_single/',
            'clean_dir': '/home/yfujita/datasets/Libri2Mix/wav16k/min/train-360/s1/',
            'name': 'Libri2Mix_train-360'
        }
    ]
    dataset = LabeledSingleChannelSEDataset(labeled_data_dirs, 16000)
    print(len(dataset))
    print(dataset[0]['noisy_wav'].shape)
    print(dataset[0]['clean_wav'].shape)
    print(dataset[0]['metadata']['name'])