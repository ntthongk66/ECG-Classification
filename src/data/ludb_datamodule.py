import os
import random
import math
import wfdb
import numpy as np
import torch
from typing import Optional
import matplotlib.pyplot as plt
# from src.utils.utils import draw_segmentation_timeline

from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from lightning import LightningDataModule


def anom_to_label(anom):
    if anom in {
        'Atrial fibrillation',
        'Atrial flutter, typical'
    }:
        return 1
    else:
        return 0


def load_ludb_tensors(ludb_files, leads=None):
    leads = leads or ['i', 'ii', 'iii', 'avr', 'avl',
                      'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']

    # initialize arrays
    all_waves = []
    all_p = []
    all_qrs = []
    all_t = []
    all_none = []
    all_cls_target = []

    for record_name in ludb_files:
        # load data
        record = wfdb.rdrecord(record_name)
        anom = record.__dict__['comments'][3][8:-1]
        waves = {
            lead: wave
            for lead, wave in zip(
                record.__dict__['sig_name'],
                record.__dict__['p_signal'].T  # shape (12,5000)
            )
        }

        # extract annotation
        for lead in leads:
            wave = waves[lead]  # shape (5000,)
            all_waves.append(wave)

            # read annotation
            annotation = wfdb.rdann(record_name, extension=lead)
            sample = annotation.__dict__['sample']
            symbol = annotation.__dict__['symbol']

            # initialize annotation dictionary
            ann_dct = {
                'p': np.zeros(5000,),
                'qrs': np.zeros(5000,),
                't': np.zeros(5000,),
            }

            # update annotation array
            on = None
            for t, symbol in zip(sample, symbol):
                if symbol == '(':  # symbol denotes onset
                    on = t
                elif symbol == ')':  # symbol denotes offset
                    off = t
                    if on != None:
                        ann_dct[key] += np.array([0]*on +
                                                 [1]*(off-on+1) + [0]*(4999-off))
                        on = None
                else:  # symbol denotes peak
                    if symbol in {'p', 't'}:
                        key = symbol
                    else:
                        assert (symbol == 'N')
                        key = 'qrs'

            # create array indicating non-labeled areas
            assert (np.max(ann_dct['p'] + ann_dct['qrs'] + ann_dct['t']) <= 1)
            ann_dct['none'] = np.ones(
                5000,) - (ann_dct['p'] + ann_dct['qrs'] + ann_dct['t'])

            all_p.append(ann_dct['p'])
            all_qrs.append(ann_dct['qrs'])
            all_t.append(ann_dct['t'])
            all_none.append(ann_dct['none'])
            all_cls_target.append(anom_to_label(anom))

    # finalize arrays
    all_waves = np.array(all_waves)
    all_p = np.array(all_p)
    all_qrs = np.array(all_qrs)
    all_t = np.array(all_t)
    all_none = np.array(all_none)
    all_seg_target = np.stack((all_p, all_qrs, all_t, all_none), axis=1)
    all_cls_target = np.array(all_cls_target)

    # convert to torch tensors
    X_torch = torch.tensor(all_waves, dtype=torch.float32)
    X_torch = X_torch.unsqueeze(dim=1)  # add channel dimension
    y_seg_torch = torch.tensor(all_seg_target, dtype=torch.float32)
    y_cls_torch = torch.tensor(all_cls_target, dtype=torch.int64)

    return X_torch, y_seg_torch, y_cls_torch


def Tnoise_powerline(fs=100, N=1000, C=1, fn=50., K=3, channels=1):
    '''powerline noise inspired by https://ieeexplore.ieee.org/document/43620
    fs: sampling frequency (Hz)
    N: lenght of the signal (timesteps)
    C: relative scaling factor (default scale: 1)
    fn: base frequency of powerline noise (Hz)
    K: number of higher harmonics to be considered
    channels: number of output channels (just rescaled by a global channel-dependent factor)
    '''
    # C *= 0.333 #adjust default scale
    t = torch.arange(0, N/fs, 1./fs)

    signal = torch.zeros(N)
    phi1 = random.uniform(0, 2*math.pi)
    for k in range(1, K+1):
        ak = random.uniform(0, 1)
        signal += C*ak*torch.cos(2*math.pi*k*fn*t+phi1)
    signal = C*signal[:, None]
    if (channels > 1):
        channel_gains = torch.empty(channels).uniform_(-1, 1)
        signal = signal*channel_gains[None]
    return signal


def Tnoise_baseline_wander(fs=100, N=1000, C=1.0, fc=0.5, fdelta=0.01, channels=1, independent_channels=False):
    '''baseline wander as in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5361052/
    fs: sampling frequency (Hz)
    N: lenght of the signal (timesteps)
    C: relative scaling factor (default scale : 1)
    fc: cutoff frequency for the baseline wander (Hz)
    fdelta: lowest resolvable frequency (defaults to fs/N if None is passed)
    channels: number of output channels
    independent_channels: different channels with genuinely different outputs (but all components in phase) instead of just a global channel-wise rescaling
    '''
    if (fdelta is None):  # 0.1
        fdelta = fs/N

    K = int((fc/fdelta)+0.5)
    t = torch.arange(0, N/fs, 1./fs).repeat(K).reshape(K, N)
    k = torch.arange(K).repeat(N).reshape(N, K).T
    phase_k = torch.empty(K).uniform_(0, 2*math.pi).repeat(N).reshape(N, K).T
    a_k = torch.empty(K).uniform_(0, 1).repeat(N).reshape(N, K).T
    pre_cos = 2*math.pi * k * fdelta * t + phase_k
    cos = torch.cos(pre_cos)
    weighted_cos = a_k * cos
    res = weighted_cos.sum(dim=0)
    return C*res


class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, wave, target):
        for t in self.transforms:
            wave, target = t(wave, target)
        return wave, target


class RandomCrop():
    def __init__(self, length, start, end):
        self.length = length
        self.start = start
        self.end = end

    def __call__(self, wave, target):
        start = random.randint(self.start, self.end-self.length)
        end = start + self.length
        return wave[:, start:end], target[:, start:end]


class ChannelResize():
    def __init__(self, magnitude_range=(0.5, 2)):
        self.log_magnitude_range = torch.log(torch.tensor(magnitude_range))

    def __call__(self, wave, target):
        channels, len_wave = wave.shape
        resize_factors = torch.exp(torch.empty(
            channels).uniform_(*self.log_magnitude_range))
        resize_factors = resize_factors.repeat(len_wave).view(wave.T.shape).T
        wave = resize_factors * wave
        return wave, target


class GaussianNoise():
    def __init__(self, prob=1.0, scale=0.01):
        self.scale = scale
        self.prob = prob

    def __call__(self, wave, target):
        if random.random() < self.prob:
            wave += self.scale * torch.randn(wave.shape)
        return wave, target


class BaselineShift():
    def __init__(self, prob=1.0, scale=1.0):
        self.prob = prob
        self.scale = scale

    def __call__(self, wave, target):
        if random.random() < self.prob:
            shift = torch.randn(1)
            wave = wave + self.scale * shift
        return wave, target


class BaselineWander():
    def __init__(self, prob=1.0, freq=500):
        self.freq = freq
        self.prob = prob

    def __call__(self, wave, target):
        if random.random() < self.prob:
            channels, len_wave = wave.shape
            wander = Tnoise_baseline_wander(fs=self.freq, N=len_wave)
            wander = wander.repeat(channels).view(wave.shape)
            wave = wave + wander
        return wave, target


class PowerlineNoise():
    def __init__(self, prob=1.0, freq=500):
        self.freq = freq
        self.prob = prob

    def __call__(self, wave, target):
        if random.random() < self.prob:
            channels, len_wave = wave.shape
            noise = Tnoise_powerline(
                fs=self.freq, N=len_wave, channels=channels).T
            wave = wave + noise
        return wave, target


class CustomTensorDataset(Dataset):
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]

        if self.transform:
            x, y = self.transform(x, y)

        return (x, y) + tuple(tensor[index] for tensor in self.tensors[2:])

    def __len__(self):
        return self.tensors[0].size(0)


class LUDBDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/ludb/data",
        batch_size: int = 32,
        n_ludb_train: int = 100,
        num_workers: int = 0,
        pin_memory: bool = False,
        sampler: bool = False
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.n_ludb_train = n_ludb_train
        self.sampler = sampler

        self.X_train: Optional[Dataset] = None
        self.y_seg_train: Optional[Dataset] = None
        self.y_cls_train: Optional[Dataset] = None
        self.X_test: Optional[Dataset] = None
        self.y_seg_test: Optional[Dataset] = None
        self.y_cls_test: Optional[Dataset] = None

    def prepare_data(self) -> None:

        return super().prepare_data()

    def setup(self, stage: Optional[str] = None):
        ludb_files = [
            os.path.abspath(os.path.join(self.data_dir, p))[:-4]
            for p in os.listdir(self.data_dir)
            if p.endswith(".hea")
        ]
        ludb_files_train = ludb_files[: self.n_ludb_train]
        ludb_files_test = ludb_files[self.n_ludb_train:]
        self.X_train, self.y_seg_train, self.y_cls_train = load_ludb_tensors(
            ludb_files_train)
        self.X_test, self.y_seg_test, self.y_cls_test = load_ludb_tensors(
            ludb_files_test)

    def train_dataloader(self):
        if self.sampler:
            target = self.y_cls_train
            weight = torch.tensor([1.0 / torch.sum(target == t)
                                  for t in torch.unique(target)])
            samples_weight = torch.tensor(
                [weight[int(t)] for t in target]).double()
            sampler = WeightedRandomSampler(
                samples_weight, len(samples_weight))
        else:
            sampler = None
        train_dataset = CustomTensorDataset(
            tensors=(self.X_train, self.y_seg_train, self.y_cls_train),
            transform=Compose(
                [
                    RandomCrop(2000, start=1000, end=4000),
                    BaselineWander(prob=0.2),
                    GaussianNoise(prob=0.2),
                    PowerlineNoise(prob=0.2),
                    ChannelResize(),
                    BaselineShift(prob=0.2),
                ]
            ),
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=(sampler is None), sampler=sampler
        )


        return train_loader

    def test_dataloader(self):
        test_dataset = CustomTensorDataset(
            tensors=(self.X_test, self.y_seg_test, self.y_cls_test))
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False)

        return test_loader

    #! dang dung thu val dataloader xem nhu nao.
    def val_dataloader(self):
        test_dataset = CustomTensorDataset(
            tensors=(self.X_test, self.y_seg_test, self.y_cls_test))
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False)

        return test_loader
    

if __name__ == "__main__":
    dm = LUDBDataModule()
    dm.setup()
    train_loader = dm.train_dataloader()
    test_loader = dm.test_dataloader()

    x, y_seg_tg, y_cls_tg = next(iter(test_loader))

    print(y_seg_tg)

    