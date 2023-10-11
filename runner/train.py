import torch
import yaml
import os
import torch.nn as nn

from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.tools import to_device, log
from data.dataset import TextMelDataset, TextMelCollate
from model.fastspeech2 import FastSpeech2
from model.loss import FastSpeech2Loss


class Train:
    def __init__(self):
        self.model_config = yaml.load(open('../config/model.yaml'), Loader=yaml.FullLoader)
        self.preprocess_config = yaml.load(open('../config/preprocessing.yaml'), Loader=yaml.FullLoader)
        self.train_config = yaml.load(open('../config/train.yaml'), Loader=yaml.FullLoader)

        os.makedirs(self.train_config['path']['ckpt_path'], exist_ok=True)
        os.makedirs(self.train_config['path']['log_path'], exist_ok=True)
        os.makedirs(self.train_config['path']['result_path'], exist_ok=True)

    def prepare_data_loader(self):
        trainset = TextMelDataset(self.training_files, self.hp)
        valset = TextMelDataset(self.validation_files, self.hp)
        collate_fn = TextMelCollate(self.n_frames_per_step)

        train_loader = DataLoader(trainset,
                                  shuffle=True,
                                  batch_size=self.batch_size,
                                  pin_memory=False,
                                  collate_fn=collate_fn,
                                  )

        return train_loader, valset, collate_fn


    def to_gpu(self, x):
        x = x.contiguous()
        if torch.cuda.is_available():
            x = x.cuda(non_blocking=True)
        return x


    def batch_to_gpu(self, batch):
        text_padded, text_lengths, mel_specgram_padded, mel_specgram_lengths, gate_padded = batch
        text_padded = self.to_gpu(text_padded).long()
        text_lengths = self.to_gpu(text_lengths).long()
        mel_specgram_padded = self.to_gpu(mel_specgram_padded).float()
        gate_padded = self.to_gpu(gate_padded).float()
        mel_specgram_lengths = self.to_gpu(mel_specgram_lengths).long()
        x = (text_padded, text_lengths, mel_specgram_padded, mel_specgram_lengths)
        y = (mel_specgram_padded, gate_padded)
        return x, y

    def run(self):
        fastspeech2 = FastSpeech2(self.preprocess_config, self.model_config)

        if self.train_config['path']['pretrained']:
            pretrained_weight = torch.load(self.train_config['path']['pretrained_path'])
            fastspeech2.load_state_dict(pretrained_weight['model'])

        # train_loader, valset, collate_fn = self.prepare_data_loader()
        #
        # optimizer = torch.optim.Adam(fastspeech2.parameters(), betas=self.betas, eps=self.eps, weight_decay=self.weight_decay)
        # criterion = FastSpeech2Loss()
        #
        # for epoch in range(self.epochs):
        #     fastspeech2.train()
        #
        #     for i, batch in enumerate(tqdm(train_loader, desc=f'[Train Epoch ==> {epoch}/{self.epochs}]: ')):
        #         (text_padded, text_lengths, mel_specgram_padded, mel_specgram_lengths), y = self.batch_to_gpu(batch)
        #         y_pred = fastspeech2(text_padded, )
        #
        #         loss = criterion(y_pred, y)


if __name__ == '__main__':
    a = Train()
    a.run()
