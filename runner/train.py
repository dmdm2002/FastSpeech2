import torch
import yaml
import os
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.tools import to_device, log, synth_one_sample
from data.dataset import Dataset
# from data.dataset import TextMelDataset, TextMelCollate
from model.fastspeech2 import FastSpeech2
from model.loss import FastSpeech2Loss
from utils.builder import get_model, get_vocoder, get_param_num


class Train:
    def __init__(self):
        self.model_config = yaml.load(open('../config/model.yaml'), Loader=yaml.FullLoader)
        self.preprocess_config = yaml.load(open('../config/preprocessing.yaml'), Loader=yaml.FullLoader)
        self.train_config = yaml.load(open('../config/train.yaml'), Loader=yaml.FullLoader)

        os.makedirs(self.train_config['path']['ckpt_path'], exist_ok=True)
        os.makedirs(self.train_config['path']['log_path'], exist_ok=True)
        os.makedirs(self.train_config['path']['result_path'], exist_ok=True)

    def run(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tr_dataset = Dataset('train.txt', self.preprocess_config, self.train_config, sort=True, drop_last=True)
        validation_dataset = Dataset('val.txt', self.preprocess_config, self.train_config, sort=False, drop_last=False)
        batch_size = self.train_config["optimizer"]["batch_size"]
        group_size = 4  # Set this larger than 1 to enable sorting in Dataset
        assert batch_size * group_size < len(tr_dataset)

        tr_loader = DataLoader(
            tr_dataset,
            batch_size=batch_size * group_size,
            shuffle=True,
            collate_fn=tr_dataset.collate_fn,
        )

        model, optimizer = get_model((self.preprocess_config, self.model_config, self.train_config),
                                     device,
                                     train=True)
        criterion = FastSpeech2Loss(self.preprocess_config, self.model_config).to(device)
        num_param = get_param_num(model)
        print(f"[Number of FastSpeech2 Parameters: {num_param}]")
        vocder = get_vocoder(self.model_config, device)

        train_log_path = os.path.join(self.train_config['path']['log_path'], 'train')
        val_log_path = os.path.join(self.train_config['path']['log_path'], 'val')

        os.makedirs(train_log_path, exist_ok=True)
        os.makedirs(val_log_path, exist_ok=True)

        train_logger = SummaryWriter(train_log_path)
        val_logger = SummaryWriter(val_log_path)

        grad_acc_step = self.train_config["optimizer"]["grad_acc_step"]
        grad_clip_thresh = self.train_config["optimizer"]["grad_clip_thresh"]

        total_step = 0

        # for epoch in range(self.train_config['ckp']['restore_step'], self.train_config['step']['epochs']):
        #     for idx, batchs in enumerate(tqdm(tr_loader, desc=f"[Train Epoch ==> {epoch}/{self.train_config['step']['epochs']}]: ", position=1)):
        #         do_synth = False
        #         if total_step % self.train_config['step']['synth_step'] == 0:
        #             do_synth = True
        #
        #         for batch in batchs:
        #             batch = to_device(batch, device)
        #             output = model(*(batch[2:]))
        #
        #             losses = criterion(batch, output)
        #             total_loss = losses[0]
        #
        #             total_loss = total_loss / grad_acc_step
        #             total_loss.backward()
        #
        #             nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
        #             optimizer.step_and_update_lr()
        #             optimizer.zero_grad()
        #
        #             log(train_logger, total_step, losses=losses)
        #
        #             if do_synth:
        #                 fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
        #                     batch, output, vocder, self.model_config, self.preprocess_config)
        #
        #                 log(train_logger,
        #                     fig=fig,
        #                     tag=f"Training/epoch_{epoch}_step_{total_step}_{tag}")
        #
        #                 sampling_rate = self.preprocess_config['preprocessing']['audio']['sampling_rate']
        #                 log(train_logger,
        #                     audio=wav_prediction,
        #                     sampling_rate=sampling_rate,
        #                     tag=f"Training/epoch_{epoch}_step_{total_step}_{tag}_synthesized")
        #
        #             total_step += 1
        #

if __name__ == '__main__':
    a = Train()
    a.run()
