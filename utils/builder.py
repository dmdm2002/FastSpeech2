import os
import json

import torch
import numpy as np

import hifigan
from model import FastSpeech2, ScheduledOptim


def get_model(configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = FastSpeech2(preprocess_config, model_config).to(device)

    if (train_config['ckp']['pretrained'] is True) and (train_config['ckp']['restore_step'] == 0):
        print("[Load Pretrained Weight...]")
        ckpt = torch.load(train_config['ckp']['pretrained_path'])
        model.load_state_dict(ckpt['model'])
    elif train_config['ckp']['restore_step']:
        print(f"[Load Checkpoint {train_config['ckp']['restore_step']}...]")
        ckpt = torch.load(f"{train_config['path']['ckpt_path']}/{train_config['ckp']['restore_step']}.pth.tar")
        model.load_state_dict(ckpt['model'])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, train_config['ckp']['restore_step']
        )
        if train_config['ckp']['restore_step']:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    else:
        model.eval()
        model.requires_grad_ = False
        return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(config, device):
    name = config['vocoder']['model']

    if name == 'MelGAN':
        vocoder = torch.hub.load("descriptinc/melgan-neurips", "load_melgan", "linda_johnson")

        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)

    elif name == "HiFi-GAN":
        with open("../hifigan/config.json", "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)

        ckpt = torch.load("../hifigan/generator_LJSpeech.pth.tar")
        vocoder.load_state_dict(ckpt['generator'])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
