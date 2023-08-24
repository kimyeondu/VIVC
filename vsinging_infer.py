import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import wavfile
from time import *

import torch
import utils
from models import Synthesizer
from prepare.data_vits import SingInput
from prepare.data_vits import FeatureInput
from prepare.phone_map import get_vocab_size


def save_wav(wav, path, rate):
    wav *= 32767 / max(0.01, np.max(np.abs(wav))) * 0.6
    wavfile.write(path, rate, wav.astype(np.int16))


use_cuda = True

# define model and load checkpoint
hps = utils.get_hparams_from_file("./configs/singing_base.json")

vocab_size = get_vocab_size()

model_path = "./logs/libri/"
saved_models = os.listdir(model_path)
iter_nums = []
for i in range(len(saved_models)):
    if os.path.splitext(saved_models[i])[1] == ".pth" and "G" in saved_models[i]:
        iter_nums.append(int(os.path.splitext(saved_models[i])[0][2:]))
iter_nums = sorted(iter_nums, reverse=True)

print("start infering (G_" + str(iter_nums[0]) + ".pth)")

net_g = Synthesizer(
    vocab_size,
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model,
)  # .cuda()

if use_cuda:
    net_g = net_g.cuda()

_ = utils.load_checkpoint(
    "./logs/libri/G_" + str(iter_nums[0]) + ".pth", net_g, None
)

net_g.eval()
# net_g.remove_weight_norm()

singInput = SingInput(hps.data.sampling_rate, hps.data.hop_length)
featureInput = FeatureInput(
    "../VISinger_data_bk/wav_dump_16k/", hps.data.sampling_rate, hps.data.hop_length
)

# check directory existence
if not os.path.exists("./libri_out"):
    os.makedirs("./libri_out")

fo = open("./vsinging_infer.txt", "r+")
while True:
    try:
        message = fo.readline().strip()
    except Exception as e:
        print("nothing of except:", e)
        break
    if message == None:
        break
    if message == "":
        break
    print(message)
    (
        file,
        labels_ids,
        labels_frames,
        scores_ids,
        scores_dur,
        labels_slr,
        labels_uvs,
    ) = singInput.parseInput(message)

    phone = torch.LongTensor(labels_ids)
    score = torch.LongTensor(scores_ids)
    score_dur = torch.LongTensor(scores_dur)
    slurs = torch.LongTensor(labels_slr)

    phone_lengths = phone.size()[0]

    begin_time = time()
    with torch.no_grad():
        if use_cuda:
            phone = phone.cuda().unsqueeze(0)
            score = score.cuda().unsqueeze(0)
            score_dur = score_dur.cuda().unsqueeze(0)
            slurs = slurs.cuda().unsqueeze(0)
            phone_lengths = torch.LongTensor([phone_lengths]).cuda()
        else:
            phone = phone.unsqueeze(0)
            score = score.unsqueeze(0)
            score_dur = score_dur.unsqueeze(0)
            slurs = slurs.unsqueeze(0)
            phone_lengths = torch.LongTensor([phone_lengths])
        audio = (
            net_g.infer(phone, phone_lengths, score, score_dur, slurs)[0][0, 0]
            .data.cpu()
            .float()
            .numpy()
        )
    end_time = time()
    run_time = end_time - begin_time
    print("Syth Time (Seconds):", run_time)
    data_len = len(audio) / hps.data.sampling_rate
    print("Wave Time (Seconds):", data_len)
    print("Real time Rate (%):", run_time / data_len)
    save_wav(audio, f"./singing_out/{file}.wav", hps.data.sampling_rate)
fo.close()
# can be deleted
os.system("chmod 777 ./singing_out -R")
