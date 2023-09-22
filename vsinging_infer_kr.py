import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import wavfile
from time import *

import torch
import utils
# from models import SynthesizerTrn
from models import Synthesizer

# from prepare.data_vits_phn_ofuton import SingInput
from prepare.data_vits_phn import SingInput
# from prepare.data_vits_phn_ofuton import FeatureInput
from prepare.data_vits_phn import FeatureInput
from prepare.phone_map import get_vocab_size
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch



def save_wav(wav, path, rate):
    wav *= 32767 / max(0.01, np.max(np.abs(wav))) * 0.6
    wavfile.write(path, rate, wav.astype(np.int16))


use_cuda = True

# define model and load checkpoint
log = './logs/emb/'
model = 'G_162000.pth'
infer_list = "./vsinging_infer_kr.txt"
output = log+'output'
hps = utils.get_hparams_from_file(log+"config.json")


vocab_size = get_vocab_size()

# load model
net_g = Synthesizer(
    vocab_size,
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model,
)  # .cuda()

if use_cuda:
    net_g = net_g.cuda()

_ = utils.load_checkpoint(log + model, net_g, None)
net_g.eval()
# net_g.remove_weight_norm()

singInput = SingInput(hps.data.sampling_rate, hps.data.hop_length)
featureInput = FeatureInput(
    "../VISinger/VISinger_data_bk/wav_dump_24k/", hps.data.sampling_rate, hps.data.hop_length
)

# check directory existence
if not os.path.exists(output):
    os.makedirs(output)

fo = open(infer_list, "r+")
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
    sid = message.split("|")[-1]
    (
        file,
        labels_ids, # label_to_ids(phon)
        labels_frames, # pone_dur(info[5]) phone_to_uv(phon) (phon = info[2])
        scores_ids, # note_ids = notes_to_id(note) (note = info[3])
        scores,
        scores_dur, # note_dur = dur_to_frame(note_dur) (note_dur = infos[4])
        dur_id,
        energy_ids,
        labels_slr, # phon_dur (info[6])
        labels_uvs,
    ) = singInput.parseInput(message)

    phone = torch.LongTensor(labels_ids)
    # score = torch.LongTensor(scores_ids)
    score_int = np.array(scores_ids, dtype=int)
    score = torch.LongTensor(score_int)
    print(f'scores_dur ? : {scores_dur}')

    score_dur = torch.LongTensor(scores_dur)
    slurs = torch.LongTensor(labels_slr)

    energy_int = np.array(energy_ids, dtype=int)
    energy = torch.LongTensor(energy_int)//2

    sid = torch.tensor(int(sid), dtype=torch.int16)

    phone_dur = score_dur


    phone_lengths = phone.size()[0]

    print(f'phone : {phone}')
    print(f'score : {score}')
    print(f'score_dur : {score_dur}')
    print(f'slurs : {slurs}')
    print(f'energy : {energy}')
    print(f'phone_dur : {phone_dur}')
    print(f'phone_lengths : {phone_lengths}')
    # print(f'mel : {len(mel)}')

    # break
    begin_time = time()
    with torch.no_grad():
        if use_cuda:
            phone = phone.cuda().unsqueeze(0)
            score = score.cuda().unsqueeze(0)
            score_dur = score_dur.cuda().unsqueeze(0)
            slurs = slurs.cuda().unsqueeze(0)
            phone_lengths = torch.LongTensor([phone_lengths]).cuda()
            # sid = sid.cuda().unsqueeze(0)
            energy = energy.cuda().unsqueeze(0)

        else:
            phone = phone.unsqueeze(0)
            score = score.unsqueeze(0)
            score_dur = score_dur.unsqueeze(0)
            energy = energy.unsqueeze(0)
            slurs = slurs.unsqueeze(0)
            phone_lengths = torch.LongTensor([phone_lengths])
        audio = (
            net_g.infer(phone, phone_lengths, score, score_dur, slurs, energy)[0][0, 0]
            # net_g.infer(phone, phone_lengths, phone_dur, score, score_dur,  slurs)[0][0, 0]
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
    filename = file.split('/')[-1]
    save_wav(audio, f"{output}/{filename}.wav", hps.data.sampling_rate)
fo.close()
# can be deleted
# os.system("chmod 777 output -R")
