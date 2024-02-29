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
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch, spectrogram_torch
from utils import load_wav_to_torch
from prepare.dur_to_frame import dur_to_frame
import scipy.io.wavfile as sciwav
import librosa
import pdb

def save_wav(wav, path, rate):
    wav *= 32767 / max(0.01, np.max(np.abs(wav))) * 0.6
    wavfile.write(path, rate, wav.astype(np.int16))

def load_spec(wav, hps):
    y, sr = librosa.load(wav)
    phon_dur = librosa.get_duration(
        y,
        sr = hps.data.sampling_rate, 
        n_fft=hps.data.filter_length,
        hop_length=hps.data.hop_length
        )
    
    '''
    y: Any | None = None,
    sr: int = 22050,
    S: Any | None = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    filename
    '''
    # phon_dur = dur_to_frame(phon_dur, hps.data.sampling_rate, hps.data.hop_length) 
    phon_dur = np.int32(phon_dur * hps.data.sampling_rate / hps.data.hop_length + 0.5)
    # phon_dur_sum =/ torch.Tensor(phon_dur).to(torch.int32)
    phon_dur_sum = phon_dur
    
    audio, sampling_rate = load_wav_to_torch(wav)

    audio_norm = audio / hps.data.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    # spec_filename = wav.replace(".wav", ".spec.pt")
    # if os.path.exists(spec_filename):
    #     spec = torch.load(spec_filename)
    # else:
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )

    # align mel and wave
    # phone_dur_sum = torch.sum(phone_dur).item()
    spec_length = spec.shape[2]

    # pdb.set_trace()

    if spec_length > phon_dur_sum:
        spec = spec[:, :, :phon_dur_sum]
    elif spec_length < phon_dur_sum:
        pad_length = phon_dur_sum - spec_length
        spec = torch.nn.functional.pad(
            input=spec, pad=(0, pad_length, 0, 0), mode="constant", value=0
        )
    assert spec.shape[2] == phon_dur_sum

    # align wav
    fixed_wav_len = phon_dur_sum * hps.data.hop_length
    if audio_norm.shape[1] > fixed_wav_len:
        audio_norm = audio_norm[:, :fixed_wav_len]
    elif audio_norm.shape[1] < fixed_wav_len:
        pad_length = fixed_wav_len - audio_norm.shape[1]
        audio_norm = torch.nn.functional.pad(
            input=audio_norm,
            pad=(0, pad_length, 0, 0),
            mode="constant",
            value=0,
        )
    assert audio_norm.shape[1] == fixed_wav_len

    # # rewrite aligned wav
    # audio = (
    #     (audio_norm * hps.data.max_wav_value)
    #     .transpose(0, 1)
    #     .numpy()
    #     .astype(np.int16)
    # )

    # sciwav.write(
    #     filename,
    #     hps.data.sampling_rate,
    #     audio,
    # )
    # save spec
    spec = torch.squeeze(spec, 0)
    return spec
    # torch.save(spec, spec_filename)   


use_cuda = True

# define model and load checkpoint

log = './logs/meta/'
log = './logs/lr_predictor/'
model = 'G_462000.pth'

infer_list = "./scallet.txt"
infer_list = "./0229.txt"
# output = log+'output'
output = './output/0229_003'
hps = utils.get_hparams_from_file(log+"config.json")
# hps['model']['use_vc'] = False
# print(hps['model'])
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
    # print(message)
    # sid = message.split("|")[-1]

    (
        file,
        labels_ids, # label_to_ids(phon)
        scores_ids, # note_ids = notes_to_id(note) (note = info[3])
        scores_dur, # note_dur = dur_to_frame(note_dur) (note_dur = infos[4])
        labels_slr, # phon_dur (info[6])
        energy_ids,
    ) = singInput.parseInput_infer(message)

    phone = torch.LongTensor(labels_ids)
    # score = torch.LongTensor(scores_ids)
    score_int = np.array(scores_ids, dtype=int)
    score = torch.LongTensor(score_int)
    print(f'scores_dur : {scores_dur}')

    score_dur = torch.LongTensor(scores_dur)
    slurs = torch.LongTensor(labels_slr)

    energy_int = np.array(energy_ids, dtype=int)
    energy = torch.LongTensor(energy_int)//2

    # sid = torch.tensor(int(sid), dtype=torch.int16)

    phone_dur = score_dur
    phone_lengths = phone.size()[0]

    # vc
    wav = "/home/work/data/scarlett/wav/scarlett_003.wav"
    wav = "./output/0229_003/spk/7302_86815_000021_000002_bits16.wav"
    wav = "./output/0229_003/spk/5652_19215_000011_000004_bits16.wav"
    # # # spec_filename = "/home/work/PJT/VISinger/VISinger_data_bk/wav_dump_24k/196/122150/196_122150_000000_000000_bits16.spec.pt"
    # wav = "/home/work/PJT/VISinger/VISinger_data_bk/wav_dump_24k/1841/150351/1841_150351_000000_000000_bits16.wav"
    # wav = "/home/work/PJT/VISinger/VISinger_data_bk/wav_dump_24k/374/180298/374_180298_000009_000000_bits16.wav"
    # wav = "/home/work/PJT/VIVC/data/dia1_utt0.wav"
    # spec_filename = wav.replace(".wav", ".spec.pt")
    # spec = torch.load(spec_filename)    

    spec = load_spec(wav, hps)

    mel = spec_to_mel_torch(
        spec,
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.mel_fmin,
        hps.data.mel_fmax,
    )

    # mel = mel_spectrogram_torch(
    #     y,
    #     hps.data.filter_length,
    #     hps.data.n_mel_channels,
    #     hps.data.sampling_rate,
    #     hps.data.hop_length,
    #     hps.data.win_length,
    #     hps.data.mel_fmin,
    #     hps.data.mel_fmax,
    # )

    # prosody_encoder
    print(f'phone : {phone}, length{phone.shape}')
    print(f'score : {score}  length{score.shape}')
    print(f'score_dur : {score_dur} length{score_dur.shape}')
    print(f'slurs : {slurs} length{slurs.shape}')
    print(f'energy : {energy} length{energy.shape}')
    print(f'phone_dur : {phone_dur} length{phone_dur.shape}')
    print(f'phone_lengths : {phone_lengths}')
    print(f'mel : {len(mel)}')

    # break
    begin_time = time()
    with torch.no_grad():
        if use_cuda:
            phone = phone.cuda().unsqueeze(0)
            score = score.cuda().unsqueeze(0)
            score_dur = score_dur.cuda().unsqueeze(0)
            energy = energy.cuda().unsqueeze(0)
            slurs = slurs.cuda().unsqueeze(0)
            phone_lengths = torch.LongTensor([phone_lengths]).cuda()
            # vc
            # sid = sid.cuda()
            mel = mel.unsqueeze(0)
            mel = mel.cuda()

        else:
            phone = phone.unsqueeze(0)
            score = score.unsqueeze(0)
            score_dur = score_dur.unsqueeze(0)
            energy = energy.unsqueeze(0)
            slurs = slurs.unsqueeze(0)
            phone_lengths = torch.LongTensor([phone_lengths])
            mel = mel.unsqueeze(0)
        
        audio = (
            # vc
            # self, phone, score, score_dur, energy, slurs, lengths
            # prosody_encoder
            # net_g.infer(phone, phone_lengths, score, score_dur, 
            # net_g.infer(phone, phone_lengths, phone_dur, score, score_dur,  slurs)[0][0, 0]
            net_g.infer(phone, phone_lengths, score, score_dur, slurs, energy, mel)[0][0, 0]
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
