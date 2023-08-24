# import parselmouth
# import textgrid
import math
from glob import glob
import os
import copy
import librosa
import numpy as np
from scipy.io import wavfile
import tqdm
import pyworld

dir = '/home/work/PJT/VISinger/filelists/vits_file_clu.txt'
audio_dir = '/home/work/data/libri/LibriTTS/train-clean-100'
# textgrid_dir = '/home/work/data/libritts_textgrid'
# /home/work/data/libri/LibriTTS/train-clean-100/40/222/40_222_000002_000000.wav

def compute_energy(filename, fs, hop_length):
    x, sr = librosa.load(filename, sr=fs)
    assert sr == fs

    # 에너지를 계산할 윈도우 크기 설정 (예: 25ms)
    window_size = int(0.025 * fs)

    # Hop 길이 설정 (주어진 hop_length 사용)
    hop_size = hop_length

    # 에너지를 저장할 리스트 초기화
    energy = []

    # 오디오를 윈도우링하고 각 윈도우에 대한 에너지 계산
    for i in range(0, len(x) - window_size, hop_size):
        window = x[i:i + window_size]
        energy.append(np.sum(window ** 2))

    return energy



# 3983/5371/3983_5371_000066_000000|a pause.|sil ah0 p ao1 z sil|4 7 9 6 4 4|0.0 245.78 399.6 174.15 38.49 0.0|0.07 0.08 0.14 0.28 0.16 0.12|0.07 0.08 0.14 0.28 0.16 0.12|0 0 9 8 9 1|0 0 0 0 0 0
# path | script | phoneme | score | pitch | duration, ..

# 함수: 두 시그널 간의 RMSE 계산
def calculate_rmse(signal1, signal2):
    return np.sqrt(np.mean((signal1 - signal2) ** 2)/abs(len(signal1)-len(signal2)) )

def get_energy(audio_file):
    wavfile, sr = librosa.load(audio_file)
    energy = abs(wavfile)
    return wavfile, energy, sr

# WAV 파일과 해당 phoneme-level 정보를 읽어옴
with open(dir, 'r') as file:
    lines = file.readlines()


# 에너지 값을 계산하여 리스트에 저장
total_energy=[] #list
total_energy_=[] #각각

new_lines = []
with open('/home/work/PJT/VISinger/filelists/e_error.txt', 'w') as file:
    for line in tqdm.tqdm(lines):
        line_parts = line.strip().split('|')
        file_path = os.path.join(audio_dir, line_parts[0])
        audio_file = file_path+'.wav'
        wav, _, sr = get_energy(audio_file)
        energy_ = compute_energy(audio_file, sr, 256)
        output_path = '/home/work/PJT/VISinger/VISinger_data/label_vits_phn/' + line_parts[0]
        # np.save(
        #     output_path+'_energy_frame.npy',
        #     energy_,
        #     allow_pickle=False,
        # ) 

        pit = np.load(output_path+'_pitch.npy')
        if len(pit) != len(energy_):
            print(audio_file)
            file.write(audio_file+'\n')

        # phonemes = line_parts[2].split()
        # durs = [float(d) for d in line_parts[5].split()]
        # acc_durs = [int(sum(durs[:d])*sr) for d in range(len(durs))]
        # acc_durs[-1] = len(wav)

        # egs = [0]
        # for d in range(1, len(acc_durs)):
        #     st = acc_durs[d-1]
        #     ed = acc_durs[d]
            
        #     eg = (sum(abs(wav[st:ed+1]**2))/(ed-st))**1/2
        #     egs.append(eg)
            
        # # min-max norm
        # max_e = max(egs)
        # min_e = min(egs)
        # egs = [str(200*(eg-min_e)/(max_e-min_e)) for eg in egs]
        # egs_id = [str(int(round(float(e), -1)//10)) for e in egs]

        # /home/work/PJT/VISinger/VISinger_data/label_vits_phn/6836/61803/6836_61803_000055_000002_label.npy
        # output_path = '/home/work/PJT/VISinger/VISinger_data/label_vits_phn/' + line_parts[0]

        # np.save(
        #     output_path+'_energy.npy',
        #     egs,
        #     allow_pickle=False,
        # )
        # np.save(
        #     output_path+'_energy_id.npy',
        #     egs_id,
        #     allow_pickle=False,
        # )    

        # append
        # total_energy.append(egs)
        # total_energy.append(egs_id)
        # total_energy_+=egs

        # line_parts.append(' '.join(egs))
        # line_parts.append(' '.join(egs_id))
        # new_lines.append(line_parts)
        
        # file.write('|'.join(line_parts)+'\n')



# np.save('total_energy', total_energy_, allow_pickle=True,)