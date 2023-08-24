import parselmouth
import textgrid
import math
from glob import glob
import os
import copy
import librosa

audio_dir = '/home/work/data/libri/LibriTTS/train-clean-100'
textgrid_dir = '/home/work/data/libritts_textgrid'

audio_list = sorted(glob(os.path.join(audio_dir, '*.wav')))
# audio_list = ['/media/zyeah/Workspace/DB/LibriTTS/train-clean-100/40_222_000001_000000_resample.wav']
# print(audio_list)


### VOLUME EXTRACTOR ###
def get_volume(audiofile):
    wavfile, sr = librosa.load(audiofile)
    volume = abs(wavfile)
    return volume

### PITCH EXTRACTOR ###
def get_pitch(audiofile):
    sound = parselmouth.Sound(audiofile)
    pitch = sound.to_pitch(time_step=0.01)
    pitch_values = pitch.selected_array['frequency']
    # pitch_values = pitch_values - 150
    # pitch_values[pitch_values == 0] = np.nan
    return pitch_values

n=0
error = ''

for audiofile in audio_list:
    n+=1

    try:
        # print(audiofile)
        volume = get_volume(audiofile)
        pitch = get_pitch(audiofile)
        # print(volume)
        # print(pitch)

        p_l = 0
        p_r = 0
        for i in range(len(pitch)-1):
            if pitch[i] != 0 and pitch[i+1] == 0:
                p_l = i
            if pitch[i] == 0 and pitch[i+1] != 0:
                p_r = i+1
            if p_l < p_r:
                if p_l != 0:
                    for j in range(p_l+1, p_r):
                        pitch[j] = pitch[p_l] + (j - p_l)*((pitch[p_r]-pitch[p_l]) / float(p_r - p_l))
                p_l = i+1
        # print(pitch)
        # print(len(volume))
        # print(len(pitch))


        ### dur, vol, F0 ext ###

        textgridfile = audiofile.replace('wavs', 'textgrid')[:-3]+'TextGrid'
        tg = textgrid.TextGrid.fromFile(textgridfile)

        phoneme_num = len(tg[1])
        # print('phn_num', phoneme_num)
        maxTime = round(100*tg.maxTime)
        # print(maxTime)

        vol_length = len(volume)/maxTime
        pit_length = len(pitch)/maxTime

        # print(vol_length)
        # print(pit_length)
        # print(maxTime*vol_length)
        # print(maxTime*pit_length)

        new_text = "phoneme, duration, volume, F0\n"
        for i in range(len(tg[1])):
            if tg[1][i].mark == "":
                new_text += 'sil'
            else:
                new_text += str(tg[1][i].mark)
            new_text += ', '

            xmin = tg[1][i].minTime
            xmax = tg[1][i].maxTime
            new_text += str(round(100*(xmax-xmin), 2))
            new_text += ', '

            xmin_v = 100*xmin*vol_length
            xmax_v = 100*xmax*vol_length
            xmin_v = math.ceil(xmin_v)
            xmax_v = math.floor(xmax_v)
            # print('vmax', xmax_v)
            volume_ = volume[xmin_v:xmax_v]
            if len(volume_) > 0:
                max_volume = 100*max(volume_)
            elif xmin_v < len(volume) and xmax_v < len(volume):
                max_volume = 100*max(volume[xmin_v], volume[xmax_v])
            else:
                print('volume max')
                print(xmin_v)
                print(xmax_v)
                max_volume = 100*volume[-1]
            # print(volume_)
            new_text += str(round(max_volume, 2))
            new_text += ', '

            xmin_p = 100*xmin*pit_length
            xmax_p = 100*xmax*pit_length
            # print(xmin_)
            # print(xmax_)
            xmin_p = math.ceil(xmin_p)
            xmax_p = math.floor(xmax_p)
            # print('pmax', xmax_p)
            pitch_ = pitch[xmin_p:xmax_p]
            # print(pitch_)
            if len(pitch_) > 0:
                avg_pitch = sum(pitch_)/len(pitch_)
            elif xmin_p < len(pitch) and xmax_p < len(pitch):
                # print(xmin_)
                # print(xmax_)
                # print(pitch[xmin_])
                # print(pitch[xmax_])
                avg_pitch = (pitch[xmin_p]+pitch[xmax_p])/2.0
            else:
                print('pitch max')
                print(xmin_p)
                print(xmax_p)
                avg_pitch = pitch[-1]
            # print(avg_pitch)
            new_text += str(round(avg_pitch, 2))
            new_text += '\n'

        # print(new_text)
        new_file = textgridfile[:-8]+'csv'
        # print(new_file)
        print(str(n)+'/'+str(len(audio_list)))
        with open(new_file, 'w') as wf:
            wf.write(new_text)
    except:
        error += audiofile+'\n'
        continue

with open('/media/zyeah/Workspace/DB/LibriTTS/error.txt', 'w') as errf:
    errf.write(error)