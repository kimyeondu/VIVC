import os
import logging
import numpy as np
import librosa
import pyworld

from prepare.phone_map import label_to_ids
from prepare.phone_uv import uv_map
from prepare.dur_to_frame import dur_to_frame
from prepare.align_wav_spec import Align


def load_midi_map():
    notemap = {}
    notemap["rest"] = 0
    fo = open("./prepare/midi-note.scp", "r+")
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
        infos = message.split()
        notemap[infos[1]] = int(infos[0])
    fo.close()
    return notemap


class SingInput(object):
    def __init__(self, sample_rate=24000, hop_size=256):
        self.fs = sample_rate
        self.hop = hop_size
        # self.notemaper = load_midi_map()
        self.align = Align(32768, sample_rate, 1024, hop_size, 1024)

    def phone_to_uv(self, phones):
        uv = []
        uv_map_lower = {k.lower(): v for k, v in uv_map.items()}
        for phone in phones:
            # uv.append(uv_map[phone.lower()])
            uv.append(uv_map_lower[phone])
        return uv

    # def notes_to_id(self, notes):
    #     note_ids = []
    #     for note in notes:
    #         note_ids.append(self.notemaper[note])
    #     return note_ids


    def notes_to_int(self, notes):
        note_ids = []
        for note in range(len(notes)):
            note_ids.append(str(int(float(notes[note]))))
        return note_ids    

    def frame_duration(self, durations):
        ph_durs = [float(x) for x in durations]
        sentence_length = 0
        for ph_dur in ph_durs:
            sentence_length = sentence_length + ph_dur
        sentence_length = int(sentence_length * self.fs / self.hop + 0.5)

        sample_frame = []
        startTime = 0
        for i_ph in range(len(ph_durs)):
            start_frame = int(startTime * self.fs / self.hop + 0.5)
            end_frame = int((startTime + ph_durs[i_ph]) * self.fs / self.hop + 0.5)
            count_frame = end_frame - start_frame
            sample_frame.append(count_frame)
            startTime = startTime + ph_durs[i_ph]
        all_frame = np.sum(sample_frame)
        assert all_frame == sentence_length
        # match mel length
        sample_frame[-1] = sample_frame[-1] - 1
        return sample_frame

    def score_duration(self, durations):
        ph_durs = [float(x) for x in durations]
        sample_frame = []
        for i_ph in range(len(ph_durs)):
            count_frame = int(ph_durs[i_ph] * self.fs / self.hop + 0.5)
            if count_frame >= 256:
                print("count_frame", count_frame)
                count_frame = 255
            sample_frame.append(count_frame)
        return sample_frame

    def parseInput(self, singinfo: str):
        infos = singinfo.split("|")
        # file, txt, phon, note_id, note, note_dur, phon_dur, dur_id, phon_slr
        file = infos[0]
        hanz = infos[1]
        phon = infos[2].split(" ")

        note_ids = infos[3].split(" ")

        note = infos[4].split(" ")
        note = self.notes_to_int(note)
        
        note_dur = infos[5].split(" ")
        phon_dur = infos[6].split(" ")
        dur_id = infos[7].split(" ")

        phon_slr = infos[8].split(" ")
        energy_real = infos[9].split(" ")
        energy_id = infos[10].split(" ")
        energy_id = self.notes_to_int(energy_id)

        logging.info("----------------------------")
        logging.info("file {}".format(file))
        logging.info("lyrics {}".format(hanz))
        logging.info("phn {}".format(phon))


        labels_ids = label_to_ids(phon)
        labels_uvs = self.phone_to_uv(phon)
        # note_ids = note
        # convert into float
        note_dur = [eval(i) for i in note_dur]
        phon_dur = [eval(i) for i in phon_dur]

        note_dur = dur_to_frame(note_dur, self.fs, self.hop)
        phon_dur = dur_to_frame(phon_dur, self.fs, self.hop)
        labels_slr = [int(x) for x in phon_slr]

        # print("labels_ids", labels_ids)
        # print("note_dur", note_dur)
        # print("phon_dur", phon_dur)
        # print("labels_slr", labels_slr)
        # file, txt, phon, note_id, note, note_dur, phon_dur, dur_id, phon_slr
        return (
            file,
            labels_ids, # phon
            phon_dur,   # phon
            note_ids,   # note clu
            note,       # note 
            note_dur,   # note dur
            dur_id,     # dur clu
            energy_id,
            labels_slr,
            labels_uvs,
        )

    def parseSong(self, singinfo: str):
        infos = singinfo.split("|")
        item_indx = infos[0]
        item_time = infos[1]
        # hanz = infos[2]
        phon = infos[3].split(" ")
        note_ids = infos[4].split(" ")
        note_dur = infos[5].split(" ")
        phon_dur = infos[6].split(" ")
        phon_slr = infos[7].split(" ")

        labels_ids = label_to_ids(phon)
        labels_uvs = self.phone_to_uv(phon)
        labels_frames = self.frame_duration(phon_dur)
        # scores_ids = [int(x) if x != "rest" else 0 for x in note_ids]
        scores_ids = [int(x) for x in note_ids]
        scores_dur = self.score_duration(note_dur)
        labels_slr = [int(x) for x in phon_slr]
        return (
            item_indx,
            item_time,
            labels_ids,
            labels_frames,
            scores_ids,
            scores_dur,
            labels_slr,
            labels_uvs,
        )

    def expandInput(self, labels_ids, labels_frames):
        assert len(labels_ids) == len(labels_frames)
        frame_num = np.sum(labels_frames)
        frame_labels = np.zeros(frame_num, dtype=np.int)
        start = 0
        for index, num in enumerate(labels_frames):
            frame_labels[start : start + num] = labels_ids[index]
            start += num
        return frame_labels

    def scorePitch(self, scores_id):
        score_pitch = np.zeros(len(scores_id), dtype=np.float)
        for index, score_id in enumerate(scores_id):
            if score_id == 0:
                score_pitch[index] = 0
            else:
                pitch = librosa.midi_to_hz(score_id)
                score_pitch[index] = round(pitch, 1)
        return score_pitch

    def smoothPitch(self, pitch):
        # 使用卷积对数据平滑
        kernel = np.hanning(5)  # 随机生成一个卷积核（对称的）
        kernel /= kernel.sum()
        smooth_pitch = np.convolve(pitch, kernel, "same")
        return smooth_pitch

    def align_process(self, file, phn_dur):
        return self.align.align_wav_spec(file, phn_dur)


class FeatureInput(object):
    def __init__(self, path, samplerate=24000, hop_size=256):
        self.fs = samplerate
        self.hop = hop_size
        self.path = path

        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

    def compute_f0(self, filename):
        x, sr = librosa.load(self.path + filename, self.fs)
        assert sr == self.fs
        f0, t = pyworld.dio(
            x.astype(np.double),
            fs=sr,
            f0_ceil=800,
            frame_period=1000 * self.hop / sr,
        )
        f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.fs)
        for index, pitch in enumerate(f0):
            f0[index] = round(pitch, 1)
        return f0

    def coarse_f0(self, f0):
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (
            self.f0_bin - 2
        ) / (self.f0_mel_max - self.f0_mel_min) + 1

        # use 0 or 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = np.rint(f0_mel).astype(np.int)
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
            f0_coarse.max(),
            f0_coarse.min(),
        )
        return f0_coarse

    def diff_f0(self, scores_pit, featur_pit, labels_frames):
        length_pit = min(len(scores_pit), len(featur_pit))
        offset_pit = np.zeros(length_pit, dtype=np.int)
        for idx in range(length_pit):
            s_pit = scores_pit[idx]
            f_pit = featur_pit[idx]
            if s_pit == 0 or f_pit == 0:
                offset_pit[idx] = 0
            else:
                tmp = int(f_pit - s_pit)
                tmp = +128 if tmp > +128 else tmp
                tmp = -127 if tmp < -127 else tmp
                tmp = 256 + tmp if tmp < 0 else tmp
                offset_pit[idx] = tmp
        offset_pit[offset_pit > 255] = 255
        offset_pit[offset_pit < 0] = 0
        # start = 0
        # for num in labels_frames:
        #     print("---------------------------------------------")
        #     print(scores_pit[start:start+num])
        #     print(featur_pit[start:start+num])
        #     print(offset_pit[start:start+num])
        #     start += num
        return offset_pit


if __name__ == "__main__":
    # output_path = "../VISinger_data/label_vits_phn/"
    output_path = "/home/work/PJT/VISinger/VISinger_data/label_vits_phn/"
    # wav_path = "../VISinger_data/wav_dump_24k/"
    # wav_path = "/home/work/PJT/VISinger/VISinger_data/wav_dump_24k/"
    wav_path = "/home/work/data/libri/LibriTTS/train-clean-100/"
    logging.basicConfig(level=logging.INFO)  # ERROR & INFO
    pitch_norm = True
    pitch_intp = True
    uv_process = False

    # notemaper = load_midi_map()
    # logging.info(notemaper)

    sample_rate = 24000
    hop_size = 256
    singInput = SingInput(sample_rate, hop_size)
    featureInput = FeatureInput(wav_path, sample_rate, hop_size)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    fo = open("/home/work/PJT/VISinger/filelists/vits_file_clu.txt", "r+")
    # vits_file = open("./filelists/vits_file_phn.txt", "w", encoding="utf-8")
    vits_file = open("/home/work/PJT/VISinger/filelists/libri_train_clu.txt", "w", encoding="utf-8")
    i = 0
    all_txt = []  # unique sentence 
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
        i = i + 1
        # if i > 5:
        #     exit()

        # try:
        infos = message.split("|")
        file = infos[0]
        hanz = infos[1]
        all_txt.append(hanz)
        phon = infos[2].split(" ")
        note = infos[3].split(" ")
        note_dur = infos[4].split(" ")
        phon_dur = infos[5].split(" ")
        phon_slur = infos[6].split(" ")

        # infos = singinfo.split("|")
        # # file, txt, phon, note_id, note, note_dur, phon_dur, dur_id, phon_slr
        # file = infos[0]
        # # hanz = infos[1]
        # phon = infos[2].split(" ")

        # note_id = infos[3]
        # note = infos[4].split(" ")
        # note = self.notes_to_int(note)
        
        # note_dur = infos[5].split(" ")
        # phon_dur = infos[6].split(" ")
        # dur_id = infos[7].split(" ")

        # phon_slr = infos[8].split(" ")

        # logging.info("----------------------------")
        # logging.info("file {}".format(file))
        # logging.info("lyrics {}".format(hanz))
        # logging.info("phn {}".format(phon))
        # logging.info(note_dur)
        # logging.info(phon_dur)
        # logging.info(phon_slur)

            # '''
            # file,
            # labels_ids, # phon
            # phon_dur,   # phon
            # note_ids,   # note clu
            # note,       # note 
            # note_dur,   # note dur
            # dur_id,     # dur clu
            # labels_slr,
            # labels_uvs,

            # '''

        (
            file,
            labels_ids,
            labels_dur,
            scores_ids,
            scores, #
            scores_dur,
            dur_ids, #
            labels_slr,
            labels_uvs,
        ) = singInput.parseInput(message)
        # labels_ids = singInput.expandInput(labels_ids, labels_frames)
        # labels_uvs = singInput.expandInput(labels_uvs, labels_frames)
        # labels_slr = singInput.expandInput(labels_slr, labels_frames)
        # scores_ids = singInput.expandInput(scores_ids, labels_frames)
        # scores_pit = singInput.scorePitch(scores_ids)
        # featur_pit = featureInput.compute_f0(f"{file}_bits16.wav")
        featur_pit = featureInput.compute_f0(f"{file}.wav")
        # wav_file = os.path.join(wav_path, file + "_bits16.wav")
        wav_file = os.path.join(wav_path, file + ".wav")

        spec_len = singInput.align_process(wav_file, labels_dur)

        # extend uv
        labels_uvs = np.repeat(labels_uvs, labels_dur, axis=0)

        featur_pit = featur_pit[:spec_len]

        if featur_pit.shape[0] < spec_len:
            pad_length = spec_len - featur_pit.shape[0]
            featur_pit = np.pad(featur_pit, pad_width=(0, pad_length), mode="constant")
        assert featur_pit.shape[0] == spec_len
        if uv_process:
            featur_pit = featur_pit * labels_uvs
        coarse_pit = featureInput.coarse_f0(featur_pit)

        # log f0
        if not pitch_norm:
            nonzero_idxs = np.where(featur_pit != 0)[0]
            featur_pit[nonzero_idxs] = np.log(featur_pit[nonzero_idxs])
        else:
            featur_pit = 2595.0 * np.log10(1.0 + featur_pit / 700.0) / 500

        if pitch_intp:
            uv = featur_pit == 0
            featur_pit_intp = np.copy(featur_pit)
            featur_pit_intp[uv] = np.interp(
                np.where(uv)[0], np.where(~uv)[0], featur_pit[~uv]
            )

        # offset_pit = featureInput.diff_f0(scores_pit, featur_pit, labels_frames)
        # print(scores_ids)
        # print(scores_dur)
        # print(len(scores_ids))
        # print(len(scores_dur))
        # print
        # assert len(labels_ids) == len(coarse_pit)
        assert len(labels_ids) == len(labels_dur)
        assert len(labels_dur) == len(scores_ids)
        assert len(scores_ids) == len(scores_dur)
        assert len(scores_dur) == len(labels_slr)

        logging.info("labels_ids {}".format(labels_ids))
        # logging.info("labels_dur {}".format(labels_dur))
        # logging.info("scores_ids {}".format(scores_ids))
        # logging.info("scores_dur {}".format(scores_dur))
        # logging.info("labels_slr {}".format(labels_slr))
        # logging.info("labels_uvs {}".format(labels_uvs))
        # logging.info("featur_pit {}".format(featur_pit))
        logging.info("featur_pit_intp {}".format(featur_pit_intp))

        spk = file.split('/')[0]
        book = file.split('/')[1]

        if not os.path.exists(os.path.join(output_path, spk)):
            os.mkdir(os.path.join(output_path, spk))
        if not os.path.exists(os.path.join(output_path, spk, book)):
            os.mkdir(os.path.join(output_path, spk, book))            

        np.save(
            output_path + f"{file}_label.npy",
            labels_ids,
            allow_pickle=False,
        )
        np.save(
            output_path + f"{file}_label_dur.npy",
            labels_dur,
            allow_pickle=False,
        )
        np.save(
            output_path + f"{file}_score.npy",
            scores_ids,
            allow_pickle=False,
        )

        np.save(
            output_path + f"{file}_score_real.npy",
            scores,
            allow_pickle=False,
        )

        np.save(
            output_path + f"{file}_score_dur.npy",
            scores_dur,
            allow_pickle=False,
        )

        np.save(
            output_path + f"{file}_durID.npy",
            dur_ids,
            allow_pickle=False,
        )

        if not pitch_intp:
            np.save(
                output_path + f"{file}_pitch.npy",
                featur_pit,
                allow_pickle=False,
            )
        else:
            np.save(
                output_path + f"{file}_pitch.npy",
                featur_pit_intp,
                allow_pickle=False,
            )
        # np.save(
        #     output_path + f"{file}_pitch.npy",
        #     coarse_pit,
        #     allow_pickle=False,
        # )
        np.save(
            output_path + f"{file}_slurs.npy",
            labels_slr,
            allow_pickle=False,
        )

        # wave path|label path|label frame|score path|score duration;上面是一个.（当前目录），下面是两个..（从子目录调用）
        path_wave = wav_path + f"{file}_bits16.wav"
        path_label = output_path + f"{file}_label.npy"
        path_label_dur = output_path + f"{file}_label_dur.npy"
        path_score = output_path + f"{file}_score.npy"
        path_score_real = output_path + f"{file}_score_real.npy"
        path_score_dur = output_path + f"{file}_score_dur.npy"
        path_dur_ids = output_path + f"{file}dur_ids.npy"

        path_pitch = output_path + f"{file}_pitch.npy"
        path_slurs = output_path + f"{file}_slurs.npy"
        print(
            f"{path_wave}|{path_label}|{path_label_dur}|{path_score}|{path_score_real}|{path_score_dur}|{path_dur_ids}|{path_pitch}|{path_slurs}",
            file=vits_file,
        )
        # except:
        #     print("@")

    fo.close()
    vits_file.close()
    print(len(set(all_txt)))  # 统计非重复的句子个数
