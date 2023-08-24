# Init
Unofficial Implement of VISinger

# Reference Repos
https://github.com/jaywalnut310/vits

https://github.com/MoonInTheRiver/DiffSinger

https://wenet.org.cn/opencpop/

https://github.com/PlayVoice/VI-SVS

# Data Preprocess
```bash
export PYTHONPATH=.
```

Generate ../VISinger_data/label_vits_phn/XXX._label.npy|XXX._label_dur.npy|XXX_score.npy|XXX_score_dur.npy|XXX_pitch.npy|XXX_slurs.npy

```bash
python prepare/data_vits_phn.py
```

Generate filelists/vits_file.txt
Format: wave path|label path|label duration path|score path|score duration path|pitch path|slurs path;

```bash
python prepare/preprocess.py
```

# VISinger training

```bash
python train.py -c configs/singing_base.json -m singing_base
```

or

```bash
./train.sh
```

# Inference

```bash
./evaluate_score.sh
```

![LOSS](/resource/vising_loss.png)
![MEL](/resource/vising_mel.png)

# Samples

<audio id="audio" controls="" preload="none">
      <source id="wav" src="/resource/2005000151.wav">
</audio>

<audio id="audio" controls="" preload="none">
      <source id="wav" src="/resource/2005000152.wav">
</audio>

<audio id="audio" controls="" preload="none">
      <source id="wav" src="/resource/2005000186.wav">
</audio>

<audio id="audio" controls="" preload="none">
      <source id="wav" src="/resource/2005000187.wav">
</audio>

<audio id="audio" controls="" preload="none">
      <source id="wav" src="/resource/2005000268.wav">
</audio>





1970/26100/1970_26100_000020_000004_dur_slow|they slanted down but not sidewise.|dh ey1 s l ae1 n t ih0 d d aw1 n b ah1 t n aa1 t s ay1 d w ay2 z sil|4 2 2 2 7 1 1 7 10 10 2 6 6 10 10 10 2 2 7 2 2 10 6 3 4|0.0 204.58 201.54 217.82 234.53 276.8 274.92 239.96 178.98 192.45 224.23 167.7 156.29 180.57 180.74 187.07 209.19 219.46 230.72 216.0 225.33 190.62 163.58 74.98 0.0|0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5|0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5|0 11 9 6 9 6 0 0 11 6 8 1 3 1 1 6 9 11 0 9 6 0 4 9 0|0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0|0
