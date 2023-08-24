import pandas as pd
from sklearn.cluster import KMeans
from glob import glob
import os
import numpy as np

n_speakers = 125

dur_model = KMeans(n_clusters=15, algorithm='auto')
vol_model = KMeans(n_clusters=12, algorithm='auto')
pit_models = [KMeans(n_clusters=12, algorithm='auto') for i in range(n_speakers)]

file_dir = '/media/zyeah/Workspace/DB/LibriTTS/data_control/prosody_control_textgrid'
filelist = sorted(glob(os.path.join(file_dir, '*.csv')))
# filelist = ['/media/zyeah/Workspace/DB/LibriTTS/libritts_aug_features/1069_133699_000001_000000_resample.csv', '/media/zyeah/Workspace/DB/LibriTTS/libritts_aug_features/1069_133699_000001_000000_resample_tempo0.8.csv', '/media/zyeah/Workspace/DB/LibriTTS/libritts_aug_features/1088_129236_000014_000004_resample_tempo1.2.csv']

data = pd.DataFrame(columns={'phoneme', 'duration', 'volume', 'F0'})
# print(data)
data.columns = ['phoneme', 'duration', 'volume', 'F0']

speaker_dict = {}
data_pit = []

n = 0
for file in filelist:
    n+=1

    speaker_id = os.path.basename(file).split('_')[0]
    # print(file)
    # print(speaker_id)
    if not speaker_id in speaker_dict.keys():
        speaker_dict[speaker_id] = len(speaker_dict)
        data_pit.append(pd.DataFrame(columns={'F0'}))
    speaker_id = speaker_dict[speaker_id]
    # data_pit[speaker_id].columns = ['F0']

    print('1: ', str(n)+'/'+str(len(filelist)))
    data_ = pd.read_csv(file, sep = ",")
    # print(data_)
    data_.columns = ['phoneme', 'duration', 'volume', 'F0']
    data = pd.concat([data, data_], axis=0)

    # data_pit_ = pd.DataFrame(columns={'F0'})
    # data_pit_.columns = ['F0']
    # data_pit_ = pd.concat([data_pit_, data_.loc[:, 'F0']], axis=0)
    # data_pit_.columns = ['-', 'F0']
    data_pit_ = data_.drop(['phoneme', 'duration', 'volume'], axis=1)
    # data_pit_ = data_pit_.loc[:, 'F0']
    # print(data_pit[speaker_id])
    # print(data_pit_)

    data_pit[speaker_id] = pd.concat([data_pit[speaker_id], data_pit_], axis=0)
    # print(data_pit[speaker_id])

print()
print('clustering...')

data.to_csv("/media/zyeah/Workspace/DB/LibriTTS/data_for_clustering.csv")
speaker_ids = list(speaker_dict.keys())
for i in range(len(data_pit)):
    data_pit[i].to_csv("/media/zyeah/Workspace/DB/LibriTTS/data_for_clustering_"+str(speaker_ids[i])+".csv")

d_dur = data.loc[:, 'duration'].to_numpy().reshape(-1, 1)
d_vol = data.loc[:, 'volume'].to_numpy().reshape(-1, 1)
dur_model.fit(d_dur)
vol_model.fit(d_vol)

for i in range(len(data_pit)):
    d_pit = data_pit[i].loc[:, 'F0'].to_numpy().reshape(-1, 1)
    pit_models[i].fit(d_pit)

# print(dur_model.cluster_centers_)
# print(dur_model.labels_)
dur_cluster_center = dur_model.cluster_centers_.squeeze().tolist()
dur_cluster_index = [dur_cluster_center.index(x) for x in dur_cluster_center]
# print(cluster_center)
# print(cluster_index)
dur_cluster = [dur_cluster_center, dur_cluster_index]
dur_cluster = [list(x) for x in zip(*dur_cluster)]
# print(cluster)
dur_cluster.sort()
# print(cluster)
dur_cluster = [list(x) for x in zip(*dur_cluster)]
dur_label = dur_cluster[1]
# print(label)
# for i in range(len(label)):
    # print(label.index(i), i)
# print(vol_model.cluster_centers_)

vol_cluster_center = vol_model.cluster_centers_.squeeze().tolist()
vol_cluster_index = [vol_cluster_center.index(x) for x in vol_cluster_center]
vol_cluster = [vol_cluster_center, vol_cluster_index]
vol_cluster = [list(x) for x in zip(*vol_cluster)]
vol_cluster.sort()
vol_cluster = [list(x) for x in zip(*vol_cluster)]
vol_label = vol_cluster[1]


pit_labels = []
for i in range(len(data_pit)):
    pit_cluster_center = pit_models[i].cluster_centers_.squeeze().tolist()
    pit_cluster_index = [pit_cluster_center.index(x) for x in pit_cluster_center]
    pit_cluster = [pit_cluster_center, pit_cluster_index]
    pit_cluster = [list(x) for x in zip(*pit_cluster)]
    pit_cluster.sort()
    pit_cluster = [list(x) for x in zip(*pit_cluster)]
    pit_labels += [pit_cluster[1]]

n = 0
print()
for file in filelist:
    n+=1

    speaker_id = os.path.basename(file).split('_')[0]
    # print(file)
    # print(speaker_id)
    speaker_id = speaker_dict[speaker_id]

    print('2: ', str(n)+'/'+str(len(filelist)))
    data_ = pd.read_csv(file, sep=",")
    data_.columns = ['phoneme', 'duration', 'volume', 'F0']
    d_dur = data_.loc[:, 'duration'].to_numpy().reshape(-1, 1)
    d_vol = data_.loc[:, 'volume'].to_numpy().reshape(-1, 1)
    d_pit = data_.loc[:, 'F0'].to_numpy().reshape(-1, 1)

    dur_predict = pd.DataFrame(dur_model.predict(d_dur))
    dur_predict.columns = ['dur_predict']
    # print(dur_predict)
    for i in range(len(dur_predict)):
        dur_predict.loc[i, 'dur_predict'] = dur_label.index(dur_predict.loc[i, 'dur_predict'])
        # dur_predict[i] = label.index(dur_predict[i])
    # print(dur_predict)

    vol_predict = pd.DataFrame(vol_model.predict(d_vol))
    vol_predict.columns = ['vol_predict']
    for i in range(len(vol_predict)):
        vol_predict.loc[i, 'vol_predict'] = vol_label.index(vol_predict.loc[i, 'vol_predict'])

    pit_predict = pd.DataFrame(pit_models[speaker_id].predict(d_pit))
    pit_predict.columns = ['pit_predict']
    # print(pit_labels)
    # print(pit_labels[speaker_id])
    for i in range(len(pit_predict)):
        pit_predict.loc[i, 'pit_predict'] = pit_labels[speaker_id].index(pit_predict.loc[i, 'pit_predict'])
    # print(pit_predict)

    data_ = pd.concat([data_, dur_predict, vol_predict, pit_predict], axis=1)
    data_.loc[data_['phoneme'] =='sil', 'vol_predict'] = 0
    data_.loc[data_['phoneme'] =='sil', 'pit_predict'] = 0
    # print(data_)

    newfile = '/media/zyeah/Workspace/DB/LibriTTS/libri_aug_features_cluster/'+os.path.basename(file)[:-4]+'_features.csv'
    # print(newfile)
    data_.to_csv(newfile, mode='w')