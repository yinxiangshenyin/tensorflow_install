import Train_data_lib
import numpy as np
from collections import Counter
import librosa  # https://github.com/librosa/librosa
import pdb
import tensorflow as tf  # 0.12


print("ok")

train_data,train_label=Train_data_lib.Get_train_data()
print("样本数：",len(train_label))



all_wordss= []
for labels in train_label:
    all_words += [word for word in label]

counter = Counter(all_words)
count_pairs = sorted(counter.items(), key=lambda x: -x[1])
pdb.set_trace()
wordsss, _ = zip(*count_pairs)
words_size = len(words)
print('词汇表大小:', words_size)

pdb.set_trace()
word_num_map = dict(zip(words, range(len(words))))
ceshi=0
to_num = lambda word: word_num_map.get(word, len(words))
labels_vector = [ list(map(to_num, label)) for label in train_label]
#print(wavs_file[0], labels_vector[0])
#wav/train/A11/A11_0.WAV -> [479, 0, 7, 0, 138, 268, 0, 222, 0, 714, 0, 23, 261, 0, 28, 1191, 0, 1, 0, 442, 199, 0, 72, 38, 0, 1, 0, 463, 0, 1184, 0, 269, 7, 0, 479, 0, 70, 0, 816, 254, 0, 675, 1707, 0, 1255, 136, 0, 2020, 91]
#print(words[479]) #绿
label_max_len = np.max([len(label) for label in labels_vector])
print('最长句子的字数:', label_max_len)




wav_max_len = 0  # 673
i=0;
pdb.set_trace()
for wav in train_data:
    i=i+1
    print(i)
    wav, sr = librosa.load(wav, mono=True)
    mfcc = np.transpose(librosa.feature.mfcc(wav, sr), [1, 0])
    if len(mfcc) > wav_max_len:
        wav_max_len = len(mfcc)

pdb.set_trace()
print("最长的语音:", wav_max_len)
pdb.set_trace()
batch_size = 16
n_batch = len(train_data) // batch_size

# 获得一个batch
pointer = 0

pdb.set_trace()
def get_next_batches(batch_size):
    global pointer
    batches_wavs = []
    batches_labels = []
    for i in range(batch_size):
        wav, sr = librosa.load(train_data[pointer], mono=True)
        mfcc = np.transpose(librosa.feature.mfcc(wav, sr), [1, 0])
        batches_wavs.append(mfcc.tolist())
        batches_labels.append(labels_vector[pointer])
        pointer += 1

        # 补零对齐
    for mfcc in batches_wavs:
        while len(mfcc) < wav_max_len:
            mfcc.append([0] * 20)
    for label in batches_labels:
        while len(label) < label_max_len:
            label.append(0)
    return batches_wavs, batches_labels


X = tf.placeholder(dtype=tf.float32, shape=[batch_size, None, 20])
sequence_len = tf.reduce_sum(tf.cast(tf.not_equal(tf.reduce_sum(X, reduction_indices=2), 0.), tf.int32),
                             reduction_indices=1)
Y = tf.placeholder(dtype=tf.int32, shape=[batch_size, None])