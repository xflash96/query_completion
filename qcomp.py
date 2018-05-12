#!/usr/bin/env python
from __future__ import print_function
import sys
import os
import re
import datetime
import hashlib
import random
import numpy as np

import keras
from keras.layers import Input, Embedding, Dense, Dropout
from keras.layers import LSTM, GRU
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras.callbacks import Callback
import h5py


charset = 'abcdefghijklmnopqrstuvwxyz0123456789 .-%_:/\\$'

invalid_sep = re.compile('[\(\)@#$*;"\']+')
invalid_char = re.compile('[^a-zA-Z0-9 \.\-%_:/\\\\]+')
multi_space = re.compile(' +')

def normalize(s):
    s = re.sub(invalid_sep, ' ', s)
    s = re.sub(invalid_char, '', s)
    s = re.sub(multi_space, ' ', s)
    return s.lower()

# parsing raw AOL input
def parse(fname):
    f = open(fname)
    prev_query = ''
    for line in f:
        line = line.decode('utf-8','ignore').encode("utf-8")
        line = line.strip().split('\t')
        timestamp = datetime.datetime.strptime(line[2], '%Y-%m-%d %H:%M:%S')
        timestamp = (timestamp - datetime.datetime(1970,1,1)).total_seconds()
        query = normalize(line[1])

        if query == '-':
            query = prev_query
        else:
            prev_query = query
        if not query:
            prev_query = ''
            continue
        clicked = len(line) > 3
        r = 2+random.randint(0,max(0,len(query)-3))
        prefix = query[:r]
        md5 = hashlib.md5(prefix).hexdigest()
        line = '\t'.join([str(int(timestamp)), query, prefix, md5])
        print(line)

class Sequencer(object):
    PAD, END = 0, 1
    def __init__(self):
        self.token_to_indice = dict([(c,i+2) for (i,c) in enumerate(charset)])
        self.vocabs = ['PAD', 'END']+list(charset)

    def encode(self, line, ending=True):
        seq = map(self.token_to_indice.__getitem__, line)
        if ending:
            seq.append(self.END)
        return seq

    def decode(self, seq):
        if not seq:
            return ''
        if seq[-1] == self.END:
            seq = seq[:-1]
        line = ''.join(map(self.vocabs.__getitem__, seq))
        return line

def padding(seq, maxlen):
    return pad_sequences(seq, maxlen, padding='post', value=0)

class WeightsSaver(Callback):
    def __init__(self, model, N):
        self.model = model
        self.N = N
        self.batch = 0

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.N == 0:
            name = 'weights/weights%08d.hdf5' % self.batch
            self.model.save_weights(name)
        self.batch += 1

class LanguageModel(object):
    def __init__(self):
        self.sqn = Sequencer()

    def save(self):
        pass

    def load(self):
        pass

    def build(self, hid_size, n_hid_layers, drp_rate, batch_size):
        cin = Input(batch_shape=(None, None))
        voc_size = len(self.sqn.vocabs)
        # A trick to map categories to onehot encoding
        emb = Embedding(voc_size, voc_size, trainable=False, weights=[np.identity(voc_size)])(cin)
        prev = emb
        for i in range(n_hid_layers):
            lstm = LSTM(hid_size, return_sequences=True, implementation=2)(prev)
            dropout = Dropout(drp_rate)(lstm)
            prev = dropout
        cout = Dense(voc_size, activation='softmax')(prev)

        self.model = Model(inputs=cin, outputs=cout)
        self.model.summary()

        self.batch_size = batch_size

    def train(self, fname, maxlen, lr=1e-3):
        ref = []

        for line in open(fname):
            line = line.strip()
            seq = self.sqn.encode(line)
            ref.append(seq)
        ref = np.array(ref)
        ref = padding(ref, maxlen+1)
        X, Y = ref[:, :-1], ref[:, 1:]
        Y = np.expand_dims(Y, -1)
        M = X>self.sqn.END
        M[:,0] = 0

        self.model.compile(
                loss='sparse_categorical_crossentropy',
                sample_weight_mode='temporal',
                optimizer=Adam(lr=lr)
                )
        self.model.fit(X, Y, batch_size=self.batch_size, sample_weight=M,
                callbacks=[WeightsSaver(self.model, 500)],
                validation_split=0.01,
                epochs=3
                )

def array_str(arr):
    s = ', '.join(['%.8e' % x for x in arr])
    return s+',\n'


def sanitize_for_tf(name):
    #HACK for make the variable names consistent between THEANO and TENSORFLOW models
    return name.replace("KERNEL:0","KERNEL").replace("BIAS:0","BIAS")

# Dumping the HDF5 weights to a model.c file
# and specifies the dimension in model.h
def dump(fname):
    f = h5py.File(fname)
    fheader = open('model.h', 'w')
    fctx = open('model.c', 'w')
    for name in f.attrs['layer_names']:
        if name.startswith('lstm') or name.startswith('dense'):
            layer = f[name][name]
            for elem in layer:
                shape = layer[elem].shape
                for i,n in enumerate(shape):
                    current_row='int '+(name+'_%s_shape_%d = %d;\n'%(elem, i, n)).upper()
                    current_row = sanitize_for_tf(current_row)
                    fheader.write(current_row)
                elem_decl = 'const float '+(name+'_'+elem).upper()+'[]'
                  
                elem_decl = sanitize_for_tf(elem_decl)

                fheader.write('extern '+elem_decl+';\n\n')

                fctx.write(elem_decl+' = {\n')
                mat = np.array(layer[elem])
                if len(shape) == 2:
                    for i in range(shape[0]):
                        fctx.write(array_str(mat[i]))
                else:
                    fctx.write(array_str(mat))

                fctx.write('};\n\n')


if __name__ == '__main__':
    prog_name = os.path.basename(sys.argv[0])
    if prog_name == 'train':
        q = LanguageModel()
        q.build(256, 2, 0.5, 256)
        q.train(sys.argv[1], 60)
    elif prog_name == 'parse':
        parse(sys.argv[1])
    elif prog_name == 'dump':
        dump(sys.argv[1])
