#!/bin/bash

set -e

DATA=sorted.txt
HEADER_CHAR=false

# Calculate line numbers
LINES=`wc -l < ${DATA}`
TEST_LINES=$((${LINES}/100))
TRAIN_LINES=$((${LINES}-${TEST_LINES}))
echo "Using ${TRAIN_LINES} for training, ${TEST_LINES} lines for testing"

# Do the splitting
if ${HEADER_CHAR}; then
    head -n ${TRAIN_LINES} ${DATA} | cut -f 1,2 | shuf > ${DATA}.train
    tail -n ${TEST_LINES} ${DATA} | cut -f 1,2 > ${DATA}.test
else
    head -n ${TRAIN_LINES} ${DATA} | cut -f 2 | shuf > ${DATA}.train
    tail -n ${TEST_LINES} ${DATA} | cut -f 2 > ${DATA}.test
fi
cat ${DATA}.train ${DATA}.test > ${DATA}.whole
# cut -f3 ${DATA}.test > ${DATA}.prefix
tail -n ${TEST_LINES} ${DATA} | cut -f 3 > ${DATA}.prefix

# train the model file
mkdir -p weights
./train ${DATA}.whole # note that we pass in the whole file to get the testing loss
mv weights/`ls -Art weights | tail -n 1` weights.hdf5
./dump weights.hdf5
make clean
make

# do the prediction
cat ${DATA}.prefix | ./stocsearch > result_stoc.txt
cat ${DATA}.prefix | ./beamsearch > result_beam.txt
cat ${DATA}.prefix | ./omnisearch > result_omni.txt
# I think trie search should only use training set?
cat ${DATA}.prefix | ./triesearch ${DATA}.whole > result_trie.txt

# count the appearance of results
cut -f2 result_beam.txt | ./trielookup ${DATA}.whole 1 > result_beam_freq.txt
cut -f2 result_omni.txt | ./trielookup ${DATA}.whole 1 > result_omni_freq.txt
cut -f2 result_trie.txt | ./trielookup ${DATA}.whole 1 > result_trie_freq.txt

# count the appearance of prefix
cat ${DATA}.prefix | ./trielookup ${DATA}.whole 0 > prefix_freq.txt

# evaluate the metrics
echo 'beamsearch'
python eval.py prefix_freq.txt result_beam_freq.txt
echo 'omnisearch'
python eval.py prefix_freq.txt result_omni_freq.txt
echo 'triesearch'
python eval.py prefix_freq.txt result_trie_freq.txt
