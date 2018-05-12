# Realtime query completion via deep language models

This is a CPU-based implementation for our paper `Realtime query completion via deep language models`,
which is capable of generating 10 query completion in 16 ms.

To build it, please first install the following
* Intel Math Kernel Library (MKL) (https://software.intel.com/en-us/mkl)
  * Please update the MKLROOT in our Makefile if it is not installed in (/opt/intel/mkl)

## The CPU-based query completion (qcomp.c)
To play with the query completion, we pre-trained a model from the AOL dataset (model.c, model.h).
The completion program can be compiled by
```
	$ make
```
The generated `qcomp` program is soft-linked to different entries (stocsearch, beamsearch, omnisearch, trielookup).
To play with our omni-completion model, please use
```
	$ omnisearch
```
and type in any prefix, and press enter.

## Train the model with AOL data (qcomp.py)

To train the model, please first install the following python dependencies:
* Keras/Theanos/Numpy
  * The dependencies can be obtained by $pip install -r requirements.txt

You will also need to download the AOL data frm the Internet and save it in `aol_raw.txt`.
Our program qcomp.py is again soft-linked to different entries (parse, train, dump).
It's quite short so please take a look before training.

First, create the parsed data by
```
	./parse aol_raw.txt > aol_parsed.txt
```

Note that the aol_parsed.txt has the following format
```
	TIMESTAMP QUERY PREFIX MD5_OF_PREFIX
```

Then, we will sort our data by different input assumption
```
	# sort by md5 of prefix, or the timestamp
	sort --key 4 -t$'\t' --parallel=8 aol_parsed.txt > sorted.txt
	# sort --key 1 -t$'\t' -g --parallel=8 aol_parsed.txt > sorted.txt
```

The last 1% of sorted.txt will be used in testing.
Now, we can run the training and evaluation using
```
	bash ./run_eval.bash
```
