#!/usr/bin/env bash

for i in 9 10 11 12 13 14 15 16 17 19 20 21 24 25 26 27 28 30 31 32 33 34 36 37 39 41 42 43 44 45 46 47 49 51 52 53 54 55 56 57 58 59 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 78 79 81 82 86 88 89 90 91 92 93 94 95 96 97 98 99
do
    idx=$(printf "%03d" $i)
    echo "================="
    echo "Running index $idx"
    echo "================="

    echo "Getting the data"
    # scp crscardellino@172.18.0.249:Projects/dnnwsd/resources/corpus_datasets/es/7k/bow/${idx}.p /home/ccardellino/datasets/dataset_corpus/es/7k/bow/ &> /dev/null
    scp crscardellino@172.18.0.249:Projects/dnnwsd/resources/corpus_datasets/en/7k/vec/${idx}.p /home/ccardellino/datasets/dataset_corpus/en/7k/vec/ &> /dev/null
    scp crscardellino@172.18.0.249:Projects/dnnwsd/resources/corpus_datasets/en/7k/vecpos/${idx}.p /home/ccardellino/datasets/dataset_corpus/en/7k/vecpos/ &> /dev/null

    CUDA_VISIBLE_DEVICES=0 ./ladder_semeval.py $i

    echo "Deleting the data (to save space)"
    # rm -f /home/ccardellino/datasets/dataset_corpus/es/7k/bow/${idx}.p
    rm -f /home/ccardellino/datasets/dataset_corpus/en/7k/vec/${idx}.p
    rm -f /home/ccardellino/datasets/dataset_corpus/en/7k/vecpos/${idx}.p

    echo "=================="
    echo "Finished index $idx"
    echo "=================="
    mv results/* resources/results/ladder_semeval/
done
