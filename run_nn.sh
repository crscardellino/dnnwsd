#!/usr/bin/env bash

for i in 000 001 002 003 004 005 006 007 008 009 010 011 013 014 017 018 019 020 021 022 023 024 025 026 027 029 030 031 032 033 034 035 037 038 039 040 042 043 044 045 046 047 049 050 051 052 053 054 055 056 057 060 061 062 064 065 066 067 068 069 071 072 073 074 075 076 077 078 079 081 082 083 084 085 087 088 089 090 091 092 093 095 096 097 098 099 100 102 103 105 106 107 108 109 110 111 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 133 134 135 136 137 138 139 140 141 142 144 146 147 148 150 151 152 153 154 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 179 180 182 183 184 185 186 187 188 190 192 193 194 195 196 197 198 199 200 201 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 223 224 225 226 227 228 229 230 232 233 234 235 236 237 238 239 241 242 244 245 246 247 248
do
    idx=$(printf "%03d" $i)
    echo "================="
    echo "Running index $idx"
    echo "================="

    echo "Getting the data"
    scp crscardellino@172.18.0.249:Projects/dnnwsd/resources/corpus_datasets/es/7k/vec/${idx}.p /home/ccardellino/datasets/dataset_corpus/es/7k/vec/ &> /dev/null
    scp crscardellino@172.18.0.249:Projects/dnnwsd/resources/corpus_datasets/es/7k/vecpos/${idx}.p /home/ccardellino/datasets/dataset_corpus/es/7k/vecpos/ &> /dev/null

    CUDA_VISIBLE_DEVICES=0 ./neuralnetworks.py $i vec cnn
    if [ $? != 0 ]
    then
        echo "Index $idx didn't finish correctly for vec and cnn" 1>&2
        exit 1
    fi

    CUDA_VISIBLE_DEVICES=0 ./neuralnetworks.py $i vec mlp
    if [ $? != 0 ]
    then
        echo "Index $idx didn't finish correctly for vec and mlp" 1>&2
        exit 1
    fi

    CUDA_VISIBLE_DEVICES=0 ./neuralnetworks.py $i vecpos cnn
    if [ $? != 0 ]
    then
        echo "Index $idx didn't finish correctly for vecpos and cnn" 1>&2
        exit 1
    fi

    CUDA_VISIBLE_DEVICES=0 ./neuralnetworks.py $i vecpos mlp
    if [ $? != 0 ]
    then
        echo "Index $idx didn't finish correctly for vecpos and mlp" 1>&2
        exit 1
    fi

    echo "Deleting the data (to save space)"
    rm -f /home/ccardellino/datasets/dataset_corpus/es/7k/vec/${idx}.p
    rm -f /home/ccardellino/datasets/dataset_corpus/es/7k/vecpos/${idx}.p

    echo "=================="
    echo "Finished index $idx"
    echo "=================="
    mv results/* resources/results/neuralnetworks
done
