#!/usr/bin/env bash

idir=$1
odir=$2

function maxjobs {
  while [ `jobs | wc -l` -ge 6 ]
  do
    sleep 5
  done
}

find $idir -type f -name "*.txt" | (while read file;
  do
    maxjobs
    >&2 echo "Parsing $file"
    ofile=$(basename $file)
    python clean_corpus.py $file $odir/$ofile &
  done
  wait
)
