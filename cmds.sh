#!/usr/bin/env bash


# clean train and test
rm train -rf
rm test -rf

mkdir -p train/arz/
mkdir -p train/ar/
mkdir -p test/arz/
mkdir -p test/ar/

# run python prepare
python prepare_data.py

# split test from train
mv train/arz/doc_009* test/arz/
mv train/ar/doc_009* test/ar/
mv train/arz/doc_010* test/arz/
mv train/ar/doc_010* test/ar/



# count train and test
ls train/arz | wc -l
ls train/ar | wc -l
ls test/arz | wc -l
ls test/ar | wc -l
