#!/usr/bin/env bash

#corpus=${1}
corpus=./ar_arz_wiki_corpus/train/
model_dir=.model
for order in 4 5 6 7
do 
rm -rf ${model_dir}
python langid-1.1.6/langid/train/index.py ${corpus}
python langid-1.1.6/langid/train/tokenize.py --max_order ${order} ${model_dir}
python langid-1.1.6/langid/train/DFfeatureselect.py --max_order ${order} ${model_dir}
python langid-1.1.6/langid/train/IGweight.py -d ${model_dir}
python langid-1.1.6/langid/train/IGweight.py -lb ${model_dir}
python langid-1.1.6/langid/train/LDfeatureselect.py ${model_dir}
python langid-1.1.6/langid/train/scanner.py ${model_dir}
python langid-1.1.6/langid/train/NBtrain.py ${model_dir}
cp ${model_dir}/model my_models/egy_arb_${order}g_langid_model
#read -n1 -r -p "Press space to continue..." key
done 
