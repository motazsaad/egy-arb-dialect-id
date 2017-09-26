#!/usr/bin/env bash



# langid training need python 2.7

# uncomment/comment the suitable python version for you
#py=python2.7
py=python # some machines python means version 2.7



# Padic corpus
bash build_lang_id_model.sh ${py} Train_Padic 4
bash build_lang_id_model.sh ${py} Train_Padic 5


# multidialect_arabic corpus (Nizar corpus)
bash build_lang_id_model.sh ${py} train_multidialect_arabic 4
bash build_lang_id_model.sh ${py} train_multidialect_arabic 5


# our corpus
bash build_lang_id_model.sh ${py} Train_Our_Corpus 4
bash build_lang_id_model.sh ${py} Train_Our_Corpus 5
