#!/usr/bin/env bash


if [ $# -ne 3 ]; then
    echo "usage ${0} python_cmd train_corpus n";
    exit -1;
fi


python_cmd=${1}
train_corpus=${2}
n=${3}

langid_train_path=~/py2env/lib/python2.7/site-packages/langid/train

printf "python_cmd: %s\n" "${python_cmd}"
printf "n gram: %s\n" "${n}"
printf "train_corpus: %s\n" "${train_corpus}"


printf "removing %s\n" "${train_corpus}_model_${n}_grams/"
rm -rf ${train_corpus}_model_${n}_grams


printf "\n\n%s\n" "step 1: index the corpus"
${python_cmd} ${langid_train_path}/index.py ${train_corpus}

read -n1 -r -p "Press space to continue..." key


printf "\n\n%s\n" "step 2: tokenization"
${python_cmd} ${langid_train_path}/tokenize.py --max_order ${n} ${train_corpus}.model

read -n1 -r -p "Press space to continue..." key


printf "\n\n%s\n" "step 3: choose features by document frequency"
${python_cmd} ${langid_train_path}/DFfeatureselect.py --max_order ${n} ${train_corpus}.model

read -n1 -r -p "Press space to continue..." key


printf "\n\n%s\n" "step 4: compute the IG weights for domain"
${python_cmd} ${langid_train_path}/IGweight.py -d ${train_corpus}.model

read -n1 -r -p "Press space to continue..." key

printf "\n\n%s\n" "step 5: compute the IG weights for language"
${python_cmd} ${langid_train_path}/IGweight.py -lb ${train_corpus}.model

read -n1 -r -p "Press space to continue..." key



printf "\n\n%s\n" "step 6: LD feature selection: take the IG weights and use them to select a feature set"
${python_cmd} ${langid_train_path}/LDfeatureselect.py ${train_corpus}.model

read -n1 -r -p "Press space to continue..." key


printf "\n\n%s\n" "step 7: build a scanner on the basis of a feature set"
${python_cmd} ${langid_train_path}/scanner.py ${train_corpus}.model

read -n1 -r -p "Press space to continue..." key


printf "\n\n%s\n" "\nstep 8: learn NB parameters using an indexed corpus and a scanner"
${python_cmd} ${langid_train_path}/NBtrain.py ${train_corpus}.model

