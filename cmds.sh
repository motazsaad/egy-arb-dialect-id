

# clean train and test
rm ar_arz_wiki_train/wiki/arz/*
rm ar_arz_wiki_train/wiki/ar/*
rm ar_arz_wiki_test/arz/*
rm ar_arz_wiki_test/ar/*



# split test from train
mv ar_arz_wiki_train/wiki/arz/doc_009* ar_arz_wiki_test/arz/
mv ar_arz_wiki_train/wiki/ar/doc_009* ar_arz_wiki_test/ar/
mv ar_arz_wiki_train/wiki/arz/doc_010* ar_arz_wiki_test/arz/
mv ar_arz_wiki_train/wiki/ar/doc_010* ar_arz_wiki_test/ar/