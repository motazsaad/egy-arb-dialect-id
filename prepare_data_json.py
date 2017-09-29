import glob
from bs4 import BeautifulSoup
import os
import clean_text
import pandas as pd
import json

ar_dir = 'ar-arz/20012017/ar/'
arz_dir = 'ar-arz/20012017/arz/'

ar_target_dir = 'train/ar/'
arz_target_dir = 'train/arz/'


def process(dir, json_out):
    labels = os.listdir(dir)
    print('labels: {}'.format(labels))
    json_writer = open(json_out, mode='w')
    for label in labels:
        p = os.path.join(dir, label) + "/*.txt"
        print('p: {}'.format(p))
        files = glob.glob(p)
        print(files)
        for f in files:
            print('processing {}'.format(f))
            filename = os.path.basename(f)
            doc = open(f).read()
            soup = BeautifulSoup(doc, 'html.parser')
            text = soup.get_text()
            text = clean_text.clean_doc(text)
            if len(text.split()) < 10:
                continue
            json.dump({'text': text, 'label': label, 'filename': filename}, json_writer)
            json_writer.write('\n')

process('ar-arz/20012017/', 'ar_arz_wiki_corpus.json')

