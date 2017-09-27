import glob
from bs4 import BeautifulSoup
import os
import clean_text

ar_dir = 'ar-arz/20012017/ar/'
arz_dir = 'ar-arz/20012017/arz/'

ar_target_dir = 'ar_arz_wiki_train/wiki/ar/'
arz_target_dir = 'ar_arz_wiki_train/wiki/arz/'


def process(src_dir, target_dir):
    files = glob.glob(src_dir + "/*.txt")
    for f in files:
        print('processing {}'.format(f))
        doc = open(f).read()
        soup = BeautifulSoup(doc, 'html.parser')
        text = soup.get_text()
        text = clean_text.clean_doc(text)
        if len(text.split()) < 10:
            continue
        filename = os.path.basename(f)
        outfile = target_dir + filename
        print('writing out file {}'.format(outfile))
        with open(outfile, mode='w') as file_writer:
            file_writer.write(text)


process(arz_dir, arz_target_dir)
process(ar_dir, ar_target_dir)
