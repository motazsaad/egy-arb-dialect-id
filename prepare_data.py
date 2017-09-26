import glob

ar_dir = 'ar-arz/20012017/ar/'
arz_dir = 'ar-arz/20012017/arz/'

ar_files = glob.glob(ar_dir + "/*.txt")
arz_files = glob.glob(arz_dir + "/*.txt")

for f in ar_files:
    doc = open(f).read()
