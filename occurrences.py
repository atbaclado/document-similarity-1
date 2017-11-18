import os, os.path
from tkinter import Tk
from tkinter import filedialog
from gensim.models.keyedvectors import KeyedVectors
from DocSim import DocSim

# --------------------------------------------------------------------------------------------

root = Tk()
root.withdraw()

path = filedialog.askdirectory()
txt_list = [f for f in os.listdir(path) if f.endswith('.txt')] 
txt_list.sort()
print(txt_list)
txt_files = list(map(lambda txt: os.path.join(path,txt), txt_list))

path_arr = path.split("/")
ln = path_arr[len(path_arr) - 2]

# --------------------------------------------------------------------------------------------

# Using the pre-trained word2vec model trained using Google news corpus of 3 billion running words.
# The model can be downloaded here: https://bit.ly/w2vgdrive (~1.4GB)
# Feel free to use to your own model.
print("Setting up googlenews model")
googlenews_model_path = './data/GoogleNews-vectors-negative300.bin'
stopwords_path = "./data/stopwords_en.txt"
model = KeyedVectors.load_word2vec_format(googlenews_model_path, binary=True)
with open(stopwords_path, 'r') as fh:
    stopwords = fh.read().split(",")
ds = DocSim(model,stopwords=stopwords)

# --------------------------------------------------------------------------------------------

for file in txt_files:
  fpath = file.split("/")
  chname = fpath[len(fpath) - 1].split('.')[0]
  print("Calculating %s occurrences" % chname)
  ocdir = "./data/output/03-run/%s" % chname
  os.makedirs(ocdir)

  ftxt = open(file, "r")
  target_docs = []
  for f in ftxt:
    if not f == '\n':
      target_docs.append(f)

  size = len(target_docs)
  for doc in target_docs:
    dets = doc.split("\t")[0].split()
    ds.para = dets[2]
    ds.chapter = int(dets[1])
    rtxt = open("%s/%s-%s-%s.txt" % (ocdir, dets[0], format(int(dets[1]), '02'), format(int(dets[2]), '03')), "w")
    rtxt.write("source: %s\n\n" % doc)
    rtxt.write(ds.calculate_similarity(doc, target_docs))
