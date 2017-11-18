from gensim.models.keyedvectors import KeyedVectors
from DocSim import DocSim

# Using the pre-trained word2vec model trained using Google news corpus of 3 billion running words.
# The model can be downloaded here: https://bit.ly/w2vgdrive (~1.4GB)
# Feel free to use to your own model.
googlenews_model_path = './data/GoogleNews-vectors-negative300.bin'
stopwords_path = "./data/stopwords_en.txt"

print("Loading gooogle news vector")
model = KeyedVectors.load_word2vec_format(googlenews_model_path, binary=True)
with open(stopwords_path, 'r') as fh:
    stopwords = fh.read().split(",")
ds = DocSim(model,stopwords=stopwords)

path = "./data/paragraph/"
source_doc = open("./data/query/q6.txt").read()
target_docs = []

for ctr in range(1,24):
	file = open(path + "01/cd" + format(ctr, '04') + ".txt")
	for f in file:
		target_docs.append("1-%d: " % (ctr+1) + f)
	# target_docs = target_docs + file.readlines()

for ctr in range(1,12):
	file = open(path + "02/cd" + format(ctr, '02') + ".txt")
	for f in file:
		target_docs.append("2-%d: " % (ctr+1) + f)

sim_scores = ds.calculate_similarity(source_doc, target_docs)

# print(sim_scores)

# Prints:
##   [ {'score': 0.99999994, 'doc': 'delete a invoice'}, 
##   {'score': 0.79869318, 'doc': 'how do i remove an invoice'}, 
##   {'score': 0.71488398, 'doc': 'purge an invoice'} ]
