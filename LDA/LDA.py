
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora
import os

filePath = "/home/hoomanvhd/Desktop/VideoSample/TEST1/VideoTextReadByLineSilent/"

fileList = []
for filename in os.listdir(filePath):
    filename = filename.replace(".txt", "")
    fileList.append(int(filename))

fileList.sort()

d = {}

for i in fileList:
    with open(filePath + str(i) + ".txt", "r") as file:
        reader = file.read()
        d["doc{0}".format(i)] = reader


doc_complete = []

for item in d:
    doc_complete.append(d[item])

stop_words = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ". join([i for i in doc.lower().split() if i not in stop_words])
    punc_free = "".join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


doc_clean = [clean(doc).split() for doc in doc_complete]

dictionary = corpora.Dictionary(doc_clean)

doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

Lda = gensim.models.ldamodel.LdaModel
ldamodel = Lda(doc_term_matrix, num_topics=3, id2word=dictionary, passes=50)

print(ldamodel.print_topics(num_topics=3, num_words=3))



