import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import tflearn
import tensorflow
import random
import json

with open("intents.json") as file:
    data = json.load(file)
print(data["intents"])

words = []
lables = []
docs = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.tokenize(pattern)
        words.extend(wrds)
        docs.append(pattern)
        
        if intent["tag"] not in lables:
            lables.append(intent["tag"])
            tflearn.DNN()