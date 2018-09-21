from gensim.models import Word2Vec
import logging

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def read_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter="\t")
    corpus = [[lis] for lis in dataset.Text.tolist()]
    return(corpus)

def get_model(corpus):
    model = Word2Vec(corpus, size=100, window=5, min_count=1, workers=4)
    model.save("word2vec.model")
    return(model)

if __name__ == "__main__":

    #######################
    #create the Model
    #######################

    corpus = read_dataset('cluster_text.csv')
    get_model(corpus)