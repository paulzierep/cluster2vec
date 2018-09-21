import logging
import random

import numpy as np
import pandas as pd

from gensim.models import doc2vec
from gensim.models.doc2vec import Doc2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def df2labeled_sentence(df):
    """
    Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
    We do this by using the LabeledSentence method. The format will be "TRAIN_i" or "TEST_i" where "i" is
    a dummy index of the review.
    """
    logging.info("Create labeled sentence")

    labeled = []
    for index, row in df.iterrows():
        label = row.Class + '_' + str(row['ID'])
        text = row.Text.split()
        labeled.append(doc2vec.LabeledSentence(text, [label]))

    return(labeled)


    # labeled = []
    # for i, v in enumerate(corpus):
    #     label = label_type + '_' + str(i)
    #     labeled.append(doc2vec.LabeledSentence(v.split(), [label]))
    # return labeled

def train_doc2vec(corpus):
    logging.info("Building Doc2Vec vocabulary")
    d2v = doc2vec.Doc2Vec(min_count=1,  # Ignores all words with total frequency lower than this
                          window=10,  # The maximum distance between the current and predicted word within a sentence
                          vector_size=50,  # Dimensionality of the generated feature vectors
                          workers=5,  # Number of worker threads to train the model
                          alpha=0.025,  # The initial learning rate
                          min_alpha=0.00025,  # Learning rate will linearly drop to min_alpha as training progresses
                          dm=1)  # dm defines the training algorithm. If dm=1 means ‘distributed memory’ (PV-DM)
                                 # and dm =0 means ‘distributed bag of words’ (PV-DBOW)
    d2v.build_vocab(corpus)

    logging.info("Training Doc2Vec model")
    # 10 epochs take around 10 minutes on my machine (i7), if you have more time/computational power make it 20
    for epoch in range(50):
        logging.info('Training iteration #{0}'.format(epoch))
        d2v.train(corpus, total_examples=d2v.corpus_count, epochs=d2v.iter)
        # shuffle the corpus
        random.shuffle(corpus)
        # decrease the learning rate
        d2v.alpha -= 0.0002
        # fix the learning rate, no decay
        d2v.min_alpha = d2v.alpha

    logging.info("Saving trained Doc2Vec model")
    d2v.save("d2v.model")
    return d2v

def read_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter="\t")
    corpus = df2labeled_sentence(dataset)
    return(corpus)


# def train_classifier(dataset):
#     logging.info("Classifier training")
#     model = RandomForestClassifier(n_estimators = 100)
#     vectors = np.array(dataset.vector.tolist())
#     labels = np.array(dataset.Class)
#     model.fit(vectors, labels)
#     training_predictions = model.predict(vectors)
#     logging.info('Training predicted classes: {}'.format(np.unique(training_predictions)))
#     logging.info('Training accuracy: {}'.format(accuracy_score(labels, training_predictions)))
#     logging.info('Training F1 score: {}'.format(f1_score(labels, training_predictions, average='weighted')))
#     return model

    # nrps = dataset[dataset.Class == 'nrps']
    # nrps_labeled = label_sentences(nrps, 'nrps')
    # pks = dataset[dataset.Class == 'pks']
    # pks_labeled = label_sentences(nrps, 'pks')
    # print(nrps_labeled[1])
    # print(pks_labeled[2])

def csv2vector(path, d2v_model):
    dataset = pd.read_csv(path, header=0, delimiter="\t")
    dataset['vector'] = dataset['Text'].apply(lambda x: d2v_model.infer_vector(x, steps = 1000))
    return(dataset)


def pca_of_vecs(dataset):
    pca = PCA(n_components=2)
    pca_vecs = pca.fit_transform(dataset['vector'].tolist())# = dataset['vector'].apply(lambda x: pca.fit_transform([x]))#
    dataset['pca_vectors_A'] = pca_vecs[:,0]
    dataset['pca_vectors_B'] = pca_vecs[:,1]
    return(dataset)

def plot_pca_scatter(dataset):
    fig = plt.figure(figsize=(8,8))
    labl = dataset.Class
    color = []
    for l in labl:
        if 'mix' in l:
          color.append('green')
        elif 'pks' in l:
          color.append('blue')
        else:
          color.append('yellow')
    #color=  ['red' if 'pks' in l else 'green' for l in labl]
    plt.scatter(dataset['pca_vectors_A'],dataset['pca_vectors_B'],color = color)
    plt.savefig('scatter.png')


if __name__ == "__main__":

    #######################
    #create the Model
    #######################

    # corpus = read_dataset('cluster_text.csv')
    # train_doc2vec(corpus)
    d2v_model = Doc2Vec.load('d2v.model')
    #print(d2v_model.docvecs)

    ######################
    #Plot the PCA
    ######################

    # id_dict = {}
    # for vec_id in d2v_model.docvecs.doctags:
    #     id_dict[vec_id] = [d2v_model.docvecs[vec_id]]

    # dataset = pd.DataFrame.from_dict(id_dict, orient='index')
    # dataset.rename(columns= {0:'vector'}, inplace = True)
    # dataset['Class'] = dataset.index

    # # print(dataset)
    # # exit()
    # dataset = pca_of_vecs(dataset)

    # plot_pca_scatter(dataset)

    ######################
    #Predict clusters
    ######################

    new_vec = d2v_model.infer_vector(['ACP','ACP','KS','AT','ACP','ACP','ACP','AT','ACP'], steps = 500, alpha = 0.025)
    sims = d2v_model.docvecs.most_similar(positive =[new_vec])
    print(sims)

    #dataset = pd.read_csv(path, header=0, delimiter="\t")
    #dataset = csv2vector('cluster_text.csv', d2v_model)
    # dataset.to_pickle('d2v_vecs.pickle')
    # dataset = pd.read_pickle('d2v_vecs.pickle')
    #dataset = pca_of_vecs(dataset)
    #plot_pca_scatter(dataset)
    #print(dataset)

    # for doc in d2v_model.docvecs:
    #   print(doc)
    # new_vec = d2v_model.infer_vector(['AT', 'KS', 'AT','AT','AT'], steps = 500)
    # new_vec = d2v_model.infer_vector(['A', 'C', 'C','C','C'], steps = 500)
    # sims = d2v_model.docvecs.most_similar([new_vec])
    # print(sims)

    # clf = train_classifier(dataset)
    # new_vec = d2v_model.infer_vector(['AT', 'KS', 'AT'], steps = 500)
    # training_predictions = clf.predict([new_vec])
    # print(training_predictions)
    # training_predictions = clf.predict_proba([new_vec])
    # print(training_predictions)
    # # print(dataset)
    # exit()






    # new_vector =  d2v_model.infer_vector(['AT', 'AT','AT', 'KS',], steps = 500)
    # sims = d2v_model.docvecs.most_similar([new_vector])
    # print(sims)


