import logging
import random

import numpy as np
import pandas as pd

from gensim.models import doc2vec
from gensim.models.doc2vec import Doc2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def label_sentences(corpus, label_type, cla):
    """
    Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
    We do this by using the LabeledSentence method. The format will be "TRAIN_i" or "TEST_i" where "i" is
    a dummy index of the review.
    """

    labeled = []
    for i, v in enumerate(corpus):
        label = label_type + '_' + str(i)
        labeled.append(doc2vec.LabeledSentence(v.split(), [label]))
    return labeled


def read_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter="\t")
    x_train, x_test, y_train, y_test = train_test_split(dataset.Text, dataset.Class, random_state=0, test_size=0.1)
    x_train = label_sentences(x_train, 'Train', y_train)
    x_test = label_sentences(x_test, 'Test', y_test)
    all_data = x_train + x_test
    return x_train, x_test, y_train, y_test, all_data


def get_vectors(doc2vec_model, corpus_size, vectors_size, vectors_type):
    """
    Get vectors from trained doc2vec model
    :param doc2vec_model: Trained Doc2Vec model
    :param corpus_size: Size of the data
    :param vectors_size: Size of the embedding vectors
    :param vectors_type: Training or Testing vectors
    :return: list of vectors
    """
    vectors = np.zeros((corpus_size, vectors_size))
    for i in range(0, corpus_size):
        prefix = vectors_type + '_' + str(i)
        vectors[i] = doc2vec_model.docvecs[prefix]
    return vectors


def train_doc2vec(corpus):
    logging.info("Building Doc2Vec vocabulary")
    d2v = doc2vec.Doc2Vec(min_count=1,  # Ignores all words with total frequency lower than this
                          window=10,  # The maximum distance between the current and predicted word within a sentence
                          vector_size=300,  # Dimensionality of the generated feature vectors
                          workers=5,  # Number of worker threads to train the model
                          alpha=0.025,  # The initial learning rate
                          min_alpha=0.00025,  # Learning rate will linearly drop to min_alpha as training progresses
                          dm=1)  # dm defines the training algorithm. If dm=1 means ‘distributed memory’ (PV-DM)
                                 # and dm =0 means ‘distributed bag of words’ (PV-DBOW)
    d2v.build_vocab(corpus)

    logging.info("Training Doc2Vec model")
    # 10 epochs take around 10 minutes on my machine (i7), if you have more time/computational power make it 20
    for epoch in range(20):
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


def train_classifier(d2v, training_vectors, training_labels):
    logging.info("Classifier training")
    train_vectors = get_vectors(d2v, len(training_vectors), 300, 'Train')
    model = LogisticRegression()
    model.fit(train_vectors, np.array(training_labels))
    training_predictions = model.predict(train_vectors)
    logging.info('Training predicted classes: {}'.format(np.unique(training_predictions)))
    logging.info('Training accuracy: {}'.format(accuracy_score(training_labels, training_predictions)))
    logging.info('Training F1 score: {}'.format(f1_score(training_labels, training_predictions, average='weighted')))
    return model


def test_classifier(d2v, classifier, testing_vectors, testing_labels):
    logging.info("Classifier testing")
    test_vectors = get_vectors(d2v, len(testing_vectors), 300, 'Test')
    print(test_vectors)
    testing_predictions = classifier.predict(test_vectors)
    logging.info('Testing predicted classes: {}'.format(np.unique(testing_predictions)))
    logging.info('Testing accuracy: {}'.format(accuracy_score(testing_labels, testing_predictions)))
    logging.info('Testing F1 score: {}'.format(f1_score(testing_labels, testing_predictions, average='weighted')))


def test_defaul_corpus(d2v, corpus):
    vectors = []
    labels = []
    for label, corp in corpus.items():
        vectors.append(d2v.infer_vector(corp))
        labels.append(label)
    return(vectors, labels)

if __name__ == "__main__":
    x_train, x_test, y_train, y_test, all_data = read_dataset('cluster_text.csv')

    # d2v_model = train_doc2vec(all_data)
    d2v_model = Doc2Vec.load('d2v.model')
    new_vector =  d2v_model.infer_vector(['KS','AT','AT','ACP','AT','A','A','A','A'], steps = 500)

    classifier = train_classifier(d2v_model, x_train, y_train)
    testing_predictions = classifier.predict([new_vector])
    print(testing_predictions)
    testing_predictions = classifier.predict_proba([new_vector])
    print(testing_predictions)
    exit()
    # sims = d2v_model.docvecs.most_similar([new_vector])
    # print(sims)
    # exit()
    # corpus = {  'nrps':['E','E','C'],
    #             'pks':['KS','AT','ACP','AT','AT']
    #             }

    #x_test, y_test = test_defaul_corpus(d2v_model, corpus)
    # print(x_test)
    # print(y_test)
    # exit()

    # print(x_test)
    # print(y_test)


    print(test_vector1)
    test_vector2 =  d2v_model.infer_vector(['KS','AT','ACP','AT','AT'])
    print(test_vector2)
    # # test_vector3 =  d2v_model.infer_vector(['KS','AT','ACP','AT','AT'])
    # # print(test_vector3)
    # testing_predictions = classifier.predict([test_vector1, test_vector2])
    # print(testing_predictions)
    # testing_predictions = classifier.predict_proba([test_vector1, test_vector2])
    # print(testing_predictions)
    # # exit()

    test_classifier(d2v_model, classifier, x_test, y_test)
