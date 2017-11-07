# coding=utf-8

import h5py
import logging
import pandas as pd

from sklearn import datasets
from sklearn import metrics
from sklearn import svm

IMAGES_AND_LABELS_HDF5_FILE = 'digit_clf_images_and_labels.hdf5'
RESHAPED_IMAGES_AND_LABELS_HDF5_FILE = 'digit_clf_reshaped_images_and_labels.hdf5'
CLASSIFIER_PKL = 'digit_clf_classifier.pkl'
IMAGES_H5_KEY = 'images'
LABELS_H5_KEY = 'labels'


def get_images_dataset():
    logging.info('Getting Dataset')
    digits = datasets.load_digits()
    images_and_labels_h5 = h5py.File(IMAGES_AND_LABELS_HDF5_FILE, 'w')
    images_and_labels_h5.create_dataset(name='images', data=digits.images)
    images_and_labels_h5.create_dataset(name='labels', data=digits.target)


def preprocess_images():
    logging.info('Pre-processing images')
    hf = h5py.File(IMAGES_AND_LABELS_HDF5_FILE, 'r')
    images, labels = hf[IMAGES_H5_KEY].value, hf[LABELS_H5_KEY].value
    reshaped_images = images.reshape((len(images), -1))
    reshaped_images_and_labels_h5 = h5py.File(RESHAPED_IMAGES_AND_LABELS_HDF5_FILE, 'w')
    reshaped_images_and_labels_h5.create_dataset(name=IMAGES_H5_KEY, data=reshaped_images)
    reshaped_images_and_labels_h5.create_dataset(name=LABELS_H5_KEY, data=labels)


def train_classifier():
    logging.info('Training Classifier')
    hf = h5py.File(RESHAPED_IMAGES_AND_LABELS_HDF5_FILE, 'r')
    images, target = hf[IMAGES_H5_KEY].value, hf[LABELS_H5_KEY].value
    n_samples = len(images)
    classifier = svm.SVC(gamma=0.001)
    # We learn the digits on the first half of the digits
    classifier.fit(images[:n_samples // 2], target[:n_samples // 2])
    pd.to_pickle(classifier, CLASSIFIER_PKL)


def predict():
    logging.info('Predicting Images')
    hf = h5py.File(RESHAPED_IMAGES_AND_LABELS_HDF5_FILE, 'r')
    images, target = hf['images'].value, hf['labels'].value
    n_samples = len(images)
    classifier = pd.read_pickle(CLASSIFIER_PKL)
    expected = target[n_samples // 2:]
    predicted = classifier.predict(images[n_samples // 2:])
    confusion_matrix = metrics.confusion_matrix(expected, predicted)
    logging.info("Classification report for classifier %s:\n%s\n"
                 % (classifier, metrics.classification_report(expected, predicted)))
    logging.info("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
    return confusion_matrix


def setup_logging():
    logging.basicConfig(format='[%(asctime)s - %(name)s - %(levelname)s] : %(message)s', level=logging.INFO)


if __name__ == '__main__':
    setup_logging()
    get_images_dataset()
    preprocess_images()
    train_classifier()
    predict()
