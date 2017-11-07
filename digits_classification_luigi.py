# coding=utf-8

import h5py
import logging
import pandas as pd
import luigi
import time

from sklearn import datasets
from sklearn import metrics
from sklearn import svm

IMAGES_AND_LABELS_HDF5_FILE = 'digit_clf_images_and_labels.hdf5'
RESHAPED_IMAGES_AND_LABELS_HDF5_FILE = 'digit_clf_reshaped_images_and_labels.hdf5'
CLASSIFIER_PKL = 'digit_clf_classifier.pkl'
CONFUSION_MATRIX_PKL = 'digit_clf_confusion_matrix.pkl'
IMAGES_H5_KEY = 'images'
LABELS_H5_KEY = 'labels'


class GetImagesDatasetTask(luigi.Task):

    def run(self):
        time.sleep(2000)
        logging.info('Getting Dataset')
        digits = datasets.load_digits()
        images_and_labels_h5 = h5py.File(self.output().path, 'w')
        images_and_labels_h5.create_dataset(name='images', data=digits.images)
        images_and_labels_h5.create_dataset(name='labels', data=digits.target)

    def output(self):
        return luigi.LocalTarget(IMAGES_AND_LABELS_HDF5_FILE)


class PreprocessImagesTask(luigi.Task):

    def requires(self):
        return GetImagesDatasetTask()

    def output(self):
        return luigi.LocalTarget(RESHAPED_IMAGES_AND_LABELS_HDF5_FILE)

    def run(self):
        logging.info('Pre-processing images')
        hf = h5py.File(self.requires().output().path, 'r')
        images, labels = hf[IMAGES_H5_KEY].value, hf[LABELS_H5_KEY].value
        reshaped_images = images.reshape((len(images), -1))
        reshaped_images_and_labels_h5 = h5py.File(self.output().path, 'w')
        reshaped_images_and_labels_h5.create_dataset(name=IMAGES_H5_KEY, data=reshaped_images)
        reshaped_images_and_labels_h5.create_dataset(name=LABELS_H5_KEY, data=labels)


class TrainingClassifierTask(luigi.Task):

    def requires(self):
        return PreprocessImagesTask()

    def output(self):
        return luigi.LocalTarget(CLASSIFIER_PKL)

    def run(self):
        logging.info('Training Classifier')
        hf = h5py.File(self.requires().output().path, 'r')
        images, target = hf[IMAGES_H5_KEY].value, hf[LABELS_H5_KEY].value
        n_samples = len(images)
        classifier = svm.SVC(gamma=0.001)
        # We learn the digits on the first half of the digits
        classifier.fit(images[:n_samples // 2], target[:n_samples // 2])
        pd.to_pickle(classifier, self.output().path)


class PredictionTask(luigi.Task):

    def requires(self):
        return TrainingClassifierTask()

    def output(self):
        return luigi.LocalTarget(CONFUSION_MATRIX_PKL)

    def run(self):
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
        pd.to_pickle(confusion_matrix, CONFUSION_MATRIX_PKL)
        logging.info(confusion_matrix)


def setup_logging():
    logging.basicConfig(format='[%(asctime)s - %(name)s - %(levelname)s] : %(message)s', level=logging.INFO)


if __name__ == '__main__':
    setup_logging()
    luigi.run()
