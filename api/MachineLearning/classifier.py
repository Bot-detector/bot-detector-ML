import logging
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             roc_auc_score)

logger = logging.getLogger(__name__)

class classifier(RandomForestClassifier):
    """
    This class is a wrapper for RandomForestClassifier.
    It adds the ability to save and load the model.
    """
    path = "api/MachineLearning/models"
    loaded = False

    def __init__(self, name, path=None, **kwargs):
        """
        Initialize the classifier object.
        :param path: path to the models
        :param name: name of the classifier
        :param kwargs: keyword arguments for RandomForestClassifier
        """
        super().__init__(**kwargs)
        self.name = name
        self.path = path if path is not None else self.path

    def __best_file_path(self, startwith: str):
        """
        This method will return the best model path based on accuracy.
        :param startwith: name of the classifier
        :return: classifier object
        """
        files = []
        for f in os.listdir(self.path):
            if f.endswith(".joblib") and f.startswith(startwith):
                # accuracy is in filename, so we have to parse it
                model_file = f.replace(".joblib", "")
                model_file = model_file.split(sep="_")
                # save to dict
                d = {
                    "path": f"{self.path}/{f}",
                    "model": model_file[0],
                    "date": model_file[1],
                    "accuracy": model_file[2],
                }
                # add dict to array
                files.append(d)

        # array of dict can be used for pandas dataframe
        df_files = pd.DataFrame(files)
        df_files.sort_values(by=["date"], ascending=False, inplace=True)
        path = df_files["path"].iloc[0]
        return joblib.load(path)

    def load(self):
        """
        Loads the model object from the file.
        :return: classifier object
        """
        try:
            self = self.__best_file_path(self.name)
            logger.debug(f"Loading: {self.name}, {self.path}")
        except Exception as exception:
            logger.warning(f"Error when loading {self.name}: {exception}", exc_info=True)
            return
        self.loaded = True
        return self

    def save(self):
        """
        Save the classifier object to the file.
        :return: None
        """
        logger.debug(f"Saving classifier: {self.name}")
        today = int(time.time())
        joblib.dump(
            self,
            f"{self.path}/{self.name}_{today}_{round(self.accuracy, 2)}.joblib",
            compress=3,
        )


    def score(self, test_y, test_x):
        """
        Calculate the accuracy and roc_auc score for the classifier.
        :param test_y: test data labels
        :param test_x: test data features
        :return: accuracy and roc_auc score
        """
        labels = np.unique(test_y).tolist()

        # make predictions
        pred_y = self.predict(test_x)
        pred_proba_y = self.predict_proba(test_x)

        if len(labels) == 2:
            pred_proba_y = pred_y  # pred_proba_y[:, 1]

        self.accuracy = balanced_accuracy_score(test_y, pred_y)
        self.roc_auc = roc_auc_score(
            test_y, pred_proba_y, labels=labels, multi_class="ovo"
        )

        labels = ["Not bot", "bot"] if len(labels) == 2 else labels

        logger.info(classification_report(test_y, pred_y, target_names=labels))
        return self.accuracy, self.roc_auc