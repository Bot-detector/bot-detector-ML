import logging
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class classifier(RandomForestClassifier):
    """
    This class is a wrapper for RandomForestClassifier.
    It adds the ability to save and load the model.
    """
    
    path = "api/MachineLearning/models"
    loaded = False

    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.name = name

    def __best_file_path(self, startwith):
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
        model_path = df_files["path"].iloc[0]
        return model_path

    def load(self):
        try:
            path = self.__best_file_path(self.name)
            logger.debug(f"Loading: {self.name}, {path}")
            object = joblib.load(path)
        except Exception as exception:
            logger.warning(f"Error when loading {self.name}: {exception}")
            return
        object.loaded = True
        return object

    def save(self) -> None:
        logger.debug(f"Saving classifier: {self.name}")
        today = int(time.time())
        joblib.dump(
            self,
            f"{self.path}/{self.name}_{today}_{round(self.accuracy, 2)}.joblib",
            compress=3,
        )

    def score(self, test_y, test_x):
        labels = np.unique(test_y).tolist()
        print(labels)

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

        print(classification_report(test_y, pred_y, target_names=labels))
        return self.accuracy, self.roc_auc
