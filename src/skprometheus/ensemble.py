from functools import wraps
from turtle import shape

import numpy as np
from sklearn import ensemble

from skprometheus.metrics import MetricRegistry
from skprometheus.utils import get_feature_names


class BaggingClassifier(ensemble.BaggingClassifier):
    @wraps(ensemble.BaggingClassifier.__init__, assigned=["__signature__"])
    def __init__(self, buckets, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buckets = buckets
        MetricRegistry.add_histogram("classification", "???", buckets=buckets, additional_labels=("method", "class_name"))

    def predict(self, X):
        """Predict class for X.
        The predicted class of an input sample is computed as the class with
        the highest mean predicted probability. If base estimators do not
        implement a ``predict_proba`` method, then it resorts to voting.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        predicted_probabilitiy = self.predict_proba(X)
                
        for name, prob in zip(predicted_probabilitiy.argmax(axis=1), predicted_probabilitiy.max(axis=1)):
            MetricRegistry.classification(method="BaggingClassifier", class_name=name).observe(prob)
        
        return self.classes_.take((np.argmax(predicted_probabilitiy, axis=1)), axis=0)