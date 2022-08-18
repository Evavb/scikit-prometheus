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
        MetricRegistry.add_histogram("classification", "???", buckets=buckets, additional_labels=("method", "model", "value"))

    def predict_proba(self, X):
        """Predict class probabilities for X.
        The predicted class probabilities of an input sample is computed as
        the mean predicted class probabilities of the base estimators in the
        ensemble. If base estimators do not implement a ``predict_proba``
        method, then it resorts to voting and the predicted class probabilities
        of an input sample represents the proportion of estimators predicting
        each class.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        # Check data
        X = self._validate_data(
            X,
            accept_sparse=["csr", "csc"],
            dtype=None,
            force_all_finite=False,
            reset=False,
        )

        # Parallel loop
        n_jobs, _, starts = _partition_estimators(self.n_estimators, self.n_jobs)

        all_proba = Parallel(
            n_jobs=n_jobs, verbose=self.verbose, **self._parallel_args()
        )(
            delayed(_parallel_predict_proba)(
                self.estimators_[starts[i] : starts[i + 1]],
                self.estimators_features_[starts[i] : starts[i + 1]],
                X,
                self.n_classes_,
            )
            for i in range(n_jobs)
        )

        for idx, _ in 
        self.classes_.take(np.argmax(all_proba[0][0]))
        MetricRegistry.classification(method="bagging", model="", value="")

        # Reduce
        proba = sum(all_proba) / self.n_estimators

        return proba
