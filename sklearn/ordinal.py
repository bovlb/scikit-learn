from typing import Iterable

import numpy as np

from .base import ClassifierMixin

def _transpose(l):
    """Swaps first and second indexes of nested array/list"""
    return list(zip(*l))


class OrdinalClassifier(ClassifierMixin):
    """This augments any sklearn-compatible classifier that supports multi-output classification
    to handle the single-output multi-class case where the classes are ordered, 
    for example if they represent the number of stars in a review.
    Implements the method outlined in https://www.cs.waikato.ac.nz/~eibe/pubs/ordinal_tech_report.pdf"""

    def __init__(self, classifier, classes: Iterable[str], scores=None):
        """Creates a classifier based on another classifier.
        
        Parameters
        ----------
        classifier : Underlying classifier to use. 
            This classifier must support multi-output classification, e.g. `RandomForestClassifier`.
            Other classifiers can be converted by wrapping them in `MultiOutputClassifier`.
        classes : Ordered iterable of class labels
        scores : Assignment of numerical values to classes.
            If not set, they are assigned evenly between 0.0 and 1.0.
            These scores are not used during fitting and prediction,
            but only for certain scoring methods.
        """
        self.classifier = classifier     
        self.classes = list(classes)
        
        assert len(self.classes) >= 2, "At least two classes are required"
        assert len(self.classes) == len(set(self.classes)), "Classes must be distinct"

        # In the underlying classifier, each class is represented as the set of classes subsequent in the list.
        # In particular, the last class is represented by no labels, and the first class label is unused.
        # Hence, in the underlying classifier, each label represents the claim that the true label precedes that label.
        # E.g. if the class labels are [Good, Neutral, Bad], then "Good" is represented as {Neutral, Bad}, where
        # Neutral means "before Neutral" which means Good, 
        # and Bad means "before Bad" which means either Good or Neutral.
                
        # This dictionary caches the multi-output representation for each input class.
        # https://scikit-learn.org/stable/modules/multiclass.html#multilabel-classification-format
        self.class_map_ = { c: [ 1 if j >= i else 0 for j in range(len(self.classes) - 1) ] for i, c in enumerate(self.classes) }
        
        if scores is not None:
            self.scores = list(scores)
        else:
            self.scores = [ x / (len(self.classes)-1) for x in range(len(self.classes)) ]
        assert len(self.scores) == len(self.classes), "scores must have same cardinality as classes"
 
    def _transform_y(self, y: Iterable[str]):
        """Converts from list of class labels into multi-label format."""
        return [self.class_map_[label] for label in y]
    
    def fit(self, X, y, *args, **kargs):
        """Fit the model according to the given training data.
        
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target vector relative to X drawn from the classes.            
        
        Any other arguments are passed to the underlying classifier.
        
        Returns
        -------
        self : object
        """
        y = self._transform_y(y)
        self.classifier.fit(X, y, *args, **kargs)
        return self
        
    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample are computed 
        according to the multi-output probabilities of the underlying classifier

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. 

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes`.
        """
        
        def convert(p):
            """Convert from multi-output format into single-output format"""
            # p2 represents the probability of the true label being to the left of each label,
            # for all class labels starting with the second
            p2 = p
            # p3 is the calcluated probabilities per class, truncated at zero
            p3 = [ p2[0], *(max(p2[i+1] - p2[i], 0) for i in range(0,len(self.classes)-2)), 1 - p2[-1] ]
            # We normalize the probabilities so they add to one.
            sum_p3 = sum(p3)
            p4 = [ x / sum_p3 for x in p3 ]
            return [ [1-p, p] for p in p4 ]
                
        # List of length n_classes-1 of arrays of length len(X) of pairs of probabilities.
        probs = self.classifier.predict_proba(X)
        
        return _transpose([
            convert([ p[i][1] for p in probs ])
            for i, x in enumerate(X)
        ])
    
    def predict_log_proba(self, X):
        """Returns logarithm of class probabilities"""
        return [  np.log(probs) for probs in self.predict_proba(X) ]
    
    def predict(self, X):
        """Returns best label"""
        probs = self.predict_proba(X)
        return [
            self.classes[max(range(len(p)), key = lambda i: p[i][1])]
            for p in _transpose(probs)
        ]

    def predict_score(self, X):
        """Produces a numerical representation of the predicted probability distribution
        across classes as a weighted mean of the scores assigned to each class.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        scores: array of shape = [n_samples]
        """
        probs = self.predict_proba(X)
        # probs is now a list of length len(classes)
        # of lists of length len(X)
        # of lists of length 2 representing the negative and positive probabilities

        def pd_to_score(pd):
            """Takes a probability distribution across classes and returns a single numerical score"""
            return np.dot(self.scores, pd)

        n_classes = len(probs)
        n_X = len(probs[0])
        result = [ pd_to_score([probs[c][i][1] for c in range(n_classes)]) for i in range(n_X) ]
        return result