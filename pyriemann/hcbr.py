import json
from random import randint
from subprocess import Popen, PIPE

import numpy as np
import os
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, matthews_corrcoef, balanced_accuracy_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, RepeatedKFold
from sklearn.datasets import load_svmlight_file

this_directory = os.path.dirname(os.path.abspath(__file__))

HCBR_BIN = this_directory + '\\HCBR.exe'

casebase = this_directory + '\\casebase.txt'
outcomes = this_directory  + '\\outcomes.txt'

def equals(x1, x2):
  for i1, i2 in zip(x1, x2):
    for j1, j2 in zip(i1, i2):
      if not j1 == j2:
        return False
  return True

class HCBRClassifier(BaseEstimator, ClassifierMixin):  

    def __init__(self, params_file, verbose=0, X=None, y=None, processVector=lambda v:v):
        self.verbose = verbose
        self.log("Initializing...")
        self.processVector = processVector
        self.params_file = params_file
        self.params = None
        self.X = X
        self.y = y
        self.local_training_param_file = this_directory + '\\training.params.json'
        self.HCBR_BIN = HCBR_BIN

    def log(self, *values, level = 1):
      if self.verbose >= level:
        print("[HCBR] ", *values)

    def vectorize(self, X):
      nbEpoch = len(X)
      nbChannels = len(X[0])
      nbSamples = len(X[0][0])
      vectors = X.reshape(nbEpoch, nbChannels*nbSamples)
      return [self.processVector(v) for v in vectors]

    def dump_casebase(self, X):
      with open(casebase, "w") as f:
        vectors = self.vectorize(X)
        self.log("size of vector is", len(vectors[0]))
        f.write(" ".join(str(e) for e in vectors[0]))
        f.write("\n")
        for v in vectors:
          f.write(" ".join(str(e) for e in v))
          f.write("\n")

    def dump_outcomes(self, y):
      with open(outcomes, "w") as f:
          if y is not None:
            f.write(str(y[0])+"\n")
            f.write("".join(str(e)+"\n" for e in y))

    def fit(self, X, y=None):
        self.log("Fitting", X.shape)
        # Check parameters
        try:
            self.log("Loading parameter file ", self.params_file)
            self.params = json.load(open(self.params_file))
        except Exception as e:
            self.log("Could not load parameter file...", e)
            return None

        # Modifying configuration
        try:
            self.log("Auto-config parameter file...")
            self.params["input"]["casebase"] = casebase
            self.params["input"]["outcomes"] = outcomes
            self.params['serialization']['serialize'] = True
            self.params['parameters']['no_prediction'] = True
            self.params['deserialization']['deserialize'] = False
            self.params['parameters']['limit'] = len(X) 
            with open(self.local_training_param_file, 'w') as f:
                f.write(json.dumps(self.params, indent=4))
        except Exception as e:
            self.log("Could not modify and save the parameter file: ", e)
            return None

        # Build the model and output the files
        try:
            self.log("Building model and serializing..")
            self.dump_casebase(X)
            self.dump_outcomes(y)
            cmd = [self.HCBR_BIN, '--params', self.local_training_param_file]
            p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
            output, err = p.communicate()
            self.log("output = ", output, level=2)
            self.log("error = ", error, level=2)
        except Exception as e:
            self.log("Could not build the model: ", e)
            return None

        return self

    def index_of(self, x):
      el = min(self.X, key=lambda e:abs(np.trace(e)-np.trace(x)))
      for i in range(len(self.X)):
        if equals(self.X[i], el):
          return i
      return -1

    def labels_of(self, X):
      self.log("Getting Xs with closest traces")
      indexes = [self.index_of(x) for x in X]
      iLen = len(indexes)
      iUniqueLen = len(np.unique(indexes))
      if iLen != iUniqueLen:
        self.log("[Warning] Could not retrieve all labels. Error estimate is (%): ", (iLen - iUniqueLen)/iLen * 100)
      return [self.y[i] for i in indexes]

    def predict(self, X, y=None):
        self.log("Predicting", X.shape)
        if y is None:
          self.log("Labels are null. Auto-detect labels using self.X")
          y = self.labels_of(X)
        # Modifying configuration
        try:
            self.log("Auto-config parameter file...")
            self.params["input"]["casebase"] = casebase
            self.params["input"]["outcomes"] = outcomes
            self.params['serialization']['serialize'] = False
            self.params['parameters']['no_prediction'] = False
            self.params['deserialization']['deserialize'] = True
            self.params['parameters']['limit'] = 0
            self.params['parameters']['keep_offset'] = False
            self.params['parameters']['training_iterations'] = 0
            with open(self.local_training_param_file, 'w') as f:
                f.write(json.dumps(self.params, indent=4))
        except Exception as e:
            self.log("Could not modify and save the parameter file: ", e)
            return None

        # Build the model and output the files
        res = []
        try:
            self.dump_casebase(X)
            self.dump_outcomes(y)
            cmd = [self.HCBR_BIN, '--params', self.local_training_param_file]
            p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
            output, err = p.communicate()
            self.log("output = ", output, level=2)
            self.log("err = ", err, level=2)
            output = open('predictions.txt', 'r').read().strip()
            res = [int(o.split()[2]) for o in output.splitlines()[1:]]
        except Exception as e:
            self.log("Could not make prediction: ", e)
            return None
        return res

    def score(self, X, y=None):
        self.log("Scoring")
        pred = self.predict(X, y)
        return balanced_accuracy_score(y, pred)

