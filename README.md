joblog
==========

Job management for scikit learn classifiers. Inspired by / copied from Joseph Turian https://github.com/turian/batchtrain/blob/master/jobman.py

This package provides an interface to transparently save Scikit-Learn classifiers
to a Mongo database. Features include:

 * Transparent saving of trained classifiers to a (potentially non-local) mongo database
 * Knowledge of when a classifier has been trained with the same hyperparameters and training data

Dependencies
------------
 * scikit-learn
 * pymongo

Installation
------------

```
pip install git+git://github.com/ChrisBeaumont/joblog.git
```


Quick Guide
===========

```
from joblog import JobFactory
from sklearn.svm import SVC
from sklearn.datasets import load_iris

jf = JobFactory('localhost') # connect to a mongo database running on localhost

iris = load_iris()
X, Y = iris.data, iris.target
params = {'C': 5.0, 'kernel': 'rbf'}

job = jf.job(SVC, X, Y, params)
clf = job.run()   # train and return a classifier.
```

A pickled version of clf is auto-saved to the database whenever `job.run` is called.
Calling `job.run` again returns this cached result

```
clf2 = job.run()  # classifier is not re-trained
clf2 = job.rerun() # force it to refit
```
Similarly, if a new job is created with the same inputs, the cached result is retrieved

```
job = jf.job(SVC, X, Y, params)  # duplicate job
job.duplicate    # True
clf = job.run()  # cached result is returned
```

`joblog` uses checksums to detect changes in the input arrays.

```
X[0] += 1
job = jf.job(SVC, X, Y, params)  # X is different, this is a new job
job.duplicate    # False
clf = job.run()  # new result is computed
```
Similarly, any change in the classification class, training data, or hyperparameters
results in a new classifier being trained and stored

By default, `joblog` saves a pickled version of the trained classifier. This is
accessible via the `result` property:

```
clf = job.result  # possibly None, if job hasn't been run
```
However, you can assign any (pickleable) value to `job.result`, which is saved
in the database

```
clf = job.run()
job.result = job.score(X, Y)
job.run() # now returns the score
```

Specifying Database Details
---
```
 #connect to a remote database
jf = JobFactory('mongodb://<dbuser>:<dbpassword>@dbh29.mongolab.com:xyz/abc')

 # specify which database and collection to use
jf = JobFactory(db='db_name', collection='collection_name')

```

Parameter Search
-------

```
jf = JobFactory()
param_grid = {C=[.1, 1, 10], kernel=['linear', 'rbf']}
for job in jf.iter_jobs(SVC, X, Y, param_grid):
	job.run()
```

By default, `iter_jobs` skips duplicate jobs. You can turn this off

```
for job in jf.iter_jobs(SVC, X, Y, param_grid, filter_duplicates=False):
	job.rerun()	# force retraining
```