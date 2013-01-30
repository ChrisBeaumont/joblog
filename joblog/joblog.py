"""
Utility for running batches of Scikit-learn jobs
across different machines and python sessions.

Jobs are encapsulated by a Job object, and are
transparently stored on a Mongo database.
"""
from hashlib import md5
from pymongo import MongoClient
import gridfs

try:
    from cPickle import dumps, load
except ImportError:
    from pickle import dumps, load


class JobFactory(object):
    """ Factory class to make create jobs, backed up to a
    specific database

    :param mongo_uri: URI for a mongo database to connect to
    :type mongo_uri: string

    :param db: Name of database to use. Defaults to job
    :type db: string

    :param collection_name: Optional collection to store jobs in.
    :type collection_name: string
    """
    def __init__(self, mongo_uri='localhost',
                 db='job',
                 collection_name='default'):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db]
        self.collection = self.db[collection_name]

    def job(self, clf, X, Y, params):
        """
        Create a new job to train a classifier class on training data

        :param clf: Classifier class to use
        :type clf: Scikit-learn classifier class (not instance)

        :param X: Training data array
        :type X: An [n-example, n-feature] numpy array

        :param Y: Training labels
        :type Y: An [n-example] numpy array

        :params: A dictionary of hyperparameters for the
                 classification class
        :type params: dict
        """
        return Job(clf, X, Y, params, self.collection)

    def iter_jobs(self, clf, X, Y, param_grid, filter_duplicates=True):
        """Return an iterator that produces jobs for
        all combinations of a parameter grid.

        Optionally, filter out duplicate jobs

        The parameter grid is a dictionary of lists,
        giving the options for each hyperparameter.
        For more, see sklearn.grid_search.IterGrid

        :param clf: Classifier class

        :param X: Training data

        :param Y: Labeled examples

        :param params: hyperparameter grid

        :param filter_duplicates: If True (default), exclude
        duplicated jobs
        """
        from sklearn.grid_search import IterGrid
        for pg in IterGrid(param_grid):
            j = self.job(clf, X, Y, pg)
            if filter_duplicates and j.duplicate:
                continue
            yield j


class Job(object):
    """Encapsulation of a single job.

    Job objects should not be created directly, but rather by callin
    :method:`skljob.JobRunner.job`

    :param clf: Classifier class. See :method:`skljob.JobRunner.job`
    :param X: Training vectors.  See :method:`skljob.JobRunner.job`
    :param Y: Training labels. See  See :method:`skljob.JobRunner.job`
    :param params:  See :method:`skljob.JobRunner.job`
    :param collection: Database collection object to store into
    """
    def __init__(self, clf, X, Y, params, collection):
        self.clf = clf
        self.X = X
        self.Y = Y
        self.params = params
        self.collection = collection
        self._fs = gridfs.GridFS(collection.database)

        self._x_hash = md5(X).hexdigest()
        self._y_hash = md5(Y).hexdigest()
        self._entry = dict(_x_hash = self._x_hash,
                           _y_hash = self._y_hash,
                           clf = dumps(clf),
                           params = params)

        self._duplicated = (self.collection.find_one(self._entry) is not None)
        if not self._duplicated:
            self.collection.insert(self._entry)

    def run(self):
        """
        If this job hasn't been run before, train the classifier
        on the data, and return the result. Otherwise, return the
        result of a previous run
        """
        r = self.result
        if r is not None:
            return r

        clf = self.clf()
        clf.set_params(**self.params)
        clf.fit(self.X, self.Y)

        self.result = clf
        return clf

    @property
    def result(self):
        entry = self.collection.find_one(self._entry)
        if entry is not None and 'result' in entry:
            result = load(self._fs.get(entry['result']))
            return result

    @result.setter
    def result(self, result):
        r = dumps(result)
        rid = self._fs.put(r)

        self.collection.update(self._entry,
                               {"$set" : {"result":rid}},
                               upsert=True)

    @property
    def duplicate(self):
        """
        Return whether this job is a duplicate (i.e. has the
        same parameters as another job that has been instantiated
        already).
        """
        return self._duplicated


    def rerun(self):
        """
        Regardless of whether this job has been run before,
        train the classifier on the data, and return the result
        """
        self.result = None
        return self.run()
