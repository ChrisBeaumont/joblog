import pymongo
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression

from . import Job, JobFactory


class TestJob(object):

    def setup_class(self):
        conn = pymongo.MongoClient('localhost')
        self.db = conn.joblog_test

    def setup_method(self, method):
        self.clf = LogisticRegression
        self.x = np.random.normal(0, 1, (10, 3))
        self.y = np.random.randint(0, 2, (10))
        self.params = dict(penalty='l2', C=2)

        self.db.drop_collection('test')
        self.collection = self.db['test']

    def test_run(self):
        j = Job(self.clf, self.x, self.y, self.params, self.collection)
        clf = j.run()

        assert isinstance(clf, LogisticRegression)

        clf2 = self.clf(**self.params).fit(self.x, self.y)
        np.testing.assert_array_equal(clf.predict(self.x),
                                      clf2.predict(self.x))

    def test_result_property(self):
        j = Job(self.clf, self.x, self.y, self.params, self.collection)
        assert j.result is None
        j.run()
        assert j.result is not None

    def test_nonunique_cached(self):
        j = Job(self.clf, self.x, self.y, self.params, self.collection)

        x = self.x.copy()
        y = self.y.copy()
        j2 = Job(self.clf, x, y, self.params, self.collection)

        assert j.result is None
        assert j2.result is None

        clf1 = j2.run()

        assert j.result is not None

    def test_detect_unique(self):
        """Each unique input gets a new entry"""

        j = Job(self.clf, self.x, self.y, self.params, self.collection)
        assert j.result is None
        j.run()

        self.x[0] += 5
        j = Job(self.clf, self.x, self.y, self.params, self.collection)
        assert j.result is None
        j.run()

        self.y[0] += 5
        j = Job(self.clf, self.x, self.y, self.params, self.collection)
        assert j.result is None
        j.run()

        self.params = {}
        j = Job(self.clf, self.x, self.y, self.params, self.collection)
        assert j.result is None
        j.run()

        self.clf = LinearRegression
        j = Job(self.clf, self.x, self.y, self.params, self.collection)
        assert j.result is None
        j.run()

    def test_rerun_overrides_cache(self):
        j = Job(self.clf, self.x, self.y, self.params, self.collection)

        clf1 = j.run()

        #hack: modify x in place after running job.
        #      rerunning will change classification rule
        self.x[:] *= 100
        clf1b = j.run()   # should not rerun
        clf2 = j.rerun()  # should rerun

        assert (clf1.coef_ == clf1b.coef_).all()
        assert not (clf1.coef_ == clf2.coef_).all()

    def test_nosave(self):
        j = Job(self.clf, self.x, self.y, self.params, self.collection)
        clf1 = j.run(store=None)
        assert j.result is None


    def test_save_score(self):

        j = Job(self.clf, self.x, self.y, self.params, self.collection)
        clf1 = j.run(store='score')
        assert j.result == clf1.score(self.x, self.y)



    def test_duplicate(self):
        j = Job(self.clf, self.x, self.y, self.params, self.collection)
        assert not j.duplicate

        j2 = Job(self.clf, self.x, self.y, self.params, self.collection)
        assert not j.duplicate
        assert j2.duplicate


class TestJobFactory(object):
    def setup_method(self, method):
        conn = pymongo.MongoClient('localhost')
        db = conn.joblog_test
        db.drop_collection('test')

    def make_factory(self):
        return JobFactory('localhost', 'joblog_test', 'test')

    def test_make_job(self):
        jf = self.make_factory()
        X, Y = np.random.normal(0, 1, (10, 3)), np.random.randint(0, 2, (10))

        j = jf.job(LogisticRegression, X, Y, {})
        assert isinstance(j, Job)

    def test_iter_jobs(self):
        jf = self.make_factory()
        X, Y = np.random.normal(0, 1, (10, 3)), np.random.randint(0, 2, (10))

        pg = dict(loss=['l1', 'l2'], C=[.01, .1, 1])
        assert len(list(jf.iter_jobs(LogisticRegression, X, Y, pg))) == 6

        #filters duplicates
        assert len(list(jf.iter_jobs(LogisticRegression, X, Y, pg))) == 0

        #turn off filtering
        assert len(list(jf.iter_jobs(LogisticRegression,
                                     X, Y, pg, False))) == 6
