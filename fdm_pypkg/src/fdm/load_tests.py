from .fdm.load_tests import load_tests as load_tests_fdm


def load_tests(loader, *_):
    """
    This function implements the ``load_tests`` protocol of the python unittest
    package so that clients using the package don't need to know where the
    tests are or what patterns they need to look for to find all tests.

    This function doesn't assume that it knows how to find all tests in the
    ``fdm`` tool.  Rather, it uses the ``load_tests`` function in that
    sub-package to gather tests.

    Developers and users can run tests using this indirectly |via|::

                         python -m unittest fdm

    :param loader: ``unittest.TestLoader`` instance doing the loading
    :return: ``unittest`` test suite that contains all tests to be run to
        check correctness of all code in the overall package
    """
    suite = loader.suiteClass()
    suite = load_tests_fdm(loader, suite, None)

    return suite
