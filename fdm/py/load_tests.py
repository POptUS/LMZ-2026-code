from pathlib import Path


def load_tests(loader, suite, _):
    """
    This function implements the ``load_tests`` protocol of the python unittest
    package.

    Developers and users can run tests using this indirectly |via|::

                         python -m unittest fdm

    to run as part of the package's full suite or |via|::

                         python -m unittest fdm.fdm

    to run alone.

    :param loader: ``unittest.TestLoader`` instance doing the loading
    :param suite: test suite being built at the time of call to this function
    :return: Given suite with all ``fdm`` tests added in
    """
    start_dir = Path(__file__).resolve().parent.joinpath("tests")

    print(f"Discover tests in {start_dir}")

    pkg_tests = loader.discover(start_dir=str(start_dir),
                                top_level_dir=str(start_dir),
                                pattern="Test*.py")
    suite.addTests(pkg_tests)

    return suite
