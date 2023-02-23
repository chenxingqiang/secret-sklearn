"""Tests for making sure experimental imports work as expected."""

import textwrap

from sflearn.utils._testing import assert_run_python_script


def test_imports_strategies():
    # Make sure different import strategies work or fail as expected.

    # Since Python caches the imported modules, we need to run a child process
    # for every test case. Else, the tests would not be independent
    # (manually removing the imports from the cache (sys.modules) is not
    # recommended and can lead to many complications).

    good_import = """
    from sflearn.experimental import enable_iterative_imputer
    from sflearn.impute import IterativeImputer
    """
    assert_run_python_script(textwrap.dedent(good_import))

    good_import_with_ensemble_first = """
    import sflearn.ensemble
    from sflearn.experimental import enable_iterative_imputer
    from sflearn.impute import IterativeImputer
    """
    assert_run_python_script(textwrap.dedent(good_import_with_ensemble_first))

    bad_imports = """
    import pytest

    with pytest.raises(ImportError, match='IterativeImputer is experimental'):
        from sflearn.impute import IterativeImputer

    import sflearn.experimental
    with pytest.raises(ImportError, match='IterativeImputer is experimental'):
        from sflearn.impute import IterativeImputer
    """
    assert_run_python_script(textwrap.dedent(bad_imports))
