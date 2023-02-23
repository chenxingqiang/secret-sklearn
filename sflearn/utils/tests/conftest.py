import pytest

import sflearn


@pytest.fixture
def print_changed_only_false():
    sflearn.set_config(print_changed_only=False)
    yield
    sflearn.set_config(print_changed_only=True)  # reset to default
