import pytest

from pypolsar.io import loadrat


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("./data/chirp_2d_test.rat", "Hello Jeanette!"),
        ("Raven", "Hello Raven!"),
    ],
)
def test_io(path, expected):
    """Example test with parametrization."""
    assert loadrat(path) == expected
