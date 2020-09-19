"""Common test support for all numpy test scripts.
This single module should provide all the common functionality for numpy tests
in a single location, so that test scripts can just import it and work right
away.
"""
from unittest import TestCase

from numpy._pytesttester import PytestTester

from ._private import decorators as dec
from ._private.nosetester import NoseTester as Tester
from ._private.nosetester import run_module_suite
from ._private.utils import *

__all__ = _private.utils.__all__ + ["TestCase", "run_module_suite"]

test = PytestTester(__name__)
del PytestTester
