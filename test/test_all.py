# RUNNING TEST AFTER INSTALLING 'WISDEM' acc. to
# = https://github.com/WISDEM/WISDEM/tree/master
# ERROR: "ModuleNotFoundError: No module named 'wisdem.ccblade._bem'"
# NOTE: still UNRESOLVED/OPEN issue on github !!!
# = https://github.com/WISDEM/CCBlade/issues/31

# ============ X =============

import os
import sys

import pytest

testpath = (
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    + os.path.sep
    + "wisdem"
    + os.path.sep
    + "test"
    + os.path.sep
)

if __name__ == "__main__":
    sys.exit(pytest.main([testpath]))
