"""Conftest for API unit tests.

Sets E2I_TESTING_MODE to bypass authentication for unit tests.
"""

import os

# Enable testing mode BEFORE any auth module imports
os.environ["E2I_TESTING_MODE"] = "1"
