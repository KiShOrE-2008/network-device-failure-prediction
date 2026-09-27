import os
import sys

# Ensure backend directory is set as working directory and added to sys.path during pytest runs
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BACKEND_DIR, "src")

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

os.chdir(BACKEND_DIR)

import pytest
import history_store

@pytest.fixture(autouse=True)
def setup_test_database():
    history_store.init_db()
    yield

