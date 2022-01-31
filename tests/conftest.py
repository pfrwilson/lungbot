import os

import pytest

ROOT = '/Users/paulwilson/data/node_21/cxr_images/proccessed_data'

@pytest.fixture
def root():

    return ROOT

