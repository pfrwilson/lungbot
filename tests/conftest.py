import os

import pytest


def data_root():
    
    # testing requires 
    # $ export DATAROOT=/root/to/data
    
    return os.getenv('DATA_ROOT')


