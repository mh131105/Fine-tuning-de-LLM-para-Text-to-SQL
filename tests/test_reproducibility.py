import pytest
import sys
import os
import random
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.reproducibility import set_global_seed

def test_global_seed(tmp_path):
    set_global_seed(42)
    val1 = random.random()
    val2 = np.random.rand()
    
    set_global_seed(42)
    val3 = random.random()
    val4 = np.random.rand()
    
    assert val1 == val3
    assert val2 == val4
