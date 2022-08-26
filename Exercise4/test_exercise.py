import os
import sys

from evaluate import evaluate_on_dataset

def test_exercise():

    if not os.path.exists('validation.bin'):
        os.system('wget https://www.dropbox.com/s/dz5prcyv4lgx779/validation.bin')
    path_to_ds = 'validation.bin'

    loss = evaluate_on_dataset(path_to_ds)

    assert loss < 0.06


