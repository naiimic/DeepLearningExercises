import os
import sys

def test_part2():

    if not os.path.exists('validation_data'):
        os.system('wget https://www.dropbox.com/s/nwyy9tv2vir95ro/validation_data.zip')
        os.system('unzip validation_data.zip')
    path_to_ds = 'validation_data/'

    from evaluate_f1 import evaluate_on_dataset

    f1_edge, f1_node = evaluate_on_dataset(path_to_ds)

    assert (f1_edge > 0.85) and (f1_node > 0.85)

