"""
Messing with Data file
"""

import numpy as np
import pandas as pd
import os


class MessingWithData:

    def __init__(self, dir, filename):
        self.dir = dir  # directory
        self.file = filename  # filename

    def read_file(self):
        # do this to fix reading problem to read_csv - dtype={"user_id": int, "username": object}
        file_1 = pd.read_csv(os.path.join(self.dir, self.file))
        # (8921483, 83) for train
        # (7853253, 82) for test
        print(file_1.shape)
        print(file_1.size)
        file_1.fillna(file_1.median())
        print(file_1)
        print(file_1.describe())


def main():
    m1 = MessingWithData('/Users/k.n./Downloads/microsoft-malware-prediction', 'test.csv')
    m1.read_file()


if __name__ == '__main__':
    main()
