import unittest
import shutil, tempfile
import pandas as pd
import pickle
import json
import contextualize
import argparse


class TestContextualize(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.temp_dir)
        shutil.rmtree(self.dataset_dir)

    def test_contextualize(self):
        sentences = ["messi kicked penalty", "judge issued death penalty", "ronaldo scored penalty",
                     "judge gave a financial penalty"]
        df_dic = {"sentence": sentences}
        df = pd.DataFrame.from_dict(df_dic)
        pickle.dump(df, open(self.dataset_dir + "/df.pkl", "wb"))
        seedwords = {"soccer": ["penalty"], "law": ["judge"]}
        json.dump(seedwords, open(self.dataset_dir + "/seedwords.json", "w"))
        contextualize.main(dataset_path=self.dataset_dir + "/", temp_dir=self.temp_dir + "/")
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
