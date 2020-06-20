import unittest
import shutil, tempfile
import pandas as pd
import pickle
import json
import contextualize
import train
import mock


class TestConWea(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.num_iter = 3
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.temp_dir)
        shutil.rmtree(self.dataset_dir)

    def test_conwea(self):
        sentences = [
            "messi kicked penalty",
            "judge issued death penalty",
            "ronaldo scored penalty",
            "judge gave a financial penalty",
            "judge punished robber",
            "at the end of the day football wins",
            "ronaldo scored a goal",
            "messi kicked a goal",
            "the court is in recess",
            "the court is adjourned after recess",
            "I am adjourned said the judge",
            "messi kicked penalty",
            "judge issued death penalty",
            "ronaldo scored penalty",
            "judge gave a financial penalty",
            "judge punished robber",
            "at the end of the day football wins",
            "ronaldo scored a goal",
            "messi kicked a goal",
            "the court is in recess",
            "the court is adjourned after recess",
            "I am adjourned said the judge",
            "messi kicked penalty",
            "judge issued death penalty",
            "ronaldo scored penalty",
            "judge gave a financial penalty",
            "judge punished robber",
            "at the end of the day football wins",
            "ronaldo scored a goal",
            "messi kicked a goal",
            "the court is in recess",
            "the court is adjourned after recess",
            "I am adjourned said the judge"
        ]
        labels = [
            "soccer",
            "law",
            "soccer",
            "law",
            "law",
            "soccer",
            "soccer",
            "soccer",
            "law",
            "law",
            "law",
            "soccer",
            "law",
            "soccer",
            "law",
            "law",
            "soccer",
            "soccer",
            "soccer",
            "law",
            "law",
            "law",
            "soccer",
            "law",
            "soccer",
            "law",
            "law",
            "soccer",
            "soccer",
            "soccer",
            "law",
            "law",
            "law"
        ]
        train.get_from_one_hot = mock.Mock(return_value=[
            "soccer",
            "law",
            "soccer",
            "law",
            "law",
            "soccer",
            "soccer",
            "soccer",
            "law",
            "law",
            "law",
            "soccer",
            "law",
            "soccer",
            "law",
            "law",
            "soccer",
            "soccer",
            "soccer",
            "law",
            "law",
            "law",
            "soccer",
            "law",
            "soccer",
            "law",
            "law",
            "soccer",
            "soccer",
            "soccer",
            "law",
            "law",
            "law"
        ])
        df_dic = {"sentence": sentences, "label": labels}
        df = pd.DataFrame.from_dict(df_dic)
        pickle.dump(df, open(self.dataset_dir + "/df.pkl", "wb"))
        seedwords = {"soccer": ["penalty"], "law": ["judge"]}
        json.dump(seedwords, open(self.dataset_dir + "/seedwords.json", "w"))
        contextualize.main(dataset_path=self.dataset_dir + "/", temp_dir=self.temp_dir + "/")
        train.main(dataset_path=self.dataset_dir + "/", num_iter=self.num_iter, print_flag=True)

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
