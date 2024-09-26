from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
import os
from pathlib import Path
import pandas as pd
import numpy as np
import wfdb
import ast


class PTBXLDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/ptb-xl/",
        train_val_test_split: Tuple[int, int, int] = (),
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """_summary_

        Args:
            data_dir (str, optional): _description_. Defaults to "data/".
            train_val_test_split (Tuple[int, int, int], optional): _description_. Defaults to ().
            batch_size (int, optional): _description_. Defaults to 32.
            num_workers (int, optional): _description_. Defaults to 0.
            pin_memory (bool, optional): _description_. Defaults to False.
        """

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_dir = data_dir

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.bath_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        """Get the number of classes

        Returns:
            int: the number of class used in IMLeNet (2)
        """
        return 2

    def prepare_data(self) -> None:

        return super().prepare_data()

    def setup(self):
        # get the absolute path to the ptb-xl data
        path = os.path.join(os.getcwd(), Path(self.data_dir))
        Y = pd.read_csv(os.path.join(path, "ptbxl_database.csv"),
                        index_col="ecg_id")  # Y.shape = (21799, 27)

        data = np.array([wfdb.rdsamp(os.path.join(path, f))[0]
                        for f in Y.filename_lr])  # data.shape = (21799, 1000, 12)

        # todo: scp_codes is a column in "ptbxl_database.csv", need to findout what it is
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        # print(Y.scp_codes)
        # ecg_id
        # 1                 {'NORM': 100.0, 'LVOLT': 0.0, 'SR': 0.0}
        # 2                             {'NORM': 80.0, 'SBRAD': 0.0}
        # 3                               {'NORM': 100.0, 'SR': 0.0}
        # 4                               {'NORM': 100.0, 'SR': 0.0}
        # 5                               {'NORM': 100.0, 'SR': 0.0}
        #                             ...
        # 21833    {'NDT': 100.0, 'PVC': 100.0, 'VCLVH': 0.0, 'ST...
        # 21834             {'NORM': 100.0, 'ABQRS': 0.0, 'SR': 0.0}
        # 21835                           {'ISCAS': 50.0, 'SR': 0.0}
        # 21836                           {'NORM': 100.0, 'SR': 0.0}
        # 21837                           {'NORM': 100.0, 'SR': 0.0}
        # Name: scp_codes, Length: 21799, dtype: object

        agg_df = pd.read_csv(os.path.join(
            path, "scp_statements.csv"), index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]
        # print(agg_df.shape) (44, 12)

        def MI_agg(y_dic):
            temp = []
            for key in y_dic.keys():
                if y_dic[key] in [100, 80, 0]:
                    if key in ["ASMI", "IMI"]:
                        temp.append(key)
            return list(set(temp))

        Y["diagnostic_subclass"] = Y.scp_codes.apply(MI_agg)
        Y["subdiagnostic_len"] = Y["diagnostic_subclass"].apply(
            lambda x: len(x))

        # MI sub-diseases ASMI and IMI
        x1 = data[Y["subdiagnostic_len"] == 1]
        y1 = Y[Y["subdiagnostic_len"] == 1]

        # print(y1)
        #         patient_id   age  sex  ...                filename_hr  diagnostic_subclass  subdiagnostic_len
        # ecg_id                         ...
        # 177        21551.0  73.0    0  ...  records500/00000/00177_hr               [ASMI]                  1
        # 181        21551.0  73.0    0  ...  records500/00000/00181_hr               [ASMI]                  1
        # 184        13112.0  74.0    0  ...  records500/00000/00184_hr               [ASMI]                  1
        # 189        13112.0  74.0    0  ...  records500/00000/00189_hr               [ASMI]                  1
        # 210        16062.0  58.0    0  ...  records500/00000/00210_hr                [IMI]                  1
        # ...            ...   ...  ...  ...                        ...                  ...                ...
        # 21805      16291.0  72.0    0  ...  records500/21000/21805_hr               [ASMI]                  1
        # 21815      14433.0  82.0    1  ...  records500/21000/21815_hr                [IMI]                  1
        # 21826       9178.0  82.0    1  ...  records500/21000/21826_hr                [IMI]                  1
        # 21827      13862.0  79.0    1  ...  records500/21000/21827_hr                [IMI]                  1
        # 21828      13862.0  79.0    1  ...  records500/21000/21828_hr                [IMI]                  1

        def norm_agg(y_dic):
            for key in y_dic.keys():
                if y_dic[key] in [100]:
                    if key == "NORM":
                        return "NORM"

        N = Y.copy()
        N["diagnostic_subclass"] = Y.scp_codes.apply(norm_agg)

        # Normal class
        x2 = data[N["diagnostic_subclass"] == "NORM"]
        y2 = N[N["diagnostic_subclass"] == "NORM"]

        # todo: Need to findout detailed about ptb_xl dataset (VERY DETAIL about each column in the csv file)

        # Train and test splits
        # * Arcording to this paper https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7248071/pdf/41597_2020_Article_495.pdf, "strat_fold" column just for split the train and test set
        # todo: why they set to split like that
        x1_train = x1[y1.strat_fold <= 8]
        y1_train = y1[y1.strat_fold <= 8]

        x1_test = x1[y1.strat_fold > 8]
        y1_test = y1[y1.strat_fold > 8]

        x2_train = x2[y2.strat_fold <= 2][:900]
        y2_train = y2[y2.strat_fold <= 2][:900]

        x2_test = x2[y2.strat_fold == 3][:200]
        y2_test = y2[y2.strat_fold == 3][:200]

        # todo: get the shape, datatype all of these
        X_train = np.concatenate((x1_train, x2_train), axis=0)
        X_test = np.concatenate((x1_test, x2_test), axis=0)

        y1_train.diagnostic_subclass = y1_train.diagnostic_subclass.apply(
            lambda x: x[0])
        y1_test.diagnostic_subclass = y1_test.diagnostic_subclass.apply(
            lambda x: x[0])
        y_train = np.concatenate(
            (y1_train.diagnostic_subclass.values,
             y2_train.diagnostic_subclass.values),
            axis=0,
        )
        y_test = np.concatenate(
            (y1_test.diagnostic_subclass.values,
             y2_test.diagnostic_subclass.values), axis=0
        )
    
        # print(X_train[0].shape) # (1000, 12)
        # print(y1_train.shape) # (1828, 29)

if __name__ == '__main__':
    DL = PTBXLDataModule()
    DL.setup()
