from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder, MultiLabelBinarizer
from sklearn.utils import shuffle
import os
from pathlib import Path
import pandas as pd
import numpy as np
import wfdb
import ast
import matplotlib.pyplot as plt


class PTBXLDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/ptb-xl/",
        # train_val_test_split: Tuple[int, int, int] = (),
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        sub_disease: bool = True,
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

        self.X_train: Optional[Dataset] = None
        self.y_train: Optional[Dataset] = None
        self.X_test: Optional[Dataset] = None
        self.y_test: Optional[Dataset] = None
        self.X_val: Optional[Dataset] = None
        self.y_val: Optional[Dataset] = None

        self.data_dir = data_dir
        self.bath_size_per_device = batch_size

        self.scaler = StandardScaler()
        self.le = LabelEncoder()
        self.sub_disease = sub_disease

    @property
    def num_classes(self) -> int:
        """Get the number of classes

        Returns:
            int: the number of class used in IMLeNet (2)
        """
        return 2

    def prepare_data(self) -> None:

        return super().prepare_data()

    def setup(self, stage: Optional[str] = None):
        if self.sub_disease:
            self.X_train, self.y_train, self.X_test, self.y_test = self.preprocess_sub_disease()
        else:
            self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val = self.preprocess()

        # Standardizing the data
        self.scaler.fit(np.vstack(self.X_train).flatten()
                        [:, np.newaxis].astype(float))
        self.X_train = self.apply_scaler(self.X_train)
        self.X_test = self.apply_scaler(self.X_test)
        if not self.sub_disease:
            self.X_val = self.apply_scaler(self.X_val)

        # Convert to PyTorch tensors
        self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
        self.y_train = torch.tensor(self.y_train, dtype=torch.float32)
        self.X_test = torch.tensor(self.X_test, dtype=torch.float32)
        self.y_test = torch.tensor(self.y_test, dtype=torch.float32)

        if not self.sub_disease:
            self.X_val = torch.tensor(self.X_val, dtype=torch.float32)
            self.y_val = torch.tensor(self.y_val, dtype=torch.float32)

        # Reshape to match the TensorFlow shape (32, 12, 1000, 1)
        self.X_train = self.X_train.permute(
            0, 2, 1).unsqueeze(-1)      # (batch_size, channels, time_steps, 1)
        self.X_test = self.X_test.permute(
            0, 2, 1).unsqueeze(-1)      # Same for test data
        if not self.sub_disease:
            self.X_val = self.X_val.permute(
                0, 2, 1).unsqueeze(-1)  # Same for validation data

    def preprocess(self):
        """ Preprocesses the dataset with superclass """
        # print("Loading dataset...", end="\n" * 2)
        path = os.path.join(self.data_dir)
        Y = pd.read_csv(os.path.join(
            path, "ptbxl_database.csv"), index_col="ecg_id")
        data = np.array([wfdb.rdsamp(os.path.join(path, f))[0]
                        for f in Y.filename_lr])
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        agg_df = pd.read_csv(os.path.join(
            path, "scp_statements.csv"), index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]

        def agg(y_dic):
            temp = []
            for key in y_dic.keys():
                if key in agg_df.index:
                    c = agg_df.loc[key].diagnostic_class
                    if pd.notnull(c):
                        temp.append(c)
            return list(set(temp))

        Y["diagnostic_superclass"] = Y.scp_codes.apply(agg)
        Y["superdiagnostic_len"] = Y["diagnostic_superclass"].apply(
            lambda x: len(x))
        counts = pd.Series(np.concatenate(
            Y.diagnostic_superclass.values)).value_counts()
        Y["diagnostic_superclass"] = Y["diagnostic_superclass"].apply(
            lambda x: list(set(x).intersection(set(counts.index.values)))
        )
        X_data = data[Y["superdiagnostic_len"] >= 1]
        Y_data = Y[Y["superdiagnostic_len"] >= 1]
        # print("Preprocessing dataset...", end="\n" * 2)
        mlb = MultiLabelBinarizer()
        mlb.fit(Y_data["diagnostic_superclass"])
        y = mlb.transform(Y_data["diagnostic_superclass"].values)

        # Stratified split
        X_train = X_data[Y_data.strat_fold < 9]
        y_train = y[Y_data.strat_fold < 9]

        X_val = X_data[Y_data.strat_fold == 9]
        y_val = y[Y_data.strat_fold == 9]

        X_test = X_data[Y_data.strat_fold == 10]
        y_test = y[Y_data.strat_fold == 10]

        return X_train, y_train, X_test, y_test, X_val, y_val

    def preprocess_sub_disease(self):
        """Preprocess the sub-diagnostic diseases of MI."""
        # print("Loading dataset...", end="\n" * 2)
        path = os.path.join(self.data_dir)
        Y = pd.read_csv(os.path.join(
            path, "ptbxl_database.csv"), index_col="ecg_id")
        data = np.array([wfdb.rdsamp(os.path.join(path, f))[0]
                        for f in Y.filename_lr])
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        agg_df = pd.read_csv(os.path.join(
            path, "scp_statements.csv"), index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]

        def MI_agg(y_dic):
            temp = []
            for key in y_dic.keys():
                if y_dic[key] in [100, 80, 0]:
                    if key in agg_df.index:
                        if key in ["ASMI", "IMI"]:
                            temp.append(key)
            return list(set(temp))

        Y["diagnostic_subclass"] = Y.scp_codes.apply(MI_agg)
        Y["subdiagnostic_len"] = Y["diagnostic_subclass"].apply(
            lambda x: len(x))

        # MI sub-diseases ASMI and IMI
        x1 = data[Y["subdiagnostic_len"] == 1]
        y1 = Y[Y["subdiagnostic_len"] == 1]

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

        # Train and test splits
        x1_train = x1[y1.strat_fold <= 8]
        y1_train = y1[y1.strat_fold <= 8]

        x1_test = x1[y1.strat_fold > 8]
        y1_test = y1[y1.strat_fold > 8]

        x2_train = x2[y2.strat_fold <= 2][:900]
        y2_train = y2[y2.strat_fold <= 2][:900]

        x2_test = x2[y2.strat_fold == 3][:200]
        y2_test = y2[y2.strat_fold == 3][:200]

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

        # Encode labels
        le = LabelEncoder()
        y_train = self.to_categorical(le.fit_transform(y_train), 3)
        y_test = self.to_categorical(le.transform(y_test), 3)

        return X_train, y_train, X_test, y_test

    def apply_scaler(self, inputs):
        """Applies standardization to the ECG signals using the fitted scaler."""
        temp = []
        for x in inputs:
            x_shape = x.shape
            temp.append(self.scaler.transform(
                x.flatten()[:, np.newaxis]).reshape(x_shape)
            )
        return np.array(temp)

    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype=np.int8)[y]

    def train_dataloader(self):
        train_dataset = TensorDataset(self.X_train, self.y_train)
        # dataloader = DataLoader(
        #     train_dataset, batch_size=self.bath_size_per_device, shuffle=True)
        # for x, y in dataloader:
        #     print(f'Input shape: {x.shape}, Labels shape: {y.shape}')
        #     break
        return DataLoader(train_dataset, batch_size=self.bath_size_per_device, shuffle=True)

    def val_dataloader(self):
        if self.sub_disease:
            return None
        val_dataset = TensorDataset(self.X_val, self.y_val)
        return DataLoader(val_dataset, batch_size=self.bath_size_per_device, shuffle=True)

    def test_dataloader(self):
        test_dataset = TensorDataset(self.X_test, self.y_test)
        return DataLoader(test_dataset, batch_size=self.bath_size_per_device, shuffle=True)

    def predict_dataloader(self):
        return self.test_dataloader()


if __name__ == '__main__':
    DL = PTBXLDataModule()
    DL.setup()
    loader = DL.train_dataloader()
    x, y = next(iter(loader))
    print(y.shape)
    # dataloader = DL.train_dataloader()
    # for batch_idx, (inputs, labels) in enumerate(dataloader):
    #     print(f"Input shape: {inputs.shape}, Label shape: {labels.shape}")

    #     # Visualize the first 5 signals in the batch
    #     num_signals_to_visualize = 5
    #     plt.figure(figsize=(15, num_signals_to_visualize * 3))

    #     for i in range(num_signals_to_visualize):
    #         plt.subplot(num_signals_to_visualize, 1, i + 1)
    #         # .squeeze() removes unnecessary dimensions
    #         plt.plot(inputs[i].numpy().squeeze())
    #         plt.title(f"ECG Signal {i + 1} - Label: {labels[i].numpy()}")
    #         plt.xlabel("Time")
    #         plt.ylabel("Amplitude")
    #     plt.tight_layout()
    #     plt.show()

    #     break  # Only process the first batch for visualization
