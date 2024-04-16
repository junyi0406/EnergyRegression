from pathlib import Path
import torch as th
import pytorch_lightning as pl
import pandas as pd
from sklearn import preprocessing
from torch.utils.data import TensorDataset, DataLoader, random_split, sampler
import json
from package import TreeOperator

class PlDataModule(pl.LightningDataModule):
    def __init__(self, hparams, region, debug):
        super(PlDataModule, self).__init__()
        self.save_hyperparameters(hparams)
        self.region = region
        self.debug = debug
        self.train_dataset = None
        self.valid_dataset = None
        with open(self.hparams.conf_path, "r") as file:
            self.conf = json.load(file)
        
    def setup_dataset(self, df, region, vars, target):
        import time
        import pandas as pd
        dataset = pd.DataFrame()
        table = {}
        df = df[df.sc_isEB == (1 if region == "eb" else 0)]
        for idx, var in enumerate(vars):
            dataset[f"var{idx}"] = eval(var.format(dfname = "df"))
            table[f"var{idx}"] = var.format(dfname = "").replace(".", "")
        dataset["target"] = eval(target.format(dfname = "df"))
        dataset = dataset.sample(frac=0.6, random_state=int(time.time()))
        # dataset.pop("isEB")
        return (table, dataset)
    
    def load_data(self, ):
        input_file = self.hparams.input_location
        # input_file = Path(self.hparams.data_folder).joinpath(self.hparams.train_input).as_posix()
        # df = pd.read_csv(input_file)
        reader = TreeOperator()
        reader.set_names(input_file, treename=self.conf["treeName"])
        df = reader.read_minitree(self.conf, fmt="pd", debug = self.debug)
        var_table, df = self.setup_dataset(df, self.region, self.conf[f"var_{self.region}"], self.hparams.target)
        
        nrow, ncol = df.shape
        df = pd.DataFrame(df.values)
        x = df.loc[:, :ncol-2].values
        y = df.loc[:, ncol-1:].values

        # commented code for normalization options. minmax or mean-std approach
        min_max_scaler = preprocessing.MinMaxScaler()
        scaled_input = min_max_scaler.fit_transform(X=x)
        # normalized_df = (df-df.mean())/df.std()

        
  
        return scaled_input, y

    def setup(self, stage=None):
        x, y = self.load_data()
        dataset = TensorDataset(
            th.tensor(x, dtype=th.float), th.tensor(y, dtype=th.float))
        train_size = int(0.90 * len(dataset))
        val_size = int(len(dataset) - train_size)# int(0.1 * len(dataset))
        self.train_dataset, self.valid_dataset = random_split(dataset, (train_size, val_size))

    def train_dataloader(self):
        dataloader = DataLoader(self.train_dataset,
                                batch_size=self.hparams.batch_size,
                                shuffle=True,
                                num_workers=self.hparams.num_workers,
                                pin_memory=True,
                                drop_last=True)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(self.valid_dataset,
                                batch_size=self.hparams.batch_size,
                                num_workers=self.hparams.num_workers,
                                pin_memory=True,
                                drop_last=True)
        return dataloader