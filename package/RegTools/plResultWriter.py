import json
import torch as th
import pandas as pd
from pathlib import Path
from package import TreeOperator
# All of this can be moved to the test loop of pytorch lightning itself
# Decided to make the inference separate from the training loops


class ResultWriter():
    def __init__(self, trained_model, hparams, region, debug):
        self.model = trained_model
        self.hparams = hparams
        self.region = region
        self.debug = debug
        with open(self.hparams.conf_path, "r") as file:
            self.conf = json.load(file)
    def setup_dataset(self, df, region, vars,):
        import time
        import pandas as pd
        dataset = pd.DataFrame()
        table = {}
        # print(df.shape)
        df = df[df.sc_isEB == (1 if region == "eb" else 0)]
        self.Tar = eval(self.hparams.target.format(dfname = "df")).values
        # with Bar("Creating dataset....", fill='#', suffix='%(percent).1f%% - %(eta)ds') as bar:
        for idx, var in enumerate(vars):
            dataset[f"var{idx}"] = eval(var.format(dfname = "df"))
            table[f"var{idx}"] = var.format(dfname = "").replace(".", "")
           
        # dataset.pop("isEB")
        return (table, dataset)
    
    def load_data(self):
        input_file = Path(self.hparams.data_folder).joinpath(self.hparams.test_input).as_posix()
        reader = TreeOperator()
        reader.set_names(input_file, treename=self.conf["treeName"])   
        df = reader.get_tree(fmt="pd", debug = self.debug)
        var_table, df = self.setup_dataset(df, self.region, self.conf[f"var_{self.region}"])  
        nrow, ncol = df.shape
        self.nFeatures = ncol
        # df = pd.read_csv(input_file)
        x = df.values
        return var_table, x

    def test_model(self):
        var_table, test_data = self.load_data()
        test_data = pd.DataFrame(test_data)
        output = []
        # with Bar("Applying dataset....", fill='#', suffix='%(percent).1f%% - %(eta)ds') as bar:
        tensors = test_data.apply(lambda line: th.tensor(line.values[:self.nFeatures], dtype=th.float), axis=1)
        output = tensors.apply(lambda t: self.model(t.view(-1, self.nFeatures)).item())

                    
        # for idx, line in enumerate(test_data):
        #     x = th.tensor(line[:self.nFeatures], dtype=th.float)
        #     y = self.model(x.view(-1, self.nFeatures))
        #     output.append(y.item())


        self.save_output(test_data, output, var_table.values())

    def save_output(self, input, output, cols):
        output_file = Path(self.hparams.output_folder).joinpath(self.hparams.test_output).as_posix()

        df = pd.DataFrame(input)
        # df.columns = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']
        df.columns = cols
        df['mean'] = output
        df['Tar'] = self.Tar

        # df.to_csv(output_file, index=False)
        import pickle as pk 
        with open(output_file.format(region=self.region), "wb") as file:
            pk.dump(df, file=file)
            