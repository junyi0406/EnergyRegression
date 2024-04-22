import uproot
import pandas as pd
import awkward as ak
from concurrent.futures import ThreadPoolExecutor

class TreeOperator:
    
    def __init__(self) -> None:
        pass
    
    def set_names(self, inpath, treename):
        self.file_path = inpath
        self.treename = treename
        
    def read_to(self, path, treename, fmt, conf, axis=1, debug=False):

        tree = uproot.open(path)[treename]
        if debug:
            df = tree.arrays(conf["branches"], library=fmt, entry_stop=10000)
        else:
            df = tree.arrays(conf["branches"], library=fmt,)
        if self.hparams.fmt == "pd":
            dfs = []
            for branch in conf["branches"]:
                obj = eval(f"ak.to_dataframe(df.{branch})")
                obj = obj.rename(columns={oldname: f"{branch}_{oldname}" for oldname in obj.columns})
                dfs.append(obj)
            return pd.concat(dfs, axis=axis)
        else:
            return df
    

    def read_minitree(self, conf, fmt="", debug=False, useDask = False):
        import dask
        import dask.dataframe as dd
        from dask.diagnostics import ProgressBar
        # if fmt == "pd":
        #     return self.read_to(ttree = tree, debug=debug )
        
        if useDask:
            if self.hparams.fmt == "ak":
                raise TypeError("dask didn't support awkward type array")
            if isinstance(self.file_path, list):
                df_merged = []
                for i, filename in enumerate(self.file_path):
                    print("loading: ", filename)
                    df_merged.append(dask.delayed(self.read_to)(filename, self.treename, fmt, conf, debug))
                df_merged = dask.delayed(pd.concat)(df_merged)
            elif isinstance(self.file_path, str):
                print("loading: ", self.file_path)
                df_merged = dask.delayed(self.read_to)(self.file_path, self.treename, fmt, conf, debug)
            else:
                print("unsupport data path type")
                exit()

            print("start computing")
            df_merged = dd.from_delayed(df_merged)
            
            with dask.config.set(pool=ThreadPoolExecutor(10)):
                with ProgressBar():
                    result = df_merged.compute()
                    # result.reset_index(inplace = True, drop = True)
        else:
            result = self.read_to(self.file_path, self.treename, fmt, conf, debug)
        print("dataframe are all set")
        return result
        
