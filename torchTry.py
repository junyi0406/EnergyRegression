import torch as th
import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from argparse import ArgumentParser
import sys
sys.path.append("..")
from package import PlDataModule, PlDNNModule, ResultWriter


early_stop_callback = EarlyStopping(monitor='train_loss', min_delta=0.0001, patience=5, verbose=True, mode='min')

def train_model(hparams, region:str, debug= True):
    
    seed_everything(77)

    ml_module = PlDNNModule(hparams=hparams, region=region)
    data_module = PlDataModule(hparams=hparams, region=region, debug=debug)

    #model_trainer = pl.Trainer.from_argparse_args(hparams)
    # model_trainer = pl.Trainer.add_argparse_args(hparams, callbacks=[early_stop_callback])
    model_trainer = pl.Trainer(callbacks=[early_stop_callback])
    
    model_trainer.logger = pl.loggers.TensorBoardLogger('logs/', name='exp')

    model_trainer.fit(ml_module, data_module)
    model_trainer.save_checkpoint(hparams.checkpoint_name.format(region=region))


def test_model(hparams, region, debug= True):
    model = PlDNNModule.load_from_checkpoint(checkpoint_path=hparams.checkpoint_name.format(region=region), hparams=hparams, region=region)
    output_writer = ResultWriter(model, hparams=hparams, region=region, debug=debug)
    output_writer.test_model()

def main():
    parser = ArgumentParser()
    print("start")
    # program arguments
    parser.add_argument('--checkpoint_name', type=str, default='/home/JYChen/EnergyRegression/log/checkpoint/{region}_final.ckpt')
    parser.add_argument('--input_location', type=str, default='/home/JYChen/EnergyRegression/samples/DoubleElectron_FlatPT-1to500_13p6TeV')
    parser.add_argument('--conf_path', type=str, default='/home/JYChen/EnergyRegression/config/Run3Electron_PostEE.json')
    # parser.add_argument('--train_input', type=str, default='DoubleElectron_ECALIdealIC_PostEE_124X_30032023.root')
    # parser.add_argument('--output_folder', type=str, default='/home/JYChen/EnergyRegression/samples/results/')
    # parser.add_argument('--test_input', type=str, default='DoubleElectron_ECALIdealIC_PostEE_124X_30032023.root')
    # parser.add_argument('--test_output', type=str, default='DoubleElectron_ECALIdealIC_PostEE_124X_{region}_applied.pk')
    parser.add_argument('--useDask', type=bool, default=False)
    parser.add_argument('--fmt', type=str, default='ak')
    

    # trainer arguments
    parser.add_argument('--target', type=str, default='{dfname}.mc_energy/{dfname}.sc_rawEnergy')
    parser.add_argument('--default_root_dir', type=str, default='./logs')
    parser.add_argument('--max_epochs', type=int, default=450)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=50)
    parser.add_argument('--gpus', type=int, default=(1 if th.cuda.is_available() else 0))
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--accelerator', type=str, default="auto")
    parser.add_argument('--auto_select_gpus', type=bool, default=True)
    parser.add_argument('--deterministic', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=0.02)

    # debug parameters - enable fast_dev_run for quick sanity check of the training loop
    parser.add_argument('--fast_dev_run', type=bool, default=False)

    args = parser.parse_args()
    print("starting training eb.............")
    train_model(hparams=args, region= "eb", debug= True)
    
    print("starting training ee.............")
    train_model(hparams=args, region= "ee", debug= True)


if __name__ == "__main__":
    main()
    
