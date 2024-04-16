export TMPDIR=/home/JYChen/EnergyRegression/tmp

#!/bin/bash

# This is the application script
trainer="torchTry.py"

regres_dir="/home/JYChen/EnergyRegression/"
# Location of root file to apply regression to
input_dir="${regres_dir}samples/DoubleElectron_FlatPT-1to500_13p6TeV"
# root file to apply regression to
train_input="${input_dir}/DoubleElectron_ECALIdealIC_PostEE_124X_30032023.root"
conf_path="${regres_dir}config/Run3Electron_PostEE.json"
# Location of regressions
regres_location="${regres_dir}results/Run3_DoubleElectron/124_ECALIdealIC_PostEE_{region}.ckpt"
# regres_location="${regres_dir}log/checkpoint/final{region}.ckpt"


#set target
target_ideal="{dfname}.mc_energy/{dfname}.sc_rawEnergy"
target_real="{dfname}.mc_energy/(({dfname}.sc_rawEnergy+{dfname}.sc_rawESEnergy)*{dfname}.regIdealMean)"
target_trk="({dfname}.mc_energy * ({dfname}.ele_trkPModeErr*{dfname}.ele_trkPModeErr + ({dfname}.sc_rawEnergy+{dfname}.sc_rawESEnergy)*({dfname}.sc_rawEnergy+{dfname}.sc_rawESEnergy)*{dfname}.regRealSigma*{dfname}.regRealSigma)  / ( ({dfname}.sc_rawEnergy+{dfname}.sc_rawESEnergy)*regIdealMean*{dfname}.ele_trkPModeErr*{dfname}.ele_trkPModeErr + {dfname}.ele_trkPMode*({dfname}.sc_rawEnergy+{dfname}.sc_rawESEnergy)*({dfname}.sc_rawEnergy+{dfname}.sc_rawESEnergy)*{dfname}.regRealSigma*{dfname}.regRealSigma ))"




python $trainer --checkpoint_name $regres_location --input_location $train_input --conf_path $conf_path --target $target_ideal
    
echo ""
# watch -n0.1 nvidia-smi
