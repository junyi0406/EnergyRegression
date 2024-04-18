# Setup the virtual environment in lxplus9
## Source a Environment from LCG System
### find the name of bundle in http://lcginfo.cern.ch/
```
source /cvmfs/sft.cern.ch/lcg/views/LCG_105a_cuda/**x86_64-el9-gcc11-opt**/setup.sh
```
## Setup the Virtual Environment
### install the python virtual env tools
``` 
pip3 install virtualenv --user
```
### add following line to .bashrc and re-log in or source .bashrc
```
export PATH="/afs/cern.ch/user/<first letter of your username>/<username>/.local/bin:$PATH"
```
### create and enable the virtual env
```
virtualenv <env name>
```
### go inside the virtual environment, your shell prompt will begin with "(<env name>)"
```
source <env name>/bin/activate
```
## check cuda toolkit
```
nvidia-smi
```
## modify the requirements.txt and run 
```
pip install -r requirements.txt
```
