brew install mmseqs2 mafft
brew install iqtree          # optional
conda create -y -n xynapse python=3.11
conda activate xynapse
pip install -r requirements.txt
pip freeze > requirements.lock   # one person commits this
