pip install -q biopython biotite freesasa pydssp lightgbm scikit-learn pandas pyarrow \
  streamlit plotly py3Dmol transformers requests pyyaml tqdm joblib matplotlib pytest
apt-get -qq update && apt-get -qq install -y mafft iqtree
wget -q https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz -O /tmp/mmseqs.tar.gz
tar xzf /tmp/mmseqs.tar.gz -C /content && ln -sf /content/mmseqs/bin/mmseqs /usr/local/bin/mmseqs
mmseqs version && mafft --version
pip freeze > requirements.lock
