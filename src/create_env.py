# Create a new conda environment named xxx with Python 3.9
conda create -n research python=3.9

# Activate the environment
conda activate research

# Install scanpy, seaborn, gseapy, numpy, and pandas using conda
conda install -c conda-forge scanpy seaborn gseapy numpy pandas

# Install sanbomics using pip
pip install sanbomics

# Install pydeseq2 using pip
pip install pydeseq2

