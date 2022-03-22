set -e
conda create -n capsules_torch12_py37 --clone torch12
conda activate  capsules_torch12_py37

conda install -c anaconda cupy   -y
pip install pynvrtc joblib

# lie_learn deps
conda install -c anaconda cython   -y
conda install -c anaconda requests   -y

# shrec17 example dep
conda install -c anaconda scipy   -y
conda install -c conda-forge rtree shapely   -y
conda install -c conda-forge pyembree   -y
pip install "trimesh[easy]"
pip install pysnooper

mkdir -p 3rd
cd 3rd
git clone https://github.com/jonas-koehler/s2cnn
cd s2cnn
python setup.py install
pip install trimesh
pip install scikit-learn
pip install lie_learn
pip install pynvrtc
