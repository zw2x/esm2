echo "Initializing submodules..."
git submodule update --init --recursive

echo "Installing openfold"
cd third_party/openfold
pip install -e .
cd ../../
