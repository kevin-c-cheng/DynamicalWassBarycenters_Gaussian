# Dynamical Wasserstein Barycenters for Time Series Modeling

This is the code related for the [Dynamical Wasserstein Barycenter](http://) model published in Neurips 2021.

To run the code and replicate the results reported in our paper, 

```python
# usage: DynamicalWassersteinBarycenters.py dataSet dataFile debugFolder interpModel [--ParamTest PARAMTEST] [--lambda LAM] [--s S]

# Sample run on MSR data                                         
>> python DynamicalWassersteinBarycenters.py MSR_Batch ../Data/MSR_Data/subj090_1.mat ../debug/MSR/subj001_1.mat Wass 

# Sample run for parameter test
>> python DynamicalWassersteinBarycenters.py MSR_Batch ../Data/MSR_Data/subj090_1.mat ../debug/ParamTest/subj001_1.mat Wass --ParamTest 1 --lambda 100 --s 1.0

```

The ``interpMethod`` is either ``Wass` for the Wasserstein barycentric model or ``GMM`` for the linear interpolation model.

## Simulated Data

The simulated data and experiment included in this supplement can be replicated using using the following commands.
```python
# Generate 2 and 3 state simulated data                                         
>> python GenerateOptimizationExperimentData.py
>> python GenerateOptimizationExperimentData_3K.py

# usage: OptimizationExperiment.py FileIn Mode File
# Sample run for optimization experiment
>> python OptimizationExperiment.py ../data/SimulatedOptimizationData_2K/dim_5_5.mat/ WB ../debug/SimulatedData/dim_5_5_out.mat 

```

The ``Mode`` is either ``WB`` for Wasserstein-Bures geometry and ``Euc`` for Euclidean geometry using Cholesky decomposition parameterization.

## Requirements
```
_libgcc_mutex=0.1=conda_forge
_openmp_mutex=4.5=1_llvm
_pytorch_select=0.2=gpu_0
blas=2.17=openblas
ca-certificates=2020.12.5=ha878542_0
certifi=2020.12.5=py38h578d9bd_1
cffi=1.14.4=py38h261ae71_0
cudatoolkit=8.0=3
cudnn=7.1.3=cuda8.0_0
cycler=0.10.0=py_2
freetype=2.10.4=h7ca028e_0
future=0.18.2=py38h578d9bd_3
immutables=0.15=py38h497a2fe_0
intel-openmp=2020.2=254
joblib=1.0.0=pyhd8ed1ab_0
jpeg=9d=h36c2ea0_0
kiwisolver=1.3.1=py38h82cb98a_0
lcms2=2.11=hcbb858e_1
ld_impl_linux-64=2.33.1=h53a641e_7
libblas=3.8.0=17_openblas
libcblas=3.8.0=17_openblas
libedit=3.1.20191231=h14c3975_1
libffi=3.3=he6710b0_2
libgcc-ng=9.3.0=h5dbcf3e_17
libgfortran-ng=7.3.0=hdf63c60_0
libgomp=9.3.0=h5dbcf3e_17
liblapack=3.8.0=17_openblas
liblapacke=3.8.0=17_openblas
libopenblas=0.3.10=pthreads_hb3c22a3_4
libpng=1.6.37=h21135ba_2
libstdcxx-ng=9.3.0=h6de172a_18
libtiff=4.1.0=h4f3a223_6
libwebp-base=1.1.0=h36c2ea0_3
llvm-openmp=11.0.0=hfc4b9b4_1
lz4-c=1.9.2=he1b5a44_3
matplotlib-base=3.3.3=py38h5c7f4ab_0
mkl=2020.4=h726a3e6_304
mkl-service=2.3.0=py38he904b0f_0
mkl_fft=1.3.0=py38h5c078b8_1
mkl_random=1.2.0=py38hc5bc63f_1
ncurses=6.2=he6710b0_1
ninja=1.10.2=py38hff7bd54_0
numpy=1.19.5=py38h18fd61f_1
numpy-base=1.18.5=py38h2f8d375_0
olefile=0.46=pyh9f0ad1d_1
openssl=1.1.1k=h7f98852_0
pillow=8.1.0=py38h357d4e7_1
pip=20.3.3=py38h06a4308_0
pot=0.7.0=py38h950e882_0
pycparser=2.20=py_2
pyparsing=2.4.7=pyh9f0ad1d_0
python=3.8.5=h7579374_1
python-dateutil=2.8.1=py_0
python_abi=3.8=1_cp38
pytorch=1.7.1=cpu_py38h36eccb8_1
readline=8.0=h7b6447c_0
scikit-learn=0.24.1=py38h658cfdd_0
scipy=1.5.2=py38h8c5af15_0
setuptools=51.1.2=py38h06a4308_4
six=1.15.0=py38h06a4308_0
sqlite=3.33.0=h62c20be_0
threadpoolctl=2.1.0=pyh5ca1d4c_0
tk=8.6.10=hbc83047_0
tornado=6.1=py38h497a2fe_1
wheel=0.36.2=pyhd3eb1b0_0
xz=5.2.5=h7b6447c_0
zlib=1.2.11=h7b6447c_3
zstd=1.4.5=h6597ccf_2
```
