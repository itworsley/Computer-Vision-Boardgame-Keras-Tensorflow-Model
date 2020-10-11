# Introduction
`project.py` contains the various OpenCV algorithms to handle the live video feed and uses the
`train.py` file to predict the tokens using the `model.h5` training model.

When training the model, `bottleneck_features_train.npy`, `bottleneck_features_validation.npy` and `class_indices.npy` are all generated and required to be able to run the prediction algorithm.

To be able to train the model, the directory should be constructed as the following:
```
.
├── train_data
    ├── test
    ├── train
    └── sample
├── project.py                   
├── train.py
└── README.md
```

You can then generate your own version of the `model.h5` file by running the `train.py` script. The application is then able to run `project.py`.
# Miniconda Environment Breakdown
Create your environment in Miniconda using the *Anaconda Prompt* and typing `conda create --name YOUR_ENV_NAME`.  
Then activate the environment using `conda activate YOUR_ENV_NAME`. You should then be able to install any of the packages in the [Python Packages](#python-packages) section.
## Global (PC) Packages
- [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
- [Cuda 11.0.3](https://developer.nvidia.com/cuda-downloads)  
After download, make sure `CUDA_PATH` is set as an environment variable to install folder.   
For Windows, mine was: *C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0*.  
Also make sure you have the following in your path: *C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\bin* & *C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\libnvvp*
- [Cudnn 11.0](https://developer.nvidia.com/cudnn) (NVIDIA Developer Program Membership Required)  
  After downloading, make sure you have the following in your PATH:  
  *C:\Program Files\NVIDIA Corporation\Nsight Compute 2020.1.2\*
  *C:\Program Files\NVIDIA Corporation\NVIDIA NvDLISR*

## Python Packages
- OpenCV - `pip install --user opencv-python`
- PyTesseract - `conda install -c conda-forge pytesseract`
- OpenCV - `pip install opencv-python`
- Keras GPU Version (can also use CPU version if you wish) - `conda install -c anaconda keras-gpu`  
  Keras also installs TensorFlow 1.14.0.  
  After install make sure the following (or similar) is in your PATH:  
  *C:\Users\Isaac\miniconda3\envs\KerasPython\Scripts*  
  *C:\Users\Isaac\miniconda3\envs\KerasPython\Library*  
  *C:\Users\Isaac\miniconda3\envs\KerasPython\Library\bin\*  
  *C:\Users\Isaac\miniconda3\envs\KerasPython\mingw-w64\bin*
- Pillow - `conda install -c anaconda pillow`
- MatPlotLib - `conda install -c conda-forge matplotlib`

## Tesseract Set Up
Initially, I used a combination of PyTesseract and Tesseract to determine the text on the given tokens.
This had several flaws, as it required a complex training system that was difficult to set up.
Tools to help me set up the training were:
- [QTBox Editor 1.12rc1c](https://github.com/zdenop/qt-box-editor/releases/download/v1.12rc1/qt-box-editor-1.12rc1c-portable.zip)
- [JTessBoxEditor 2.2.0](https://github.com/nguyenq/jTessBoxEditor/releases/download/Release-2.2.0/jTessBoxEditor-2.2.0.zip)

JTessBoxEditor also includes a version of Tesseract (4.0.0.20181030), so feel free to use that.  
Make sure your Tesseract root folder is set in your PC's environment variables.

The most similar font I found to the *Settlers of Catan* tokens was [EB Garamond](https://fonts.google.com/specimen/EB+Garamond).

## Package List
Retrieve using `conda list` in *Anaconda Prompt*. Make sure you install the packages above before attempting to download any
of the following as many of the packages are included as dependencies of others.

```
# Name                    Version                   Build  Channel
_tflow_select             2.1.0                       gpu    anaconda
absl-py                   0.9.0                    py36_0    anaconda
astor                     0.8.1                    py36_0    anaconda
blas                      1.0                         mkl    anaconda
ca-certificates           2020.6.20            hecda079_0    conda-forge
certifi                   2020.6.20        py36h9f0ad1d_0    conda-forge
cudatoolkit               10.0.130                      0    anaconda
cudnn                     7.6.5                cuda10.0_0    anaconda
cycler                    0.10.0                     py_2    conda-forge
freetype                  2.10.2               hd328e21_0
gast                      0.4.0                      py_0    anaconda
grpcio                    1.12.1           py36h1a1b453_0    anaconda
h5py                      2.7.1            py36he54a1c3_0    anaconda
hdf5                      1.10.1           vc14hb361328_0  [vc14]  anaconda
icc_rt                    2019.0.0             h0cc432a_1    anaconda
icu                       58.2                 ha925a31_3
importlib-metadata        1.7.0                    py36_0    anaconda
intel-openmp              2020.2                      254    anaconda
jpeg                      9b               vc14h4d7706e_1  [vc14]  anaconda
keras-applications        1.0.8                      py_1    anaconda
keras-base                2.3.1                    py36_0    anaconda
keras-gpu                 2.3.1                         0    anaconda
keras-preprocessing       1.1.0                      py_1    anaconda
kiwisolver                1.2.0            py36h246c5b5_0    conda-forge
libpng                    1.6.37               h2a8f88b_0
libprotobuf               3.13.0               h200bbdf_0    anaconda
libtiff                   4.1.0                h56a325e_1
lz4-c                     1.9.2                h62dcd97_1
markdown                  3.2.2                    py36_0    anaconda
matplotlib                3.3.1                         1    conda-forge
matplotlib-base           3.3.1            py36h856a30b_1    conda-forge
mkl                       2019.4                      245    anaconda
mkl-service               2.3.0            py36hb782905_0    anaconda
mkl_fft                   1.1.0            py36h45dec08_0    anaconda
mkl_random                1.1.0            py36h675688f_0    anaconda
numpy                     1.19.1           py36h5510c5b_0    anaconda
numpy-base                1.19.1           py36ha3acd2a_0    anaconda
olefile                   0.46                     py36_0
openssl                   1.1.1g               he774522_1    conda-forge
pillow                    7.2.0            py36hcc1f983_0    anaconda
pip                       20.2.2                   py36_0
protobuf                  3.13.0           py36h6538335_0    anaconda
pyparsing                 2.4.7              pyh9f0ad1d_0    conda-forge
pyqt                      5.9.2            py36h6538335_4    conda-forge
python                    3.6.9                h5500b2f_0
python-dateutil           2.8.1                      py_0    conda-forge
python_abi                3.6                     1_cp36m    conda-forge
pyyaml                    5.3.1            py36he774522_0    anaconda
qt                        5.9.7            vc14h73c81de_0
scipy                     1.5.2            py36h9439919_0    anaconda
setuptools                49.6.0                   py36_0
sip                       4.19.8           py36h6538335_0
six                       1.15.0                     py_0    anaconda
sqlite                    3.33.0               h2a8f88b_0
tensorboard               1.14.0           py36he3c9ec2_0    anaconda
tensorflow                1.14.0          gpu_py36h305fd99_0    anaconda
tensorflow-base           1.14.0          gpu_py36h55fc52a_0    anaconda
tensorflow-estimator      1.14.0                     py_0    anaconda
tensorflow-gpu            1.14.0               h0d30ee6_0    anaconda
termcolor                 1.1.0                    py36_1    anaconda
tk                        8.6.10               he774522_0
tornado                   6.0.4            py36hfa6e2cd_0    conda-forge
vc                        14.1                 h0510ff6_4
vs2015_runtime            14.16.27012          hf0eaf9b_3
werkzeug                  1.0.1                      py_0    anaconda
wheel                     0.35.1                     py_0
wincertstore              0.2              py36h7fe50ca_0
wrapt                     1.12.1           py36he774522_1    anaconda
xz                        5.2.5                h62dcd97_0
yaml                      0.1.7            vc14h4cb57cf_1  [vc14]  anaconda
zipp                      3.1.0                      py_0    anaconda
zlib                      1.2.11               h62dcd97_4
zstd                      1.4.5                h04227a9_0
```

## Final Remarks
If this set up doesn't work for you, sorry, good luck!
