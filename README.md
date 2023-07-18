# ProTSM

This is the official implementation of the paper "Progressive  Pretraining Network for 3D System Matrix Calibration in Magnetic Particle Imaging".

### Important Dependencies:
```
h5py==3.6.0
monai==0.9.0
numpy==1.21.2
pytorch==1.7.0
```
### Data Preprocessing

You should first download the raw .mdf data from the openMPI website: https://magneticparticleimaging.github.io/OpenMPIData.jl/latest/index.html

Please download the Calibration Measurements 6 and 7, and put them in the raw_mdf_data/ folder.

Then you can run the following command to preprocess the data:
```
python preprocess_mdf_data.py
```

### Train

After data preprocessing, you can run the following command to train the model:
```
python main_finetune.py
```

### Predict
After training, you can run the following command to predict the system matrix:
```
python evaluate.py
```

### Reference
If you take advantage of this paper in your research, please cite the following in your manuscript:
"Coming soon"

If you have any problem, please email me with this address: shigen@buaa.edu.cn
