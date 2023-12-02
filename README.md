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

Make directories:
```
mkdir raw_mdf_data
mkdir preprocessed_data
```

Please download the Calibration Measurements 6 and 7, and put them in the raw_mdf_data/ folder.

Make sure the following file structure:
```
--raw_mdf_data
----6.mdf
----7.mdf
```

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

```
@ARTICLE{10189221,
  author={Shi, Gen and Yin, Lin and An, Yu and Li, Guanghui and Zhang, Liwen and Bian, Zhongwei and Chen, Ziwei and Zhang, Haoran and Hui, Hui and Tian, Jie},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Progressive Pretraining Network for 3D System Matrix Calibration in Magnetic Particle Imaging}, 
  year={2023},
  volume={42},
  number={12},
  pages={3639-3650},
  doi={10.1109/TMI.2023.3297173}}
```

If you have any problem, please email me with this address: shigen@buaa.edu.cn
