# Uncertainty Aware Sampling Framework (UASF)

Sample informative tiles from WSIs taken from TCGA with UASF. The model takes as inputs weak-labeled tiles, where tiles are given their WSI label. The model adopted Monte-Carlo dropout that are used for variational inference to generate not only a prediction probability but also an uncertainty measure.  

## Installation
Create a virtual environment and install the required packages

### Environment setup

#### Environment from file
```bash
conda env create --file env.yml
source activate uacnn
```
#### Environment from scratch
```bash
conda create -n uacnn python=3.7 openslide pytorch torchvision torchaudio cudatoolkit=11.4 -c pytorch -c conda-forge
conda activate uacnn
```
#### Required Packages
```bash
#HistomicsTK
pip install histomicstk --find-links https://girder.github.io/large_image_wheels
#Additional packages 
pip install tqdm rasterio large-image-source-openslide tensorboard pytorch-lightning
```
Note: code was run with python 3.7

## Data Collection and preprocessing
To insure reproducibility of the results, data splits and tiles coordinates are provided in MICCAI_data.gz
To uncompress it, run
```bash
tar -xzvf MICCAI_data.tar.gz
```
### Download TCGA Slides
We originally downloaded the whole-slide images from the TCGA data portal https://portal.gdc.cancer.gov/ via the gdc-client tool.
Go to dedicated folder to store files (all FFPE slides from TCGA is approx. 1.0 TB). Download images using the corresponding manifest file:
```bash
mkdir data
cd data
mkdir slides
cd slides
gdc-client download -m gdc_manifest.TCGA_SARC_dx_slides_2022-05-28
```

### Preprocessing WSIs
#### Tiling WSIs
The code in preprocessing folder is designed to extract non-overlapping tiles directly from whole-slide images. We extracted 256x256 pixel tiles at x10 magnification level. To extract tiles run:
```bash
python extract_tiles.py --tile [path/to/slides/folder] --wsi [file_name.svs] -wd [tile_width] -ht [tile_height] -m[magnification] -o [path/to/output/]
#Example
python preprocessing/extract_tiles.py --tile data/slides/ --wsi TCGA-X9-A973-01Z-00-DX5.D6A52779-A0A4-4119-9AB4-1A7A6BD98337.svs -m 10 -wd 256 -ht 256 -o data/tiles/
```

## UA-CNN Experiment
### Training UA-CNN
To run an experiment, write first an app file or use the example available in folder app/LMS/. You can modify tiling and training parameters in app/LMS/config file.
Launch training experiment with a train-validation split:
```bash
cd app/LMS/
python lms_tcga_sarc.py train --img_dir data/tiles/ \
 --train_data_desc MICCAI_data/LMS_tiles_clinical_data_fold1_train.csv \
  --val_data_desc MICCAI_data/LMS_tiles_clinical_data_fold1_val.csv \
   --weights=histossl \
   --task=FNCLCC_grade
```
#### Pretrained weights
In our experiments, we compared the performance of UA-CNN preteined on Imagnet and  UA-CNN preteined on Histossl.
Histossl is a pretrained model for self supervised histopathology. Reference: [Self supervised contrastive learning for digital histopathology](https://arxiv.org/pdf/2011.13971.pdf). You can download it from here: 
```link
https://github.com/ozanciga/self-supervised-histopathology
```
### Evaluation Uncertainty for trained UA-CNN
```bash
python lms_tcga_sarc.py eval_uncertainty --img_dir data/tiles/  \
  --train_data_desc MICCAI_data/LMS_tiles_clinical_data_fold1_train.csv \
    --val_data_desc MICCAI_data/LMS_tiles_clinical_data_fold1_val.csv \
      --weights=LMS_SCE_uacnn.pth  \
       --task=FNCLCC_grade 
```
### Generate whole-slide image uncertainty map
```bash
python lms_tcga_sarc.py wsi_uncertainty_map --img_dir data/slides/TCGA-X9-A973-01Z-00-DX5.D6A52779-A0A4-4119-9AB4-1A7A6BD98337.svs \
--weights=LMS_SCE_uacnn.pth --tile_size=256 --mag=10 --task=FNCLCC_grade --nclasses=3 --class2index="0,1,2" 
```
