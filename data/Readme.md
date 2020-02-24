# Tcga data

For this project, we are interested only in the diagnostic slides; overall around 1000.

### Location of the data

- /mnt/netcache/pathology/datasets/Lung/TCGA_LUAD
- /mnt/netcache/pathology/datasets/Lung/TCGA_LUSC

the tissue background masks are already created, e.g.

- /mnt/netcache/pathology/datasets/Lung/TCGA_LUAD/tissue_masks_diagnostic
- /mnt/netcache/pathology/datasets/Lung/TCGA_LUSC/tissue_masks_diagnostic​

H:\projects\pathology-lung-tumor-segmentation\Slides\Training

### Nueral Compression Method

The neural image compression method has the following steps:

0. Train Encoder (for later)
For the BIGAN encoder: extract\_patches, train\_bigan\_encoder
There are already trained encoder which you can start to use; you can find them in the github repo
There is also a new supervised encoder:
/mnt/netcache/pathology/projects/pathology-proacting/neoadjuvant\_nki/nic/encoder\_zoo/supervsied\_enc\_2019\_4tasks.h5

1. Vectorize: We work not directly on the slide tifs, but on a more compact numpy representation
   (currently patches 128x128 from spacing 05)
2. Featurize: Apply the encoder to create the compressed representation for each slide
3. Train wsi classifier 
for this you will need a yaml-config for train/valid/test, here is an example:
/mnt/netcache/pathology/projects/pathology-proacting/neoadjuvant\_nki/configs/clf/data\_2cl\_03\_03\_03\_split1.yaml
4. Test wsi classifier

