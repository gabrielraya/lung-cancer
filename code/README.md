# Commands


Data is already converted to tif files, and mask have been created.

## symlink:

```shell
ln -s /mnt/netcache/pathology/projects/pathology-weakly-supervised-lung-cancer-growth-pattern-prediction/code/ ./
```


### Vectorize images

Default settings: vectorize images at magnification level 1 with patch size of 128 pixels.

* LUAD
```shell
python3 /mnt/netcache/pathology/projects/pathology-weakly-supervised-lung-cancer-growth-pattern-prediction/code/00_preprocessing/vectorize_wsi.py --wsiFolder="/mnt/netcache/pathology/archives/lung/TCGA_LUAD/wsi_diagnostic_tif" --maskFolder="/mnt/netcache/pathology/archives/lung/TCGA_LUAD/tissue_masks_diagnostic" --outputDir="/mnt/netcache/pathology/projects/pathology-weakly-supervised-lung-cancer-growth-pattern-prediction/results/tcga_luad/vectorized"
```
    
* LUSC
```shell
python3 /mnt/netcache/pathology/projects/pathology-weakly-supervised-lung-cancer-growth-pattern-prediction/code/00_preprocessing/vectorize_wsi.py --wsiFolder="/mnt/netcache/pathology/archives/lung/TCGA_LUSC/wsi_diagnostic_tif" --maskFolder="/mnt/netcache/pathology/archives/lung/TCGA_LUSC/tissue_masks_diagnostic" --outputDir="/mnt/netcache/pathology/projects/pathology-weakly-supervised-lung-cancer-growth-pattern-prediction/results/tcga_lusc/vectorized"
```