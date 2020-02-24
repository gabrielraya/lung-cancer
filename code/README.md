<<<<<<< HEAD
# Command list


### Docker 

#### Uploading your own container

1. Run interactive mode using :

`./c-submit --require-mem=10G --require-cpus=4 --gpu-count=0   --priority="high" --interactive gabrielrodriguez 8892 4 oni:11500/johnmelle/pathology_base_cuda_10:3  `


2. Install all libraries needed 
3. In the terminal retrieve all docker instance running with `docker ps` and copy your CONTAINER ID: i.e, `2805cc88db30`
4. Shutdown job using `../c-stop job-id`
-     `docker commit [containerid] doduo1.umcn.nl/[user name]/[container name]:[version string]`
-     `docker commit 2805cc88db30 doduo1.umcn.nl/gabrielrodriguez/lung_cancer_nature:1`
-     `docker push oni:11500/[user name]/[container name]:[version string]`
-     `docker push doduo1.umcn.nl/gabrielrodriguez/lung_cancer_nature:1`
-     
 

=======
# Commands


Data is already converted to tif files, and mask have been created.

## symlink:

```shell
ln -s /mnt/netcache/pathology/projects/pathology-weakly-supervised-lung-cancer-growth-pattern-prediction/code/ ./
```

### Convert svs to tif files


### Generate wsi background masks

```shell
./c-submit --require-mem=16G --require-cpus=4 --require-diskspace=10G --require-gpu-mem="8G" --gpu-count=1 --constraint avx gabrielrodriguez 8892 100 doduo1.umcn.nl/witali/background:1 /mnt/netcache/pathology/users/witali/tissue_background_segmentation/background_seg.sh "/mnt/netcache/pathology/users/gabriel/DeepPATH_NIC/data/wsi_tif" "tif" "/mnt/netcache/pathology/users/gabriel/DeepPATH_NIC/data/wsi_masks"
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

### Featurize
```shell
    To run the scrip just change the data_dir name
    
    The following will run bigan encoder with augmentations:
        
            python3 featurize_wsi.py 1 1
            
    The following will run 4task encoder with no augmentations:
        
            python3 featurize_wsi.py 0 0       
```
>>>>>>> master
