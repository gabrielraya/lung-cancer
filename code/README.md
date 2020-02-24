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
 

