# RCPMOD
This repo contains the code of ACMMM 2024 "Regularized Contrastive Partial Multi-view Outlier Detection"
## Configuration

The hyper-parameters, the training options (including **the missing rate** and **the outlier rate**) are defined in configure.py.

## Datasets

The BDGP, LandUse-21, Scene-15 and Handwritten datasets are in this repository, but Fashion is to large to upload. Download link of Fashion dataset : https://pan.baidu.com/s/1OgCGR6zWGk6KCr2kWpbKCw?pwd=ve5b code: ve5b 

## Requirements

pytorch==2.0.1 

scikit-learn==1.3.1

numpy==1.19.5

## Usage
To run the code on a specific dataset, use the following command:
```bash
python run.py --dataset 1 --print_num 5 --test_time 5
```
The number following --dataset ranges from 1 to 5, representing Scene-15, LandUse_21, BDGP, Handwritten, and Fashion, respectively.

Alternatively, you can run the code on all datasets with a missing rate of 0.3 by using the following commands:

```bash
chmod +x run_all_datasets.sh
./run_all_datasets.sh
```