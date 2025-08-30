# EvoFill: Evolutionary Trajectory-Informed Genotype Imputation

**内部开发中**

Pingcode：https://pchz20250707025859383.pingcode.com/ship/products/JYX

GitHub 仓库不包含`./data`，`./ckpt` 等数据文件夹

完整本地项目请见：`192.168.10.5:/mnt/qmtang/EvoFill/`

## Project Structure

```
EvoFill/
├── README.md                    # Project documentation
├── config/                      # Configuration files
├── data/                        # Data files (excluded from git if >1MB)
├── plots/                       # Analysis plots
├── simulation/                  # stdpopsim simulation scripts
│   ├── run_all_scenarios.py     # Main simulation controller
│   ├── scenario1_east_asian_modern.py
│   ├── scenario2_three_populations.py
│   └── scenario3_ancient_modern.py
├── src/                         # Source code
├── notebook/                    # Jupyter notebooks
├── scripts/                     # Utility scripts
└── tests/                       # Test files
```

## Installation and configuration

```bash
conda create -f environment.yml
conda activate evofill
```

## Dataset preparation

1. split training and validation set
```bash
bash scripts/split_vcf.sh
```

2. calculate the p-distance among samples:
```bash
Vcf2Dis -InPut training_vcf.gz -OutPut training.p_dis.mat
Vcf2Dis -InPut val_vcf.gz -OutPut val.p_dis.mat
```

3. Convert vcf to tensor
```bash
python src/vcf2tensor.py
```

## Simulation with stdpopsim

Run population genetic simulations for training data:

```bash
cd simulation/
conda activate Imputation2
python run_all_scenarios.py
```

This will generate three simulation scenarios:
1. East Asian modern population (1000 individuals)
2. Three modern populations (East Asian, African, European)
3. Modern populations + ancient DNA samples

## Training

1. config the model and training process

2a. start training locally
```bash
deepspeed train.py --deepspeed config/ds_config.json
```

2b. start training in cluster
```bash
deepspeed train.py --deepspeed config/ds_config.json
```

3. monitor the training
```bash
[todo]
```

## Imputation

```bash
python imputation.py
```

## Results visualization

```bash
jupyter notebook ...
```
