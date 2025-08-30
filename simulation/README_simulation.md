# stdpopsim 三个情景模拟

本模块使用 stdpopsim 软件模拟三个不同的人类基因组情景，重点关注22号染色体的数据。

## 目录结构

```
simulation/
├── README_simulation.md         # 本说明文档
├── run_all_scenarios.py         # 主控脚本
├── scenario1_east_asian_modern.py    # 情景1脚本
├── scenario2_three_populations.py    # 情景2脚本
└── scenario3_ancient_modern.py       # 情景3脚本

输出文件将保存到：
../data/                         # 数据文件（.trees, .vcf, .csv）
../plots/                        # 分析图表（.png）
```

## 环境要求

- conda 环境: Imputation2
- Python 3.7+
- stdpopsim 0.3.0+

## 安装依赖

在 Imputation2 环境中运行：

```bash
conda activate Imputation2
pip install stdpopsim matplotlib pandas scikit-learn seaborn scipy
```

## 三个模拟情景

### 情景1: 东亚现代人群
- **文件**: `scenario1_east_asian_modern.py`
- **描述**: 模拟1000个东亚现代人群基因组的22号染色体
- **模型**: OutOfAfrica_3G09 (CHB人群)
- **样本数**: 1000个个体

### 情景2: 三个现代人群
- **文件**: `scenario2_three_populations.py`  
- **描述**: 东亚、非洲、欧洲三个人群各1000个基因组的22号染色体
- **模型**: OutOfAfrica_3G09
- **样本数**: 
  - YRI (非洲): 1000个个体
  - CEU (欧洲): 1000个个体  
  - CHB (东亚): 1000个个体

### 情景3: 现代人群 + 古DNA
- **文件**: `scenario3_ancient_modern.py`
- **描述**: 三个现代人群各1000个基因组，加上模拟的古DNA样本
- **模型**: AncientEurasia_9K19 (如果可用) 或 OutOfAfrica_3G09
- **样本数**:
  - 现代人群: 每个人群1000个个体
  - 古代样本: 模拟不同时期的古DNA样本

## 使用方法

### 方法1: 运行所有情景 (推荐)

```bash
cd simulation/
conda activate Imputation2
python run_all_scenarios.py
```

这将自动：
1. 检查并安装必要的依赖包
2. 依次运行三个情景
3. 生成分析图表
4. 创建总结报告

### 方法2: 单独运行情景

```bash
cd simulation/
conda activate Imputation2

# 运行情景1
python scenario1_east_asian_modern.py

# 运行情景2  
python scenario2_three_populations.py

# 运行情景3
python scenario3_ancient_modern.py
```

## 输出文件

每个情景会生成以下类型的文件：

### 数据文件 (保存在 ../data/ 目录)
- `scenario*_chr22.trees`: 树序列文件 (tskit格式)
- `scenario*_chr22.vcf`: VCF格式的基因型数据
- `scenario*_*.csv`: 统计分析结果

### 图表文件 (保存在 ../plots/ 目录)
- `scenario1_analysis.png`: 情景1分析图表
- `scenario2_population_analysis.png`: 情景2人群结构分析
- `scenario3_temporal_analysis.png`: 情景3时间序列分析

### 报告文件 (保存在项目根目录)
- `simulation_summary_report.txt`: 总结报告

## 分析内容

### 情景1分析
- 核苷酸多样性 (π)
- Tajima's D 统计量
- 站点频谱 (SFS)
- 树高度分布
- 突变密度分布
- 样本间遗传距离

### 情景2分析  
- 主成分分析 (PCA)
- 人群遗传多样性比较
- 人群间 Fst 距离
- 等位基因频率分布
- 多维标度分析 (MDS)

### 情景3分析
- 时间序列 PCA 分析
- 遗传多样性随时间变化
- 人群间距离随时间变化
- 不同时期的等位基因频率谱
- 古代和现代样本比较

## 注意事项

1. **计算时间**: 每个情景可能需要几分钟到几十分钟的运行时间
2. **内存使用**: 建议至少4GB可用内存
3. **磁盘空间**: 每个情景大约需要100-500MB磁盘空间
4. **染色体长度**: 为了加快模拟速度，使用了22号染色体10%的长度

## 故障排除

### 常见问题

1. **ImportError**: 确保在正确的conda环境中运行
   ```bash
   conda activate Imputation2
   pip install [缺失的包名]
   ```

2. **内存不足**: 减少样本数量或染色体长度
   ```python
   # 在脚本中修改
   contig = species.get_contig("chr22", length_multiplier=0.05)  # 使用5%长度
   samples = {"CHB": 500}  # 减少样本数
   ```

3. **模型不可用**: 脚本会自动回退到更简单的模型

### 获取帮助

查看 stdpopsim 可用模型：
```bash
conda activate Imputation2
stdpopsim HomSap --help-models
```

查看可用染色体：
```bash
stdpopsim HomSap --help-genetic-maps
```

## 参考文献

- Adrion, J.R., et al. (2020). A community-maintained standard library of population genetic models. eLife, 9, e54967.
- stdpopsim 文档: https://stdpopsim.readthedocs.io/

## 版本信息

- stdpopsim: 0.3.0
- Python: 3.7+
- 创建日期: 2025-08-30
