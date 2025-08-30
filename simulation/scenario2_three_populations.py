#!/usr/bin/env python3
"""
情景2: 东亚、非洲、欧洲三个人群各1000个基因组的22号染色体模拟
使用stdpopsim模拟三个现代人群的22号染色体数据
"""

import stdpopsim
import tskit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

def simulate_three_populations_chr22():
    """
    模拟东亚、非洲、欧洲三个人群各1000个个体的22号染色体
    """
    print("开始模拟情景2: 东亚、非洲、欧洲三个人群各1000个基因组的22号染色体")
    
    # 设置随机种子
    np.random.seed(42)
    
    # 获取人类物种和22号染色体
    species = stdpopsim.get_species("HomSap")
    contig = species.get_contig("chr22", length_multiplier=0.1)  # 使用10%的长度
    
    # 使用OutOfAfrica_3G09模型，包含三个人群
    model = species.get_demographic_model("OutOfAfrica_3G09")
    
    # 设置样本 - 每个人群1000个个体
    samples = {
        "YRI": 1000,  # 非洲人群 (约鲁巴人)
        "CEU": 1000,  # 欧洲人群 (欧洲血统)
        "CHB": 1000   # 东亚人群 (汉族)
    }
    
    # 运行模拟
    print("正在运行模拟...")
    engine = stdpopsim.get_engine("msprime")
    ts = engine.simulate(
        demographic_model=model,
        contig=contig,
        samples=samples,
        seed=42
    )
    
    print(f"模拟完成!")
    print(f"树序列包含 {ts.num_samples} 个样本")
    print(f"树序列包含 {ts.num_trees} 棵树")
    print(f"树序列包含 {ts.num_mutations} 个突变")
    print(f"序列长度: {ts.sequence_length:.0f} bp")
    
    # 保存树序列到data目录
    output_file = "../data/scenario2_three_populations_chr22.trees"
    ts.dump(output_file)
    print(f"树序列已保存到: {output_file}")

    # 导出VCF格式到data目录
    vcf_file = "../data/scenario2_three_populations_chr22.vcf"
    with open(vcf_file, "w") as f:
        ts.write_vcf(f)
    print(f"VCF文件已保存到: {vcf_file}")
    
    return ts

def analyze_population_structure(ts):
    """
    分析人群结构
    """
    print("\n开始分析人群结构...")
    
    # 创建输出目录
    os.makedirs("../plots", exist_ok=True)
    
    # 获取人群标签
    population_labels = []
    for pop_id, pop in enumerate(ts.populations()):
        pop_metadata = pop.metadata
        if pop_metadata and 'name' in pop_metadata:
            pop_name = pop_metadata['name']
        else:
            pop_name = f"Pop_{pop_id}"
        
        # 计算该人群的样本数量
        pop_samples = [ind.id for ind in ts.individuals() if ind.population == pop_id]
        population_labels.extend([pop_name] * len(pop_samples))
    
    # 如果没有元数据，使用默认标签
    if len(population_labels) != ts.num_individuals:
        population_labels = []
        samples_per_pop = ts.num_individuals // 3
        population_labels.extend(['YRI'] * samples_per_pop)
        population_labels.extend(['CEU'] * samples_per_pop)
        population_labels.extend(['CHB'] * (ts.num_individuals - 2 * samples_per_pop))

    # 计算基因型矩阵
    print("计算基因型矩阵...")
    genotype_matrix = ts.genotype_matrix()

    # 转置矩阵使得行为样本，列为SNP
    genotype_matrix = genotype_matrix.T

    # 为每个样本创建标签（每个个体有2个样本）
    sample_labels = []
    for ind_label in population_labels:
        sample_labels.extend([ind_label, ind_label])  # 每个个体有2个样本

    # 确保标签数量与样本数量匹配
    if len(sample_labels) != ts.num_samples:
        print(f"警告: 标签数量 ({len(sample_labels)}) 与样本数量 ({ts.num_samples}) 不匹配")
        # 重新创建标签
        sample_labels = []
        samples_per_pop = ts.num_samples // 3
        sample_labels.extend(['YRI'] * samples_per_pop)
        sample_labels.extend(['CEU'] * samples_per_pop)
        sample_labels.extend(['CHB'] * (ts.num_samples - 2 * samples_per_pop))
    
    # 只使用前1000个SNP进行PCA分析（减少计算时间）
    max_snps = min(1000, genotype_matrix.shape[1])
    genotype_subset = genotype_matrix[:, :max_snps]
    
    print(f"使用 {genotype_subset.shape[0]} 个样本和 {genotype_subset.shape[1]} 个SNP进行分析")

    return genotype_subset, sample_labels

def plot_population_analysis(genotype_matrix, population_labels):
    """
    绘制人群分析图表
    """
    print("生成人群分析图表...")
    
    # 创建颜色映射
    unique_pops = list(set(population_labels))
    colors = ['red', 'blue', 'green'][:len(unique_pops)]
    color_map = dict(zip(unique_pops, colors))
    
    plt.figure(figsize=(15, 10))
    
    # 1. PCA分析
    plt.subplot(2, 3, 1)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(genotype_matrix)
    
    for pop in unique_pops:
        mask = [label == pop for label in population_labels]
        plt.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                   c=color_map[pop], label=pop, alpha=0.6, s=20)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.title('Principal Component Analysis (PCA)')
    plt.legend()

    # 2. 计算每个人群的多样性
    plt.subplot(2, 3, 2)
    diversities = []
    pop_names = []

    for pop in unique_pops:
        pop_indices = [i for i, label in enumerate(population_labels) if label == pop]
        if pop_indices:
            pop_genotypes = genotype_matrix[pop_indices, :]
            # 计算每个位点的等位基因频率
            allele_freqs = np.mean(pop_genotypes, axis=0)
            # 计算期望杂合度
            diversity = np.mean(2 * allele_freqs * (1 - allele_freqs))
            diversities.append(diversity)
            pop_names.append(pop)

    bars = plt.bar(pop_names, diversities, color=[color_map[pop] for pop in pop_names])
    plt.ylabel('Expected Heterozygosity')
    plt.title('Genetic Diversity by Population')
    plt.xticks(rotation=45)
    
    # 3. 人群间Fst计算
    plt.subplot(2, 3, 3)
    fst_matrix = np.zeros((len(unique_pops), len(unique_pops)))
    
    for i, pop1 in enumerate(unique_pops):
        for j, pop2 in enumerate(unique_pops):
            if i != j:
                pop1_indices = [k for k, label in enumerate(population_labels) if label == pop1]
                pop2_indices = [k for k, label in enumerate(population_labels) if label == pop2]
                
                if pop1_indices and pop2_indices:
                    pop1_genotypes = genotype_matrix[pop1_indices, :]
                    pop2_genotypes = genotype_matrix[pop2_indices, :]
                    
                    # 简化的Fst计算
                    p1 = np.mean(pop1_genotypes, axis=0)
                    p2 = np.mean(pop2_genotypes, axis=0)
                    pt = (p1 + p2) / 2
                    
                    # 避免除零错误
                    valid_sites = (pt > 0) & (pt < 1)
                    if np.sum(valid_sites) > 0:
                        ht = np.mean(2 * pt[valid_sites] * (1 - pt[valid_sites]))
                        hs = np.mean(2 * p1[valid_sites] * (1 - p1[valid_sites]) + 
                                   2 * p2[valid_sites] * (1 - p2[valid_sites])) / 2
                        fst = (ht - hs) / ht if ht > 0 else 0
                        fst_matrix[i, j] = max(0, fst)  # 确保Fst非负
    
    im = plt.imshow(fst_matrix, cmap='viridis')
    plt.colorbar(im)
    plt.xticks(range(len(unique_pops)), unique_pops, rotation=45)
    plt.yticks(range(len(unique_pops)), unique_pops)
    plt.title('Pairwise Fst Distance')

    # 4. 等位基因频率分布
    plt.subplot(2, 3, 4)
    for pop in unique_pops:
        pop_indices = [i for i, label in enumerate(population_labels) if label == pop]
        if pop_indices:
            pop_genotypes = genotype_matrix[pop_indices, :]
            allele_freqs = np.mean(pop_genotypes, axis=0)
            plt.hist(allele_freqs, bins=20, alpha=0.5, label=pop,
                    color=color_map[pop], density=True)

    plt.xlabel('Allele Frequency')
    plt.ylabel('Density')
    plt.title('Allele Frequency Distribution')
    plt.legend()

    # 5. 样本数量统计
    plt.subplot(2, 3, 5)
    pop_counts = [population_labels.count(pop) for pop in unique_pops]
    bars = plt.bar(unique_pops, pop_counts, color=[color_map[pop] for pop in unique_pops])
    plt.ylabel('Sample Count')
    plt.title('Sample Count by Population')
    plt.xticks(rotation=45)
    
    # 在柱状图上添加数值标签
    for bar, count in zip(bars, pop_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(count), ha='center', va='bottom')
    
    # 6. MDS分析
    plt.subplot(2, 3, 6)
    # 计算样本间距离矩阵（使用前100个样本以减少计算时间）
    n_samples = min(300, genotype_matrix.shape[0])
    subset_genotypes = genotype_matrix[:n_samples, :]
    subset_labels = population_labels[:n_samples]
    
    # 计算欧几里得距离
    from scipy.spatial.distance import pdist, squareform
    distances = pdist(subset_genotypes, metric='euclidean')
    distance_matrix = squareform(distances)
    
    # MDS分析
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    mds_result = mds.fit_transform(distance_matrix)
    
    for pop in unique_pops:
        mask = [label == pop for label in subset_labels]
        if any(mask):
            plt.scatter(mds_result[mask, 0], mds_result[mask, 1], 
                       c=color_map[pop], label=pop, alpha=0.6, s=20)
    
    plt.xlabel('MDS1')
    plt.ylabel('MDS2')
    plt.title('Multidimensional Scaling (MDS)')
    plt.legend()

    plt.tight_layout()
    plt.savefig("../plots/scenario2_population_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

    return fst_matrix, diversities

def main():
    """
    主函数
    """
    print("=" * 70)
    print("stdpopsim 情景2模拟: 东亚、非洲、欧洲三个人群各1000个基因组的22号染色体")
    print("=" * 70)
    
    # 运行模拟
    ts = simulate_three_populations_chr22()
    
    # 分析人群结构
    genotype_matrix, population_labels = analyze_population_structure(ts)
    
    # 绘制分析图表
    fst_matrix, diversities = plot_population_analysis(genotype_matrix, population_labels)
    
    # 保存统计结果
    unique_pops = list(set(population_labels))
    
    # 基本统计
    stats_df = pd.DataFrame({
        'Statistic': ['Total Sample Count', 'Tree Count', 'Mutation Count', 'Sequence Length (bp)'],
        'Value': [ts.num_samples, ts.num_trees, ts.num_mutations, int(ts.sequence_length)]
    })

    # 人群多样性
    diversity_df = pd.DataFrame({
        'Population': unique_pops,
        'Expected_Heterozygosity': diversities,
        'Sample_Count': [population_labels.count(pop) for pop in unique_pops]
    })
    
    # Fst矩阵
    fst_df = pd.DataFrame(fst_matrix, index=unique_pops, columns=unique_pops)
    
    # 保存结果到data目录
    stats_df.to_csv("../data/scenario2_basic_statistics.csv", index=False)
    diversity_df.to_csv("../data/scenario2_population_diversity.csv", index=False)
    fst_df.to_csv("../data/scenario2_fst_matrix.csv")
    
    print("\n" + "=" * 70)
    print("情景2模拟完成!")
    print("=" * 70)
    print("\n生成的文件:")
    print("- scenario2_three_populations_chr22.trees (树序列文件)")
    print("- scenario2_three_populations_chr22.vcf (VCF格式)")
    print("- scenario2_basic_statistics.csv (基本统计)")
    print("- scenario2_population_diversity.csv (人群多样性)")
    print("- scenario2_fst_matrix.csv (Fst距离矩阵)")
    print("- plots/scenario2_population_analysis.png (人群分析图表)")

if __name__ == "__main__":
    main()
