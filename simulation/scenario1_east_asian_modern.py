#!/usr/bin/env python3
"""
情景1: 1000个东亚现代人群基因组的22号染色体模拟
使用stdpopsim模拟东亚人群的22号染色体数据
"""

import stdpopsim
import tskit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def simulate_east_asian_chr22():
    """
    模拟1000个东亚现代人群的22号染色体
    """
    print("开始模拟情景1: 1000个东亚现代人群基因组的22号染色体")
    
    # 设置随机种子以确保结果可重现
    np.random.seed(42)
    
    # 获取人类物种和22号染色体
    species = stdpopsim.get_species("HomSap")
    contig = species.get_contig("chr22", length_multiplier=0.1)  # 使用10%的长度以加快模拟
    
    # 使用OutOfAfrica_3G09模型，这个模型包含东亚人群(CHB)
    model = species.get_demographic_model("OutOfAfrica_3G09")
    
    # 设置样本 - 1000个东亚人群个体(CHB)
    samples = {"CHB": 1000}
    
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
    output_file = "../data/scenario1_east_asian_chr22.trees"
    ts.dump(output_file)
    print(f"树序列已保存到: {output_file}")

    # 导出VCF格式到data目录
    vcf_file = "../data/scenario1_east_asian_chr22.vcf"
    with open(vcf_file, "w") as f:
        ts.write_vcf(f)
    print(f"VCF文件已保存到: {vcf_file}")
    
    return ts

def analyze_and_plot_results(ts):
    """
    分析模拟结果并生成图表
    """
    print("\n开始分析模拟结果...")

    # 创建输出目录
    os.makedirs("../plots", exist_ok=True)
    
    # 1. 计算多样性统计
    diversity = ts.diversity()
    print(f"Nucleotide diversity (π): {diversity:.6f}")

    # 2. 计算Tajima's D
    tajimas_d = ts.Tajimas_D()
    print(f"Tajima's D: {tajimas_d:.6f}")

    # 3. 绘制站点频谱(SFS)
    plt.figure(figsize=(12, 8))

    # 计算站点频谱
    afs = ts.allele_frequency_spectrum(polarised=True)

    plt.subplot(2, 2, 1)
    plt.bar(range(len(afs)), afs)
    plt.xlabel("Allele Frequency Category")
    plt.ylabel("Number of Sites")
    plt.title("Site Frequency Spectrum (SFS)")
    plt.yscale('log')

    # 4. 绘制树高度分布
    plt.subplot(2, 2, 2)
    tree_heights = [tree.total_branch_length for tree in ts.trees()]
    plt.hist(tree_heights, bins=50, alpha=0.7)
    plt.xlabel("Tree Height")
    plt.ylabel("Frequency")
    plt.title("Tree Height Distribution")
    
    # 5. 绘制突变密度
    plt.subplot(2, 2, 3)
    # 修复tskit API兼容性问题
    positions = []
    for variant in ts.variants():
        positions.append(variant.site.position)

    if positions:
        plt.hist(positions, bins=100, alpha=0.7)
        plt.xlabel("Genomic Position")
        plt.ylabel("Number of Mutations")
        plt.title("Mutation Density Distribution")
    else:
        plt.text(0.5, 0.5, "No mutation data", ha='center', va='center', transform=plt.gca().transAxes)
        plt.title("Mutation Density Distribution")

    # 6. 绘制样本关系树(前100个样本)
    plt.subplot(2, 2, 4)
    # 简化树序列以便可视化
    simplified_ts = ts.simplify(samples=list(range(min(100, ts.num_samples))))
    tree = simplified_ts.first()

    # 计算样本间的遗传距离矩阵
    num_samples = min(50, simplified_ts.num_samples)  # 限制样本数以便可视化
    distances = np.zeros((num_samples, num_samples))

    for i in range(num_samples):
        for j in range(i+1, num_samples):
            # 计算两个样本间的遗传距离
            mrca_time = tree.tmrca(i, j)
            distances[i, j] = distances[j, i] = mrca_time

    # 绘制距离矩阵热图
    im = plt.imshow(distances, cmap='viridis')
    plt.colorbar(im)
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Index")
    plt.title("Genetic Distance Between Samples")
    
    plt.tight_layout()
    plt.savefig("../plots/scenario1_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 保存统计结果到data目录
    stats_df = pd.DataFrame({
        'Statistic': ['Sample Count', 'Tree Count', 'Mutation Count', 'Sequence Length (bp)', 'Nucleotide Diversity', "Tajima's D"],
        'Value': [ts.num_samples, ts.num_trees, ts.num_mutations,
               int(ts.sequence_length), diversity, tajimas_d]
    })

    stats_df.to_csv("../data/scenario1_statistics.csv", index=False)
    print("Statistics saved to: ../data/scenario1_statistics.csv")
    
    return stats_df

def main():
    """
    主函数
    """
    print("=" * 60)
    print("stdpopsim 情景1模拟: 1000个东亚现代人群基因组的22号染色体")
    print("=" * 60)
    
    # 运行模拟
    ts = simulate_east_asian_chr22()
    
    # 分析结果
    stats = analyze_and_plot_results(ts)
    
    print("\n" + "=" * 60)
    print("情景1模拟完成!")
    print("=" * 60)
    print("\n生成的文件:")
    print("- scenario1_east_asian_chr22.trees (树序列文件)")
    print("- scenario1_east_asian_chr22.vcf (VCF格式)")
    print("- scenario1_statistics.csv (统计结果)")
    print("- plots/scenario1_analysis.png (分析图表)")

if __name__ == "__main__":
    main()
