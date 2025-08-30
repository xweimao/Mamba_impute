#!/usr/bin/env python3
"""
运行所有三个stdpopsim模拟情景的主控脚本
"""

import subprocess
import sys
import os
import time
from datetime import datetime

def install_dependencies():
    """
    安装必要的Python包
    """
    print("检查并安装必要的依赖包...")
    
    required_packages = [
        'matplotlib',
        'pandas', 
        'scikit-learn',
        'seaborn',
        'scipy'
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package} 已安装")
        except ImportError:
            print(f"安装 {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} 安装完成")

def run_scenario(script_name, scenario_name):
    """
    运行单个情景脚本
    """
    print(f"\n{'='*60}")
    print(f"开始运行 {scenario_name}")
    print(f"脚本: {script_name}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # 运行脚本
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✓ {scenario_name} 运行成功!")
        print(f"运行时间: {duration:.2f} 秒")
        
        # 显示输出
        if result.stdout:
            print("\n--- 标准输出 ---")
            print(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✗ {scenario_name} 运行失败!")
        print(f"运行时间: {duration:.2f} 秒")
        print(f"错误代码: {e.returncode}")
        
        if e.stdout:
            print("\n--- 标准输出 ---")
            print(e.stdout)
        
        if e.stderr:
            print("\n--- 错误输出 ---")
            print(e.stderr)
        
        return False
    
    except Exception as e:
        print(f"\n✗ {scenario_name} 运行时发生未知错误: {e}")
        return False

def create_summary_report():
    """
    创建总结报告
    """
    print(f"\n{'='*60}")
    print("创建总结报告...")
    print(f"{'='*60}")

    # 检查生成的文件 - 根据新的目录结构调整路径
    expected_files = {
        "情景1": [
            "../data/scenario1_east_asian_chr22.trees",
            "../data/scenario1_east_asian_chr22.vcf",
            "../data/scenario1_statistics.csv",
            "../plots/scenario1_analysis.png"
        ],
        "情景2": [
            "../data/scenario2_three_populations_chr22.trees",
            "../data/scenario2_three_populations_chr22.vcf",
            "../data/scenario2_basic_statistics.csv",
            "../data/scenario2_population_diversity.csv",
            "../data/scenario2_fst_matrix.csv",
            "../plots/scenario2_population_analysis.png"
        ],
        "情景3": [
            "../data/scenario3_ancient_modern_chr22.trees",
            "../data/scenario3_ancient_modern_chr22.vcf",
            "../data/scenario3_sample_info.csv",
            "../data/scenario3_statistics.csv",
            "../plots/scenario3_temporal_analysis.png"
        ]
    }
    
    report = []
    report.append("stdpopsim 三个情景模拟总结报告")
    report.append("=" * 50)
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    for scenario, files in expected_files.items():
        report.append(f"{scenario}:")
        report.append("-" * 20)
        
        for file_path in files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                report.append(f"  ✓ {file_path} ({file_size:,} bytes)")
            else:
                report.append(f"  ✗ {file_path} (未找到)")
        report.append("")
    
    # 统计总文件数和大小
    total_files = 0
    total_size = 0
    
    for files in expected_files.values():
        for file_path in files:
            if os.path.exists(file_path):
                total_files += 1
                total_size += os.path.getsize(file_path)
    
    report.append("总计:")
    report.append(f"  生成文件数: {total_files}")
    report.append(f"  总文件大小: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")
    
    # 保存报告到项目根目录
    report_path = "../simulation_summary_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    # 显示报告
    print("\n".join(report))
    print(f"\n报告已保存到: {report_path}")

def main():
    """
    主函数
    """
    print("stdpopsim 三个情景模拟 - 主控脚本")
    print("=" * 60)
    print("本脚本将依次运行以下三个情景:")
    print("1. 1000个东亚现代人群基因组的22号染色体")
    print("2. 东亚、非洲、欧洲三个人群各1000个基因组的22号染色体")
    print("3. 三个现代人群各1000个基因组 + 古DNA样本的22号染色体")
    print("=" * 60)
    
    # 安装依赖
    install_dependencies()

    # 创建输出目录 - 根据新的目录结构
    os.makedirs("../data", exist_ok=True)
    os.makedirs("../plots", exist_ok=True)
    
    # 定义要运行的脚本
    scenarios = [
        ("scenario1_east_asian_modern.py", "情景1: 东亚现代人群"),
        ("scenario2_three_populations.py", "情景2: 三个现代人群"),
        ("scenario3_ancient_modern.py", "情景3: 现代人群 + 古DNA")
    ]
    
    # 记录总体开始时间
    total_start_time = time.time()
    successful_scenarios = 0
    
    # 运行每个情景
    for script_name, scenario_name in scenarios:
        if os.path.exists(script_name):
            success = run_scenario(script_name, scenario_name)
            if success:
                successful_scenarios += 1
        else:
            print(f"\n✗ 脚本文件 {script_name} 不存在，跳过 {scenario_name}")
    
    # 计算总运行时间
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # 创建总结报告
    create_summary_report()
    
    # 显示最终结果
    print(f"\n{'='*60}")
    print("所有情景运行完成!")
    print(f"{'='*60}")
    print(f"成功运行的情景: {successful_scenarios}/{len(scenarios)}")
    print(f"总运行时间: {total_duration:.2f} 秒 ({total_duration/60:.2f} 分钟)")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if successful_scenarios == len(scenarios):
        print("\n🎉 所有情景都成功完成!")
    else:
        print(f"\n⚠️  有 {len(scenarios) - successful_scenarios} 个情景运行失败")
    
    print("\n生成的主要文件:")
    print("- *.trees: 树序列文件")
    print("- *.vcf: VCF格式基因型文件") 
    print("- *.csv: 统计分析结果")
    print("- plots/*.png: 分析图表")
    print("- simulation_summary_report.txt: 总结报告")

if __name__ == "__main__":
    main()
