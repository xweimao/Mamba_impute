# Genome Simulation Notebooks

This directory contains Jupyter notebooks for understanding and using stdpopsim for genome data simulation, specifically designed for imputation research.

## Notebooks Overview

### 1. `genome_simulation_tutorial.ipynb`
**Comprehensive Tutorial Notebook**

A detailed educational notebook covering:
- Introduction to stdpopsim concepts and workflow
- Three simulation scenarios with increasing complexity
- Population genetic analysis and visualization
- Comparative analysis across scenarios
- Theoretical background and practical applications

**Best for**: Learning population genetics concepts and understanding stdpopsim in depth.

### 2. `stdpopsim_practical_guide.ipynb`
**Practical Implementation Guide**

A hands-on notebook focused on:
- Quick setup and configuration
- Essential simulation workflows
- Data export for downstream analysis
- Creating missing data scenarios for imputation testing
- Ready-to-use code snippets

**Best for**: Practical implementation and generating data for imputation research.

## Getting Started

### Prerequisites

```bash
# Install required packages
conda install -c conda-forge stdpopsim tskit
pip install matplotlib pandas seaborn scikit-learn
```

### Quick Start

1. **For learning**: Start with `genome_simulation_tutorial.ipynb`
2. **For research**: Use `stdpopsim_practical_guide.ipynb`

### Running the Notebooks

```bash
# Navigate to notebook directory
cd notebook/

# Start Jupyter
jupyter notebook

# Or use JupyterLab
jupyter lab
```

## Simulation Scenarios

### Scenario 1: Single Population (East Asian)
- **Purpose**: Basic simulation workflow
- **Samples**: 1,000 CHB individuals
- **Focus**: Understanding tree sequences and basic statistics

### Scenario 2: Three Modern Populations
- **Purpose**: Population structure analysis
- **Samples**: 1,000 each of YRI, CEU, CHB
- **Focus**: PCA, Fst, population differentiation

### Scenario 3: Ancient + Modern Populations
- **Purpose**: Temporal genetic analysis
- **Samples**: Modern + ancient DNA samples
- **Focus**: Genetic diversity changes over time

## Key Features

### Data Generation
- Realistic human demographic models (Out-of-Africa)
- Chromosome 22 simulations with adjustable length
- Multiple output formats (VCF, CSV, tree sequences)

### Analysis Tools
- Population genetic statistics (π, Tajima's D, Fst)
- Principal Component Analysis (PCA)
- Site Frequency Spectrum (SFS)
- Genetic diversity visualization

### Imputation Testing
- Missing data scenario creation
- Truth vs. incomplete data comparison
- Customizable missing data rates
- Ready-to-use imputation test datasets

## Output Files

The notebooks generate several types of output files:

### Simulation Data
- `*.trees` - Native tree sequence format
- `*.vcf` - Variant Call Format for standard tools
- `*_samples.csv` - Sample metadata and population information
- `*_statistics.csv` - Basic population genetic statistics

### Imputation Testing
- `*_truth.csv` - Complete genotype data (ground truth)
- `*_incomplete.csv` - Data with missing genotypes
- `*_missing_mask.csv` - Boolean mask of missing positions

### Visualizations
- Population structure plots (PCA, diversity, Fst)
- Temporal analysis charts
- Comparative statistics graphs

## Customization Options

### Simulation Parameters
```python
# Adjust chromosome length
contig = species.get_contig("chr22", length_multiplier=0.5)  # 50% of chr22

# Change sample sizes
samples = {"YRI": 2000, "CEU": 1500, "CHB": 1000}

# Different demographic models
model = species.get_demographic_model("OutOfAfrica_2T12")
```

### Missing Data Patterns
```python
# Different missing rates
create_missing_data_scenario(ts, missing_rate=0.05)  # 5% missing
create_missing_data_scenario(ts, missing_rate=0.20)  # 20% missing

# Population-specific missing patterns
# (implement custom masking logic)
```

## Integration with Imputation Software

The generated data can be directly used with:

- **BEAGLE**: Use VCF files
- **IMPUTE2/4**: Convert VCF to required format
- **Minimac**: Use VCF files
- **Custom algorithms**: Use CSV genotype matrices

## Tips for Research Use

1. **Start small**: Use `length_multiplier=0.1` for testing
2. **Scale up**: Increase to `length_multiplier=1.0` for full analysis
3. **Multiple replicates**: Change `seed` parameter for independent runs
4. **Different chromosomes**: Try various chromosomes for different LD patterns
5. **Custom scenarios**: Modify demographic models for specific research questions

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce `length_multiplier` or sample sizes
2. **Slow simulations**: Use smaller chromosomes or fewer samples
3. **Missing dependencies**: Check conda/pip installations
4. **Plot display issues**: Ensure matplotlib backend is properly configured

### Performance Tips

- Use `length_multiplier=0.1` for development/testing
- Limit sample sizes to <5000 per population for interactive use
- Consider using cluster computing for large-scale simulations

## Further Reading

- [stdpopsim documentation](https://popsim-consortium.github.io/stdpopsim-docs/)
- [tskit documentation](https://tskit.dev/)
- [Population genetics primer](https://github.com/popsim-consortium/stdpopsim/blob/main/docs/tutorial.rst)

## Citation

If you use these notebooks in your research, please cite:
- stdpopsim: Adrion et al. (2020) eLife
- tskit: Kelleher et al. (2018) Nature Genetics
