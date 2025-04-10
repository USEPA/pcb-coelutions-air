# pcb-coelutions-air Repository

## Brief Project Description

This repository contains files to recreate the analysis in "Resolving polychlorinated biphenyl congener coelutions: a case study in environmental air samples" produced by the Office of Research and Development (ORD) Center for Public Health and Environemntal Assessment (CPHEA) and the Center for Computational Toxicology and Exposure (CCTE).

## Dependencies
The major dependencies needed to run this code involve both `R` and `Python` packages. Install a conda (mamba) - specific installation of `R` using `rpy2`

### Python

1. [rpy2](https://rpy2.github.io/)
2. [seaborn](https://seaborn.pydata.org/)
3. [arviz](https://python.arviz.org/en/stable/)

These can be installed with `mamba install rpy2 seaborn arviz`

### R
Use the `R` built for the conda (mamba) environment. When using miniforge3, this build can be found at `miniforge/bin/R`

1. brms
2. tidybayes
3. dplyr
3. tidyr

These can be installed within an R instnace using `install.packages(c('brms', 'tidybayes', 'dplyr', 'tidyr'))`

## Running the code
All extracted data used for the analysis is in `data/` and each coelution-specific regression model is built as a `process_coelution.coelution` object.

Example quickstart (`run_coelutions.ipynb` provides live notebook)
```
from process_coelution import coelution
PCBs = ['PCB17', 'PCB18', 'PCB49'] # Define list of PCBs present in coelution
model_type = 'intercept_only' # Type of regression model to run: intercept_only, sample_only, etc.
my_coelution = coelution(PCBs, model_type=model_type)
my_coelution.prep_data() # Prep the raw data and use the specified PCB columns to calculate proportions
my_coelution.fit_coeultion(disp_summary) # fit the dirichlet regression. disp_summary=True prints the brms summary statistics from the fit
my_coelution.sample_posterior() # Sample the posterior fits to predict probabilstic proportions of the coelution

display(my_coelution.prop_summry) # Summary of posterior distributions for PCB-specific propositons
```

The `run_all_pcbs_model.py` script in `scripts/` carries out the full analysis for a specific `model_type` and all traces are saved to `brms_models/<model_type>/`. Model selection for each `model_type` is stored in `model_selection_<model_type>.csv` with y-randomiation results in `model_selection_<model_type>_randomized.csv`.

The corresponding `manuscript_results.ipynb` notebook contains the post-processing analysis following creation of all regression models.

This repository contains the link to [EPA's System Lifecycle Management Policy and Procedure](https://www.epa.gov/irmpoli8/policy-procedures-and-guidance-system-life-cycle-management-slcm) which lays out EPA's Open Source Software Policy and [EPA's Open Source Code Guidance](https://www.epa.gov/developers/open-source-software-and-epa-code-repository-requirements). 

### Disclaimer

The United States Environmental Protection Agency (EPA) GitHub project code is provided on an "as is" basis and the user assumes responsibility for its use.  EPA has relinquished control of the information and no longer has responsibility to protect the integrity , confidentiality, or availability of the information.  Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by EPA.  The EPA seal and logo shall not be used in any manner to imply endorsement of any commercial product or activity by EPA or the United States Government.
