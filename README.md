# bayesian_ab_test

A Python package built on PyMC3 (https://docs.pymc.io/) for AB testing using Bayesian methods. Priors are assumed to be uniform and posteriors are assumed to be Bernoulli.

Dependencies include:
- pymc3
- scipy
- matplotlib
- seaborn
- pandas
- numpy

Example:
```
# import dependencies
from bayesian_ab_test import bayesian_ab_test

# run test
test = bayesian_ab_test(sample_a_total=4590, 
                        sample_a_responses=1360, 
                        sample_b_total=3975, 
                        sample_b_responses=1215,
                        N_simulations=1000, 
                        pct_tune=50, 
                        gr_threshold=1, 
                        N_additional_draws=1000)
```
Argument definitions:
- sample_a_total: total opportunity for sample A to respond (i.e., total emails sent to sample A)
- sample_a_responses: total responses for sample A (i.e., number of clicks in sample A)
- sample_b_total: total opportunity for sample B to respond (i.e., total emails sent to sample B)
- sample_b_responses: total responses for sample B (i.e., number of clicks in sample B)
- N_simulations: number of tuned samples to draws (see https://docs.pymc.io/api/inference.html; default = 1000)
- pct_tune: percentage of N_simulations to use for tuning (note: this number is added back to N_simulations; default = 50)
- gr_threshold: threshold to use for Gelman-Rubin statistic to determine if additional draws are necessary (default = 1)
- N_additional_draws: number of draws to add to N_simulations is Gelman-Rubin threshold is not met (default = 1000)

Attributes that can be returned:
```
# Data frame of metrics
test.df

# Least plausible value plot
test.lpv_plot

# Bayesian fraction missing information
test.bfmi

# Maximum gelman-rubin statistic
test.max_gr

# Plot of distributions
test.dist_plot

# Proportion of sample A greater than B
test.proportion_A_greater_than_B

# Proportion of sample B greater than A
test.proportion_B_greater_than_A

# t value from hypothesis test
test.t_test_t

# p value from hypothesis test
test.t_test_sig

# Cohens d effect size
test.cohens_d

# Interpretation of effect size
test.size_of_effect

# Conclusion from t test in APA format
test.t_test_conclusion

# Bar plot comparing means
test.bar_plot
```
To install, use: pip install git+https://github.com/aaronengland/bayesian_ab_test.git

Source: http://camdavidsonpilon.github.io/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/
