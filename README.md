# bayesian_ab_test

A Python package built on PyMC3 for AB testing using Bayesian methods.

Dependencies include:
- pymc3
- scipy
- matplotlib
- seaborn
- pandas
- numpy

Example:

from bayesian_ab_test import *

test = bayesian_ab_test(sample_a_total=4590, 
                        sample_a_responses=1360, 
                        sample_b_total=3975, 
                        sample_b_responses=1215,
                        N_simulations=1000, 
                        pct_tune=50, 
                        gr_threshold=1, 
                        N_additional_draws=1000)

Return attributes:

Data frame of metrics
- test.df

Least plausible value plot
- test.lpv_plot

Trace from PyMC3 draws
- test.trace

Bayesian fraction missing information
- test.bmfi

Maximum gelman-rubin statistic
- test.max_gr

Plot of distributions
- test.dist_plot

Proportion of sample A greater than B
- test.proportion_A_greater_than_B

Proportion of sample B greater than A
- test.proportion_B_greater_than_A

t value from hypothesis test
- test.t_test_t

p value from hypothesis test
- test.t_test_sig

Cohens d effect size
- test.cohens_d

Interpretation of effect size
- test.size_of_effect

Conclusion from t test in APA format
- test.t_test_conclusion

Bar plot comparing means
- test.bar_plot

To install, use: pip install git+https://github.com/aaronengland/bayesian_ab_test.git
