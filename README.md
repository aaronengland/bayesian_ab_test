# bayesian_ab_test

A Python package built on [PyMC3](https://docs.pymc.io/) for AB testing using Bayesian methods. Priors are assumed to be uniform and posteriors are assumed to be Bernoulli.

Dependencies include:
- pymc3
- scipy
- matplotlib
- seaborn
- pandas
- numpy

## bayesian_ab_test_count

Example:
```
# import dependencies
from bayesian_ab_test import bayesian_ab_test_count

# run test
test = bayesian_ab_test_prob(sample_A_n=100, 
                             sample_A_count=50, 
                             sample_B_n=50, 
                             sample_B_count=15,
                             N_simulations=1000, 
                             pct_tune=50, 
                             gr_threshold=1.001, 
                             N_additional_draws=1000,
                             lpv_height=15,
                             n_x_observed=2)
```
Argument definitions:
- ```sample_A_n```: sample size for sample A
- ```sample_A_count```: count (i.e., responses) for sample A
- ```sample_B_n```: sample size for sample B
- ```sample_B_count```: count (i.e., responses) for sample B
- ```N_simulations```: number of tuned samples to draws (see [PyMC3 documentation](https://docs.pymc.io/api/inference.html); default = 1000)
- ```pct_tune```: percentage of ```N_simulations``` to use for tuning (note: this number is added back to N_simulations; default = 50)
- ```gr_threshold```: threshold to use for Gelman-Rubin statistic to determine if additional draws are necessary (default = 1.001)
- ```N_additional_draws```: number of draws to add to ```N_simulations``` if Gelman-Rubin threshold is not met (default = 1000)
- ```lpv_height```: height of the vertical LPV line in the ```dist_plot``` attribute (default = 15)
- ```n_x_observed```: value > 1 to multiply ```sample_A_n```, ```sample_A_count```, ```sample_B_n```, and ```sample_B_count``` by when generating prior distributions (the larger the number, the greater the value possible in the range of the flat distribution; default = 2)

Attributes that can be returned:
```
# Data frame of metrics
test.df

# Lower plausible value plot
test.lpv_plot
```
<a href="https://www.codecogs.com/eqnedit.php?latex=UpperPlausibleValue&space;=&space;\frac{a}{a&plus;b}&plus;1.65\sqrt{\frac{ab}{(a&plus;b)^{2}(a&plus;b&plus;1)}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?UpperPlausibleValue&space;=&space;\frac{a}{a&plus;b}&plus;1.65\sqrt{\frac{ab}{(a&plus;b)^{2}(a&plus;b&plus;1)}}" title="UpperPlausibleValue = \frac{a}{a+b}+1.65\sqrt{\frac{ab}{(a+b)^{2}(a+b+1)}}" /></a>

<img src="https://latex.codecogs.com/gif.latex?LowerPlausibleValue&space;=&space;\frac{a}{a&plus;b}&space;-&space;1.65\sqrt{\frac{ab}{(a&plus;b)^{2}(a&plus;b&plus;1)}}" title="LowerPlausibleValue = \frac{a}{a+b} - 1.65\sqrt{\frac{ab}{(a+b)^{2}(a+b+1)}}" /></a>

Where:

<a href="https://www.codecogs.com/eqnedit.php?latex=a&space;=&space;1&space;&plus;&space;N_{yes}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a&space;=&space;1&space;&plus;&space;N_{yes}" title="a = 1 + N_{yes}" /></a>

<img src="https://latex.codecogs.com/gif.latex?b&space;=&space;1&space;&plus;&space;N_{nonclicks}" title="b = 1 + N_{nonclicks}" /></a>

```
# Distribution of sample A posterior
test.sample_A_count_div_n_list

# Distribution of sample B posterior
test.sample_B_count_div_n_list

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
```

## bayesian_ab_test_prob

Example:
```
# import dependencies
from bayesian_ab_test import bayesian_ab_test_prob

# run test
test = bayesian_ab_test_prob(sample_a_total=4590, 
                             sample_a_responses=1360, 
                             sample_b_total=3975, 
                             sample_b_responses=1215,
                             N_simulations=1000, 
                             pct_tune=50, 
                             gr_threshold=1, 
                             N_additional_draws=1000)
```
Argument definitions:
- ```sample_a_total```: total opportunity for sample A to respond (i.e., total emails sent to sample A)
- ```sample_a_responses```: total responses for sample A (i.e., number of clicks in sample A)
- ```sample_b_total```: total opportunity for sample B to respond (i.e., total emails sent to sample B)
- ```sample_b_responses```: total responses for sample B (i.e., number of clicks in sample B)
- ```N_simulations```: number of tuned samples to draws (see [PyMC3 documentation](https://docs.pymc.io/api/inference.html); default = 1000)
- ```pct_tune```: percentage of ```N_simulations``` to use for tuning (note: this number is added back to N_simulations; default = 50)
- ```gr_threshold```: threshold to use for Gelman-Rubin statistic to determine if additional draws are necessary (default = 1.001)
- ```N_additional_draws```: number of draws to add to ```N_simulations``` if Gelman-Rubin threshold is not met (default = 1000)
- ```lpv_height```: height of the vertical LPV line in the ```dist_plot``` attribute (default = 15)
- ```n_x_observed```: value > 1 to multiply ```sample_A_n```, ```sample_A_count```, ```sample_B_n```, and ```sample_B_count``` by when generating prior distributions (the larger the number, the greater the value possible in the range of the flat distribution; default = 2)

Attributes that can be returned:
```
# Data frame of metrics
test.df

# Lower plausible value plot
test.lpv_plot
```
<a href="https://www.codecogs.com/eqnedit.php?latex=UpperPlausibleValue&space;=&space;\frac{a}{a&plus;b}&plus;1.65\sqrt{\frac{ab}{(a&plus;b)^{2}(a&plus;b&plus;1)}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?UpperPlausibleValue&space;=&space;\frac{a}{a&plus;b}&plus;1.65\sqrt{\frac{ab}{(a&plus;b)^{2}(a&plus;b&plus;1)}}" title="UpperPlausibleValue = \frac{a}{a+b}+1.65\sqrt{\frac{ab}{(a+b)^{2}(a+b+1)}}" /></a>

<img src="https://latex.codecogs.com/gif.latex?LowerPlausibleValue&space;=&space;\frac{a}{a&plus;b}&space;-&space;1.65\sqrt{\frac{ab}{(a&plus;b)^{2}(a&plus;b&plus;1)}}" title="LowerPlausibleValue = \frac{a}{a+b} - 1.65\sqrt{\frac{ab}{(a+b)^{2}(a+b+1)}}" /></a>

Where:

<a href="https://www.codecogs.com/eqnedit.php?latex=a&space;=&space;1&space;&plus;&space;N_{clicks}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a&space;=&space;1&space;&plus;&space;N_{clicks}" title="a = 1 + N_{clicks}" /></a>

<img src="https://latex.codecogs.com/gif.latex?b&space;=&space;1&space;&plus;&space;N_{nonclicks}" title="b = 1 + N_{nonclicks}" /></a>

```
# Distribution of sample A posterior
test.p_A_samples

# Distribution of sample B posterior
test.p_B_samples

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
```

## bayesian_t_test

Example:
```
bayesian_ttest = bayesian_t_test(sample_A=test.p_A_samples, 
                                 sample_B=test.p_A_samples, 
                                 v_minus_1=100,
                                 N_simulations=1000,
                                 pct_tune=50)
```
Argument definitions:
- ```sample_A```: array of values for sample A
- ```sample_B```: array of values for sample B
- ```v_minus_1```: degrees of freedom paramter
- ```N_simulations```: number of tuned samples to draws (see [PyMC3 documentation](https://docs.pymc.io/api/inference.html); default = 1000)
- ```pct_tune```: percentage of ```N_simulations``` to use for tuning (note: this number is added back to N_simulations; default = 50)

Attributes that can be returned:
- ```bayesian_t_test.summary```: summary from analysis

## parametric_t_test

Example:
```
# import dependencies
from bayesian_ab_test import parametric_t_test

# run test
t_test = parametric_t_test(sample_A=test.p_A_samples, 
                           sample_b=test.p_B_samples,
                           name_of_metric='Click-through rate')
```
Argument definitions:
- ```sample_A```: array of values for sample A
- ```sample_B```: array of values for sample B
- ```name_of_metric```: name of the metric for comparison (used for plotting)

Attributes that can be returned:
```
# t value from t-test
t_test.t_test_t

# p value from t-test
t_test.t_test_sig

# Cohens d effect size
t_test.cohens_d
```
<img src="https://latex.codecogs.com/gif.latex?d&space;=&space;\frac{M_{1}-M_{2}}{SD_{pooled}}" title="d = \frac{M_{1}-M_{2}}{SD_{pooled}}" /></a>

Where:

<img src="https://latex.codecogs.com/gif.latex?SD_{pooled}&space;=&space;\sqrt{\frac{SD_{1}^{2}&plus;SD_{2}^{2}}{2}}" title="SD_{pooled} = \sqrt{\frac{SD_{1}^{2}+SD_{2}^{2}}{2}}" /></a>

```
# Interpretation of effect size
t_test.size_of_effect
```
![alt text](https://www.polyu.edu.hk/mm/effectsizefaqs/formula/t1.jpg)
```
# Conclusion from t test in APA format
t_test.t_test_conclusion

# Bar plot comparing means
t_test.bar_plot
```

To install, use: ```pip install git+https://github.com/aaronengland/bayesian_ab_test.git```

Source: [Probabilistic Programming & Bayesian Methods for Hackers](http://camdavidsonpilon.github.io/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/)
