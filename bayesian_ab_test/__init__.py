# bayesian ab test

# import dependencies
import pymc3 as pm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import ttest_ind
import numpy as np

# define function
def bayesian_ab_test(sample_a_total, sample_a_responses, sample_b_total, sample_b_responses, N_simulations=1000, pct_tune=50, gr_threshold=1, N_additional_draws=1000):
    ###########################################################################
    # get parameters for model
    # make pct_tune into a proportion
    prop_tune = pct_tune/100
    # calculate number to tune
    N_tune = round(N_simulations*prop_tune)
    # calculate additional tuning steps
    N_additional_tune = round(N_additional_draws*prop_tune)

    ###########################################################################
    # get data for lower and upper plausible values

    # define helper function for plausible values
    def plausible_values(total, upvotes):
      # get downvotes
      d = total - upvotes
      # 1 + upvotes
      a = 1 + upvotes
      # 1 + downvotes
      b = 1+ d
      # calculate lower plausible value
      lpv = (a/(a+b)) - (1.65 * np.sqrt((a*b)/(((a+b)**2)*(a+b+1))))
      # calculate upper plausible value
      upv = (a/(a+b)) + (1.65 * np.sqrt((a*b)/(((a+b)**2)*(a+b+1))))
      # return lpv and upv
      return lpv, upv
    
    # sample A
    # number of non-clicks
    N_nonclicks_A = sample_a_total - sample_a_responses
    # calculate CTR
    observed_p_A = sample_a_responses/sample_a_total
    # calculate lower plausible value in sample A
    sample_A_lpv = plausible_values(total=sample_a_total, upvotes=sample_a_responses)[0]
    # calculate upper plausible value in sample A
    sample_A_upv = plausible_values(total=sample_a_total, upvotes=sample_a_responses)[1]

    # sample B
    # calculate non-clicks
    N_nonclicks_B = sample_b_total - sample_b_responses
    # calculate CTR
    observed_p_B = sample_b_responses/sample_b_total
    # calculate lower plausible value in sample B
    sample_B_lpv = plausible_values(total=sample_b_total, upvotes=sample_b_responses)[0]
    # calculate upper plausible value in sample B
    sample_B_upv = plausible_values(total=sample_b_total, upvotes=sample_b_responses)[1]

    # put into df
    df = pd.DataFrame({'Variable': ['Sent','Clicked','Non-Clicked','CTR','LPV','UPV'],
                       'Sample A': [sample_a_total, sample_a_responses, N_nonclicks_A, observed_p_A, sample_A_lpv, sample_A_upv],
                       'Sample B': [sample_b_total, sample_b_responses, N_nonclicks_B, observed_p_B, sample_B_lpv, sample_B_upv]})
    # create a col that is sample A minus sample B
    df['A - B'] = df['Sample A'] - df['Sample B']

    ###########################################################################
    # plot it
    x = ('Sample A', 'Sample B')
    y = (observed_p_A, observed_p_B)
    # get error values
    # lower
    # sample A
    A_low_err = observed_p_A - sample_A_lpv
    # sample B
    B_low_err = observed_p_B - sample_B_lpv
    # upper
    # sample A
    A_upp_err = sample_A_upv - observed_p_A
    # sample B
    B_upp_err = sample_B_upv - observed_p_B
    # create a programmatic title
    # print message
    if sample_A_lpv > sample_B_lpv:
      title = 'Sample A has a greater LPV'
    else:
      title = 'Sample B has a greater LPV'
    yerr = np.array([(A_low_err, B_low_err), (A_upp_err, B_upp_err)])
    lpv_plot, axes = plt.subplots()
    axes.errorbar(x, y, yerr, fmt='o')
    axes.set_title(title)
    
    ###########################################################################
    # place the user input into artificial observations
    # sample A
    # create list for number of zeros (non-clicks)
    observations_A = [0]*N_nonclicks_A
    # create list for number of 1s
    N_clicks_list_A = [1]*sample_a_responses
    # combine lists
    observations_A.extend(N_clicks_list_A)
    
    # sample B
    # create list for number of zeros
    observations_B = [0]*N_nonclicks_B
    # create list for number of 1s
    N_clicks_list_B = [1]*sample_b_responses
    # combine lists
    observations_B.extend(N_clicks_list_B)
    
    ###########################################################################
    # set up pymc3 model assuming uniform prior and Bernoulli posterior
    # print a message
    print('\n')
    print('Model being built using {} initial, tuned draws...'.format(N_simulations))
    print('\n')
    # instantiate model
    with pm.Model() as model:
      # get prior probabilities from Uniform distribution because we don't know what they are (objective prior)
      prior_A = pm.Uniform('prior_A', 0, 1)
      prior_B = pm.Uniform('prior_B', 0, 1)
      # fit our observations to a (posterior) Bernoulli distribution, could also try Binomial?
      posterior_A = pm.Bernoulli('posterior_A', prior_A, observed=observations_A)
      posterior_B = pm.Bernoulli('posterior_B', prior_B, observed=observations_B)
      # get samples from the posterior distribution
      trace = pm.sample(draws=N_simulations+N_tune, tune=N_tune)
    # get maximum value of Gelman-Rubin test
    max_gr = max(np.max(gr_stats) for gr_stats in pm.gelman_rubin(trace).values()) 
    # if model has not converged, continue to...
    while max_gr > gr_threshold:
      # print message
      print('\n')
      print('Gelman-Rubin statistic: {}'.format(max_gr))
      print('Gelman-Rubin statistic is too large, {} additional draws will be taken.'.format(N_additional_draws))
      print('\n')
      with model:
        trace = pm.sample(draws=N_additional_draws+N_additional_tune, tune=N_additional_tune)
      # get maximum value of Gelman-Rubin test
      max_gr = max(np.max(gr_stats) for gr_stats in pm.gelman_rubin(trace).values())
      # add N_additional_draws to N_simulations
      N_simulations += N_additional_draws
    # print message
    print('\n')
    print('Success! Model has converged after {} draws. Final Gelman-Rubin: {}'.format(N_simulations, max_gr))
    
    ###########################################################################
    # display conversion stats
    # calculate the bayesian fraction of missing information
    bfmi = pm.bfmi(trace)
    # get maximum value of Gelman-Rubin test
    max_gr = max(np.max(gr_stats) for gr_stats in pm.gelman_rubin(trace).values())
    # print message
    print('Bayesian fraction of missing information: {}'.format(bfmi))
    print('\n')

    ###########################################################################
    # get distributions
    # sample A
    p_A_samples = trace['prior_A']
    # sample B
    p_B_samples = trace['prior_B']
    
    ###########################################################################
    # plot the distributions
    sns.set(style='white', palette='muted', color_codes=True)
    # Set up the matplotlib figure
    dist_plot, axes = plt.subplots(figsize=(7, 7), sharex=True)
    dist_plot.suptitle('Posterior distributions of $p_A$ (blue) and $p_B$ (red) after {} draws'.format(N_simulations))
    sns.despine(left=True)
    # posterior A
    p1 = sns.distplot(p_A_samples, color='b', label='Posterior of $p_A$')
    p1.vlines(sample_A_lpv, 0, 30, colors='b', linestyle='--', label='Sample A LPV: {0:0.3f}'.format(sample_A_lpv))
    p1.legend(loc='upper left')
    # posterior B
    p2 = sns.distplot(p_B_samples, color='r', label='Posterior of $p_B$')
    p2.vlines(sample_B_lpv, 0, 30, colors='r', linestyle='--', label='Sample B LPV: {0:0.3f}'.format(sample_B_lpv))
    p2.legend(loc='upper left')
    # display plot
    plt.tight_layout() # fix any overlapping
    
    ###########################################################################
    # get proportion of p_A_samples that are greater than p_B_samples
    # iterate through p_A_samples and p_B_samples
    sum_of_A_greater_than_B = 0
    for i in range(len(p_A_samples)):
        if p_A_samples[i] > p_B_samples[i]:
            sum_of_A_greater_than_B += 1
    
    # calculate proportion A greater than B    
    proportion_A_greater_than_B = sum_of_A_greater_than_B/len(p_A_samples)
    # calculate proportion B greater than A
    proportion_B_greater_than_A = 1-proportion_A_greater_than_B

    ###########################################################################
    # hypothesis testing
    # independent t-test
    t_test_t = ttest_ind(p_A_samples, p_B_samples)[0]
    t_test_sig = ttest_ind(p_A_samples, p_B_samples)[1]
    # effect size
    sd_pooled = np.sqrt(((np.std(p_A_samples)**2)+(np.std(p_B_samples)**2))/2)
    cohens_d = abs((np.mean(p_A_samples) - np.mean(p_B_samples))/sd_pooled)
    # size of effect size
    if cohens_d < .2:
      size_of_effect = 'trivial'
    elif cohens_d < .5:
      size_of_effect = 'small'
    elif cohens_d < .8:
      size_of_effect = 'moderate'
    elif cohens_d < 1.3:
      size_of_effect = 'large'
    else:
      size_of_effect = 'very large'
    
    # print message
    if t_test_sig < .05:
        t_test_conclusion = 'After {0} draws, there was a significant difference between sample A (mean = {1:0.3f}) and sample B (mean = {2:0.3f}) and the effect size was {3}, diff = {4:0.3f}, t = {5:0.3f}, p = {6:0.3f}, d = {7:0.3f})'.format(N_simulations,
                                                      np.mean(p_A_samples),
                                                      np.mean(p_B_samples),
                                                      size_of_effect,
                                                      abs(np.mean(p_A_samples) - np.mean(p_B_samples)),
                                                      t_test_t,
                                                      t_test_sig,
                                                      cohens_d)
    else:
        t_test_conclusion = 'After {0} draws, there was not a significant difference between sample A (mean = {1:0.3f}) and sample B (mean = {2:0.3f}) and the effect size was {3}, diff = {4:0.3f}, t = {5:0.3f}, p = {6:0.3f}, d = {7:0.3f})'.format(N_simulations,
                                                     np.mean(p_A_samples),
                                                     np.mean(p_B_samples),
                                                     size_of_effect,
                                                     abs(np.mean(p_A_samples) - np.mean(p_B_samples)),
                                                     t_test_t,
                                                     t_test_sig,
                                                     cohens_d)
    
    ###########################################################################
    # create bar plot
    bar_plot, axes = plt.subplots()
    axes.bar(['Sample A', 'Sample B'], [np.mean(p_A_samples), np.mean(p_B_samples)], yerr=[np.std(p_A_samples), np.std(p_B_samples)], alpha=.5, capsize=10, color=('b','r'))
    axes.set_title('Mean click probability by sample after {} draws'.format(N_simulations))
    axes.set_ylabel('Probability of click')
    
    ###########################################################################
    # put all of the objects we want inside of a class so they can be returned
    class Attributes:
        def __init__(self, df, lpv_plot, trace, bfmi, max_gr, dist_plot, proportion_A_greater_than_B, proportion_B_greater_than_A, t_test_t, t_test_sig, cohens_d, size_of_effect, t_test_conclusion, bar_plot):
            self.df = df
            self.lpv_plot = lpv_plot
            self.trace = trace
            self.bfmi = bfmi
            self.max_gr = max_gr
            self.dist_plot = dist_plot
            self.proportion_A_greater_than_B = proportion_A_greater_than_B
            self.proportion_B_greater_than_A = proportion_B_greater_than_A
            self.t_test_t = t_test_t
            self.t_test_sig = t_test_sig
            self.cohens_d = cohens_d
            self.size_of_effect = size_of_effect
            self.t_test_conclusion = t_test_conclusion
            self.bar_plot = bar_plot
    x = Attributes(df, lpv_plot, trace, bfmi, max_gr, dist_plot, proportion_A_greater_than_B, proportion_B_greater_than_A, t_test_t, t_test_sig, cohens_d, size_of_effect, t_test_conclusion, bar_plot)
    return x

