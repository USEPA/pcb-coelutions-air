# Use this script to run all coelutions for a specified model type

import os
import pandas as pd
#os.environ['R_HOME'] = "C:\\Program Files\\R\\R-4.3.1"

import sys
sys.path.append('..') # Add one directory back to the path

from process_coelution import coelution
import rpy2
from rpy2.robjects.packages import importr
#from rpy2.robjects import pandas2ri as p2r
#p2r.activate()
brms = importr('brms')
base = importr('base')



# model choices:
#model_type = 'detector_only' # MS vs. ECD detector
#model_type = 'pa_only' # passive vs. active sampling
#model_type = 'sample_only' # indoor vs. outdoor
model_type = 'phase_only' # particle vs. gas phase
#model_type = 'intercept_only' # No fixed effects

#model_type = 'full'

# run y-randomization
# If True, the model_selection_<model_type>.csv file must be available
randomize_y = True
if randomize_y:
    sig_coelutions = pd.read_csv('model_selection_%s.csv'%model_type, index_col=0)
    sig_coelutions = sig_coelutions[sig_coelutions[model_type] == '1']
    # split each value in sig_coelutions into a list and save all indexes of sig_coelutions
    coelutions = sig_coelutions.index.to_list()
    coelution_list = [s.split('_') for s in coelutions]
else:
    coelution_df = pd.read_csv('../data/full list of coelutions_TZ.csv')
    coelution_list = []
    for idx, row in coelution_df.iterrows():
        conj = row.dropna().astype(int)
        store_conj = ['PCB'+str(x) for x in conj]
        coelution_list.append(store_conj)
# Print first 5 pairs
print(coelution_list[:5])

model_store = {}
for i, PCBs in enumerate(coelution_list):
    #PCBs = ['PCB5', 'PCB8']
    PCB_name = '_'.join(PCBs)
    if PCB_name in model_store.keys():
        print('%s already done'%PCB_name)
        continue
    print('%s: %s out of %s' % (PCB_name, i+1, len(coelution_list)))
    PCB_sample = coelution(PCBs, model_type=model_type, randomize_y=randomize_y)
    if PCB_sample.X.empty:
        model_store[PCB_name] = 'no_data'
        continue
    PCB_sample.prep_data()
    if PCB_sample.data.empty:
        model_store[PCB_name] = 'no_data'
        continue
    PCB_sample.fit_coelution()

    # check diagnostics
    PCB_sample.check_diagnostics()

    if PCB_sample.pass_check:
        if ((PCB_sample.effects_significant) | (model_type == 'intercept_only')) & (PCB_sample.model_type == model_type):
            # Passed diagnostics and has significant fixed effects
            model_store[PCB_name] = PCB_sample.model_type
            PCB_sample.sample_posterior()
            PCB_sample.plot_comparison(save=True, use_legend=False)
            model_store[PCB_name] = 1
        else:
             model_store[PCB_name] = 0
        
    else:
        # One of them failed (doesn't matter which one)
        model_store[PCB_name] = 'model_failed'
model_series = pd.Series(model_store, name=model_type)
if randomize_y:
    model_series.to_csv('model_selection_%s_randomized.csv'%model_type)
else:
    model_series.to_csv('model_selection_%s.csv'%model_type)