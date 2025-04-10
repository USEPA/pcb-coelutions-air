# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 11:16:25 2023

@author: Todd Zurlinden
"""
import re
import numpy as np
import scipy.stats as scs
import statsmodels.api as sm
import sklearn.metrics as skm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import arviz as az

import rpy2
from rpy2.robjects import r
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri as p2r
from rpy2.robjects import numpy2ri as n2r
import rpy2.robjects.packages as rpack
from rpy2.robjects.packages import importr

def check_site_data(df, column_name, valid_values, PCBs):
    invalid_values = df[~df[column_name].isin(valid_values)][column_name]
    if not invalid_values.empty:
        warning_message = f"Warning: {','.join(PCBs)} column '{column_name}' contains invalid values: {invalid_values.unique()}"
        warnings.warn(warning_message, UserWarning)

def check_empty_df(df, PCBs):
    if df.empty:
        warning_message = f"Warning: No data available for the {'_'.join(PCBs)} coelution combination"
        warnings.warn(warning_message, UserWarning)

def check_for_gas(df, column_name, value_to_check):
    if value_to_check not in df[column_name].values:
        raise ValueError(f"The value '{value_to_check}' is not contained in '{column_name}'. Rerun compare_IADN(phase='<available_phase>')")

p2r.activate()
n2r.activate()

base = importr('base')
brms = importr('brms')
tidybayes = importr('tidybayes')
rstan = importr('rstan')
posterior = importr('posterior')

class coelution:
    def __init__(self, PCBs, data_fpath=['../data/air_coelutions data extraction 4.20.csv', '../data/air_coelutions data extraction_2024.csv'
                                         ], 
                 model_type = 'full', fit_phi='var',
                 replace_list = [0, '0', 'x', 'd', '?0.5', '0.1<', '<1.0', 'nd', 'Nd', '<0.40', '<0.51', "\xa0n.d.", 'n.d.', 'ND', ' ND', 'NS', '<0.007', '<0.001', '<0.20', '/', '-', '--', '---', '<0.005', 'deleted "ND"', 'LT+', '<0.41', '<2', '<LOQ', '<LOD', 'BDL', '<loq', 'NDb', '<1.5', 'nd?(0.08)'],
                 feature_list = ['HERO ID', 'sample name (reported in study)', 'sample_type', 'phase', 'detector', 'pa_sampling', 'units', 'individual sample or summary data (e.g., individual sample, sample average, site average, site-time average, study average)',
                                 'sample location', 'number of observations', 'notes',
                                 ], 
                 draw_type = '.prediction', nested = False, hero_map = '../data/hero_reference_list.csv', randomize_y=False,
                 loo_comparison=False, aroclor=False, default_LoD=0.000001, site_level_only=False):        
        
        self.PCBs = PCBs # List of PCBs for coelution
        self.data_fpath = data_fpath
        new_lod = [f'<{0.1*i:.1f}' for i in range(1,10)]
        new_lod2 = [f'<{i:.0f}' for i in range(1,10)]
        self.replace_list = replace_list + new_lod + new_lod2
        self.feature_list = feature_list
        self.loo_comparison = loo_comparison
        self.fit_phi = fit_phi
        self.aroclor = aroclor
        self.draw_type = draw_type
        self.nested = nested
        self.default_LoD = default_LoD
        self.site_level_only = site_level_only
        self.randomize_y = randomize_y
        if hero_map is not None:
            self.hero_map = pd.read_csv(hero_map)
            self.hero_map['hero_id'] = self.hero_map['hero_id'].astype(str)
            # joint the 'author_first' column with the 'year' column with a space in between and label 'short'
            self.hero_map['short'] = self.hero_map['author_first'].astype(str) + ' ' + self.hero_map['year'].astype(str)
        self._assign_effects(model_type)
        # Clean up raw data for the PCBs right away
        self.X, self.features_df = self._import_data()
        check_empty_df(self.X, self.PCBs)
        
        # Check if all values of a congener are LoD and override to 'intercept_only' if they are
        if self._all_LoD(self.X).any() and self.model_type != 'intercept_only':
            warning_message = f"Warning: Data for {'_'.join(PCBs)} are all 'LoD' for one congener. Changing model from {model_type} to intercept_only"
            warnings.warn(warning_message, UserWarning)
            model_type = 'intercept_only'
            self._assign_effects(model_type)

    # Private functions
    def _assign_effects(self, model_type):
        self.model_type = model_type
        if model_type == 'full':
            self.fixed_effects = ['sample_type', 'phase', 'detector']
            self.orig_effects = ['sample_type', 'phase', 'detector']
        elif model_type == 'sample_phase':
            self.fixed_effects = ['sample_type', 'phase']
            self.orig_effects = ['sample_type', 'phase']
        elif model_type == 'sample_detector':
            self.fixed_effects = ['sample_type', 'detector']
            self.orig_effects = ['sample_type', 'detector']
        elif model_type == 'sample_only':
            self.fixed_effects = ['sample_type']
            self.orig_effects = ['sample_type']
        elif model_type == 'detector_only':
            self.fixed_effects = ['detector']
            self.orig_effects = ['detector']
        elif model_type == 'phase_only':
            self.fixed_effects = ['phase']
            self.orig_effects = ['phase']
        elif model_type == 'pa_only':
            self.fixed_effects = ['pa_sampling']
            self.orig_effects = ['pa_sampling']
        elif model_type == 'intercept_only':
            self.fixed_effects = []
            self.orig_effects = []
    def _assign_model(self, fixed_effects):
        if not fixed_effects:
            self.model_type = 'intercept_only'
        elif fixed_effects == ['sample_type', 'phase', 'detector']:
            self.model_type = 'full'
        elif fixed_effects == ['sample_type', 'detector']:
            self.model_type = 'sample_detector'
        elif fixed_effects == ['sample_type', 'phase']:
            self.model_type = 'sample_phase'
        elif fixed_effects == ['sample_type']:
            self.model_type = 'sample_only'
        elif fixed_effects == ['detector']:
            self.model_type = 'detector_only'
        elif fixed_effects == ['phase']:
            self.model_type = 'phase_only'
        elif fixed_effects == ['pa_sampling']:
            self.model_type = 'pa_only'
        else:
            raise ValueError(f"'{fixed_effects}' does not have a model type")

    def _keep_site(self, full_df, hero_id, site_info):
        df = full_df.copy()
        if hero_id in df.HERO_ID.values:
            drop_idx = df[(df.HERO_ID==hero_id) & (df.site_info != site_info)].index
            df.drop(drop_idx, inplace=True)
        return df
    
    def _calc_individual(self, full_df, hero_id):
        df = full_df.copy()
    
    def _clean_up_data(self, pcb_df):
        tmp_df = pcb_df.copy()

        # Handle studies where we only keep one label
        tmp_df = self._keep_site(tmp_df, 156096, 'site/season/year-level average')
        tmp_df = self._keep_site(tmp_df, 2151511, 'site/date-level')
        tmp_df = self._keep_site(tmp_df, 2158252, 'time-average')
        tmp_df = self._keep_site(tmp_df, 2180083, 'date-level average')
        tmp_df = self._keep_site(tmp_df, 2180083, 'date-level average')
        tmp_df = self._keep_site(tmp_df, 136825, 'site-level average')
        
        # Handle the studies where we relabel as site-average 
        hero_ids = [156096, 1597570, 198175, 2023480, 2150623, 2151511, 2155075, 
                    2158252, 2180083, 3023532, 3355481, 3869165, 458898, 5016961, 
                    5431796, 5880710, 5882141, 5925969, 6956497, 6956819, 6956865,
                    2153443, 2155391, 2150915, 5016959, 5880535]
        particle_ids = [2157879, 2162454, 2175094, 2930584, 3347444, 2150287] # TODO: Keep them separate for now until we handle gas vs. particles
        tmp_df.loc[tmp_df.HERO_ID.isin(hero_ids+particle_ids), 'site_info'] = 'site-level average'
        
        # Handle the studies where we relabel as study average
        study_hero_ids = [2154069, 2158746, 2159383, 2161175, 2164050, 2166893,
                          2181948, 5080467, 5880746, 6956489, 6956564]
        tmp_df.loc[tmp_df.HERO_ID.isin(study_hero_ids), 'site_info'] = 'study average'
        
        # Rename any PCBs that are site average instead of site-level average
        tmp_df['site_info'] = tmp_df['site_info'].replace('site average', 'site-level average')
        
        # Handle the weights
        weight_dict = {
            1058259: 37*4, # Thirty seven sites in Chicago, sampled over four seasons
            198165: 3*15, # Average over "15–20 sampling events at each of the three sites conducted over the period from August 2004 to September 2005."
            2169595: 40*4, # Ambient air samples were collected during four seasons (summer, fall, winter, and spring) at forty different sites
            3605143: 22*4, # Atmospheric concentrations of PAHs and PCBs were detected in Istanbul during four different sampling periods from September to December 2014 at 22 sampling sites
            4217246: 35, # PCB contamination in the indoor air was measured in 35 office rooms that were frequented by the participants
            198253: 97, # average over 97 different sites, collected summer/fall (mid-July to mid-October)
            6956489: 10, # A total of ten (10) PAS were deployed at different locations
            6394036: 33, # We sampled eleven sampling sites in three zones
            3867953: 12*23, # 12 different sampling periods from February 2015 to February 2016 at 23 sampling site
            2149665: 10, # Ten (10) sampling locations were selected for deploying Polyurethane Foam Passive Air Samplers (PUF-PAS)
            }

        tmp_df['N'] = tmp_df['HERO_ID'].map(weight_dict).fillna(tmp_df['N'])
        tmp_df['N'] = tmp_df['N'].fillna(1)
        tmp_df.loc[tmp_df.site_info == 'site-level average', 'N'] = 1 # Set all site-level avergaes to N=1
        return tmp_df.reset_index(drop=True)
    
    def _import_data(self):
        df = pd.concat([pd.read_csv(x, encoding='ISO-8859-1',low_memory=False) for x in self.data_fpath]).reset_index(drop=True) # Changed 0.0.74 --> 0.074 in row 540, PCB52 of original 
        self.pre_df = df.copy()
        if self.aroclor:
            df = df[df['sample location'].isin(['Aroclor'])] # Remove Arochlor data
            df = df.replace('\xa0','', regex=True)
            df = df[df['HERO ID'] == '647268']
            self.df = df
        else:
            df = df[df['sample_type'].isin(['outdoor', 'indoor'])] # Filter to indoor/outdoor
            df = df[~df['HERO ID'].isin(['IADN', '5880847', '2159695', '5431796'])] # Remove IADN data and 5880847 (report colorants), 2159695 (exclusion email) and 5431796 (exclusion email)
            df = df[~df['sample location'].isin(['Aroclor'])] # Remove Arochlor data
            df = df[~df['individual sample or summary data (e.g., individual sample, sample average, site average, site-time average, study average)'].isin(['raw extraction'])]
            # Handle not-reported covariates for specific model
            for effect in ['detector', 'phase', 'pa_sampling', 'sample_type']:
                if effect in self.fixed_effects: 
                    df = df[~df[effect].isin(['NR', 'Other', 'unclear', 'N/A', np.nan])] # Remove and NR rows
    
            if 'detector' in self.fixed_effects:
                df['detector'] = df['detector'].replace(['HRMS', 'MS/MS', 'MS-MS'], 'MS')
            
            if 'pa_sampling' in self.fixed_effects:
                df['pa_sampling'] = df['pa_sampling'].replace(['passive '], 'passive')

            if 'phase' in self.fixed_effects:
                df['phase'] = df['phase'].replace(['gas '], 'gas')
                df['phase'] = df['phase'].replace(['particle+gas', 'aerosol', 'gas+aerosol', 'gas+particles', 'particles'], 'particle')
            
        self.init_df = df.copy() # Initial df available for modeling
        
        PCB_list = [x for x in df.columns if 'PCB' in x] # Get the PCB columns from df
        self.PCB_list = PCB_list
        pcb_df = df.loc[:, self.feature_list+PCB_list].copy()
        self.test_pcb_df = pcb_df.copy()

        pcb_df = pcb_df.infer_objects(copy=False).replace('\x02','', regex=True) # Clean up the spaces which are coded as \x02
        pcb_df = pcb_df.replace('\xa0','', regex=True) # Clean up the spaces which are coded as \xa2
        pcb_df = pcb_df.replace(self.replace_list, 'LoD') # Make sure LoD is used for measured values below LoD which is different than NaN (not measured at all)
        
        rename_dict = {'HERO ID': 'HERO_ID',
                        'sample location': 'sample_location',
                        'individual sample or summary data (e.g., individual sample, sample average, site average, site-time average, study average)': 'site_info',
                        'sample name (reported in study)': 'sample_name',
                        'number of observations': 'N'}
        pcb_df = pcb_df.rename(columns=rename_dict) # Clean up columns
        self.feature_list = [rename_dict.get(item, item) for item in self.feature_list] # Rename the features
        X = pcb_df[self.PCBs].copy()

        self.missing_PCBs = X.columns[X.isna().any(axis=0)]
        #X.dropna(how='all', axis=1, inplace=True)
        X.dropna(how='any', axis=0, inplace=True)
        #X.drop_duplicates(keep=True, inplace=True) # TODO CHECK THE DUPLICATION ERROR
        X = X[~(X == 'LoD').all(axis=1)] # Remove rows that are all 'LoD'
        features_df = pcb_df.loc[X.index, self.feature_list].copy()
        
        features_df['HERO_ID'] = features_df['HERO_ID'].astype(int)
        return X, features_df
   
    def _R_code(self):
        all_pcbs = ', '.join(self.PCBs)
        ref_pcb = self.PCBs[0]
        if self.handle_zeros == 'drop_ND':
            nd_flag = 'drop_ND_'
        else:
            nd_flag = ''

        if self.randomize_y:
            yrand_flag = 'yrand_'
        else:
            yrand_flag = ''

        fsave = yrand_flag + nd_flag + 'model_'+'_'.join(self.PCBs)
        fixed_effect_str = ' + '.join(self.fixed_effects)
        
        if not fixed_effect_str:
            fixed_effect_str = '1'
            prior_str = "priors <- c(set_prior('normal(0, 0.5)', class = 'sd', dpar='phi', lb=0))"
        else:
            base_str = "priors <- c(set_prior('normal(0, 0.5)', class = 'sd', dpar='phi', lb=0), set_prior('normal(0, 3)', class = 'b', dpar='phi'), "
            prior_str = base_str + ', '.join([f"set_prior('normal(0, 3)', class = 'b', dpar='mu{x}')" for x in self.PCBs[1:]]) + ')'
        
        
        if self.random_effects:
            if self.nested:
                #rnd_effect_str = '+ (%s|HERO_ID/site_info)' % fixed_effect_str
                rnd_effect_str = '+ (1|HERO_ID/site)' 
            else:
                rnd_effect_str = '+ (%s|HERO_ID)' % fixed_effect_str
                #rnd_effect_str = '+ (1|HERO_ID)'
            #rnd_effect_str = '+ (sample_type|HERO_ID)'
        else:
            rnd_effect_str = ''
            prior_str = "priors <- NULL"
        
        if self.fit_phi:
            phi_str = 'phi ~ ' + fixed_effect_str + rnd_effect_str
            #phi_str = 'phi ~ ' + fixed_effect_str + '+ (%s|HERO_ID)' % fixed_effect_str
            #phi_str = 'phi ~ ' + fixed_effect_str + ','
            
            phi_save = ''
        else:
            phi_str = ''
            phi_save = '_const_phi'
        # phi ~ TREAT
        # bind(%s) ~ sample_type + (1|HERO_ID),
        # brmsformula(bind(%s)|weights(N) ~ %s %s, %s family = dirichlet(refcat='%s')),
        # brmsformula(bind({all_pcbs})|weights(N) ~ {fixed_effect_str} {rnd_effect_str}, {phi_str}, family = dirichlet(refcat='{ref_pcb}')),
        # brmsformula({ref_pcb} | weights(N) ~ {fixed_effect_str} {rnd_effect_str}, {phi_str}, family=Beta()),
        # formula <- bf(bind({all_pcbs}) ~ {fixed_effect_str} {rnd_effect_str}, {phi_str}) # WORKING
        # 
        #backend='cmdstanr',
        R_str = f"""
        library(brms)
        library(tidybayes)
        library(dplyr)
        library(tidyr)
        bind <- function(...) cbind(...)
        
        run_brms <- function(df, comp) {{
            formula <- bf(bind({all_pcbs}) ~ {fixed_effect_str} {rnd_effect_str}, {phi_str})
            {prior_str}
            mod <- brm(
                formula,
                family = dirichlet(refcat='{ref_pcb}'),
                data = df, backend='cmdstanr', prior = priors,
                chains=4, cores=4, iter = 2000, control = list(adapt_delta = 0.99), file = "../brms_models/{self.model_type}{phi_save}/{fsave}"
                
            )
            if (comp == T){{
                mod <- add_criterion(mod, 'loo')
            }}
            return(mod)
        }}
        
        
        sample_ppc_pop2 <- function(mod) {{
            res <- epred_draws(mod, newdata=tibble(sample_type = c('indoor', 'outdoor')), re_formula = NA)
            return(res)
            }}
        
        sample_ppc <- function(mod, df, draw_type, study=F) {{

            input_list <- lapply(df, unique)
            new_data_oos <- do.call(expand_grid, input_list)
            #print(new_data_oos)

            if (study == T) {{
                if (draw_type == '.epred') {{
                    res <- epred_draws(mod, newdata=new_data_oos)
                }}
                else {{
                    res <- predicted_draws(mod, newdata=new_data_oos)
                }}
            }} else {{
                if (draw_type == '.epred') {{
                    res <- epred_draws(mod, newdata=new_data_oos, re_formula = NA)
                }} else {{
                    res <- predicted_draws(mod, newdata=new_data_oos, re_formula = NA)
                }}
            }}
            
            return(as.data.frame(res))
            }}
        
        sample_ppc_new_study <- function(mod, df, draw_type) {{
            input_list <- lapply(df, unique)
            new_data_oos <- do.call(expand_grid, input_list)
            if (draw_type == '.epred') {{
                res <- epred_draws(mod, newdata=new_data_oos, re_formula=NULL, allow_new_levels = TRUE, sample_new_levels = "uncertainty")
            }} else {{
                res <- predicted_draws(mod, newdata=new_data_oos, re_formula=NULL, allow_new_levels = TRUE, sample_new_levels = "uncertainty")
            }}
        }}
            
        sample_ppc_study <- function(mod, studies) {{
            res <- epred_draws(mod, newdata=expand_grid(sample_type = c('indoor', 'outdoor'), HERO_ID=studies))
            return(res)
            }}
        
        ppc_summary <- function(mod, df) {{
            input_list <- lapply(df, unique)
            res <- as.data.frame(predict(mod, newdata=do.call(expand_grid, input_list), re_formula = NA))
            return(res)
            }}
        
        fixed_params <- function(mod) {{
            return(as.data.frame(fixef(mod)))
        }}
            
        return_sig_params <- function(mod) {{
            fixed_effects <- as.data.frame(fixef(mod))
            sig_effects = fixed_effects[(fixed_effects[,3]>0 & fixed_effects[,4]>0) | (fixed_effects[,3]<0 & fixed_effects[,4]<0), ]
            return(sig_effects)
            
            }}
        
        """
        #print(R_str)
        #saf
        self.R_str = R_str
        self.run_R = rpack.STAP(R_str, "run_R")
    
    def compile_R_code(self):
        self._R_code()
    def _are_all_effects_significant(self, sig_brms_effects):
        self.mu_sig_effects = [value_a for value_a in self.fixed_effects if any(re.compile(f'(?=.*mu.*{value_a}).*').match(value_b) for value_b in sig_brms_effects)]
        self.phi_sig_effects = [value_a for value_a in self.fixed_effects if any(re.compile(f'(?=.*phi.*{value_a}).*').match(value_b) for value_b in sig_brms_effects)]
        
        #self.effects_significant = ((len(self.mu_sig_effects) == len(self.fixed_effects)) | (len(self.phi_sig_effects) == len(self.fixed_effects))) # Check sig. effect on mean and precision
        self.effects_significant = (len(self.mu_sig_effects) == len(self.fixed_effects)) # Only check if there's an effect on the mean proportion
        
    def _assign_sig_effects(self):
        """Assign the significant effects to the class"""
        sig_brms_effects = p2r.rpy2py(self.run_R.return_sig_params(self.mod))
        self._are_all_effects_significant(list(sig_brms_effects.index))
    
    def check_diagnostics(self):
        pass_check = True
        n_div = np.sum(n2r.rpy2py(rstan.get_divergent_iterations(self.mod.rx2('fit'))))
        if n_div > 0:
            pass_check = False
            warnings.warn(f"Warning: Divergences present ({n_div})", UserWarning)
        
        max_tree_depth =  rstan.get_num_max_treedepth(self.mod.rx2('fit'))[0]
        if max_tree_depth > 10:
            pass_check = True # Pass as long as divergences passes
            warnings.warn(f"Warning: Max tree depth > 10 ({max_tree_depth})", UserWarning)
        
        rhat_array = posterior.rhat(self.mod)
        if any(rhat_array > 1.05):
            pass_check = False
            warnings.warn(f"Warning: rhat > 1.05 ({max(rhat_array)})", UserWarning)
        
        summary = p2r.rpy2py(base.summary(self.mod).rx2('fixed'))
        if any(summary['Bulk_ESS'].values < 400):
            pass_check = False
            warnings.warn(f"Warning: ESS less than 100 per chain ({min(summary['Bulk_ESS'].values)})", UserWarning)
        self.pass_check = pass_check
    def _all_LoD(self, df):
        a = df.values
        return ('LoD' == a).all(0)
    
    def _check_proportion(self, df, groupby_columns, prop_filter, target_value='LoD'):
        grouped_data = df.groupby(groupby_columns)[self.PCBs].apply(lambda x: (x == target_value).mean())
        self.grouped_data = grouped_data
        grouped_prop = grouped_data > prop_filter
        return (grouped_data > prop_filter).any().any(), list(grouped_prop.index[grouped_prop.any(axis=1)])
    def prep_data(self, handle_zeros = 'study_consistent', clean_up = True, prop_filter=0.8, set_max_prop=True):
        self.handle_zeros = handle_zeros
        X = self.X.copy()
        features_df = self.features_df.loc[X.index].copy()
        full_df = pd.concat([features_df, X], axis=1)

        self.full_df_pre = full_df.copy()
        # Handle studies that are individual-level (repliactes)
        fix_indiv_hero_ids = [84544, 3984192, 5881108, 5881797, 2152167, 2155065] # HERO IDs that are individual data
        extra_indiv_hero_ids = [2159252, 2154384, 2150880, 198177] # HERO IDs from additional extraciton
        indiv_hero_ids = fix_indiv_hero_ids  + extra_indiv_hero_ids
        indiv_df = full_df[full_df.HERO_ID.isin(indiv_hero_ids)].copy()
        indiv_df[indiv_df=='LoD'] = 0
        indiv_df[self.PCBs] = indiv_df[self.PCBs].astype(float)
        # groupby on 'sample location'
        #grouped_df = df.groupby(self.feature_list)[self.PCBs].mean().reset_index()
        grouped_df = indiv_df.groupby(['HERO_ID', 'sample_type', 'phase', 'detector', 'pa_sampling', 'sample_location'])[self.PCBs].mean().reset_index().copy()

        grouped_df['site_info'] = 'site-level average' # These data are now site-level averages
        grouped_df[self.PCBs] = grouped_df[self.PCBs].astype(str)

        grouped_df[grouped_df == '0.0'] = 'LoD'
        full_df = full_df[~full_df.HERO_ID.isin(indiv_hero_ids)]
        full_df = pd.concat([full_df, grouped_df])

        # Number of measurements that have a No detect
        full_df['LoD_count'] = full_df.eq('LoD').sum(axis=1)

        # Create mapper conc based on lowest reported concentration
        df_map = full_df[['HERO_ID'] + self.PCBs].apply(pd.to_numeric, errors='coerce').groupby('HERO_ID')[self.PCBs].min()/np.sqrt(2)
        self.df_map_init = df_map.copy()
        #replace_LoD = df_map.median().astype(float) # Use the median of the pseudo-LoDs to fill in all the rest
        #self.replace_LoD = replace_LoD.copy()
        #replace_LoD[replace_LoD > self.default_LoD] = self.default_LoD
        #df_map = df_map.fillna(replace_LoD).astype(float)
        df_map = df_map.fillna(self.default_LoD).astype(float)

        self.df_map = df_map.copy()
        self.full_df_post = full_df.copy()

        if handle_zeros == 'global_percentage':
            self.raw_ND_count = full_df['LoD_count'].sum()
            self.raw_frac_ND = full_df['LoD_count'].mean()
            self.raw_perc_ND = self.raw_frac_ND*100
            if self.raw_frac_ND < 0.4 and not self.aroclor:
                # If fewer than 40% are non-detects, drop the rows that have NDs
                # Don't drop aroclor non-detects
                self.fitted_ND_count = 0
                self.fitted_perc_ND = 'Drop < 40%'
                full_df.replace('LoD', np.nan, inplace=True)
                full_df.dropna(how='any', axis=0, inplace=True, subset=self.PCBs)
            else:
                self.fitted_ND_count = self.raw_ND_count
                self.fitted_perc_ND = self.raw_perc_ND
                
                # Only need to do covariate ND check if keeping NDs
                if self.model_type != 'intercept_only':
                    LoD_flag, covariates = self._check_proportion(pd.concat([self.features_df, X], axis=1), self.fixed_effects, prop_filter)
                    if LoD_flag:
                        warning_message = f"Warning: Data for {'_'.join(self.PCBs)} are > 80% 'LoD' for {','.join(covariates)}. Changing model from {self.model_type} to intercept_only"
                        warnings.warn(warning_message, UserWarning)
                        model_type = 'intercept_only'
                        self._assign_effects(model_type)
                full_df.replace('LoD', self.default_LoD, inplace=True)

        elif handle_zeros == 'study_percentage':
            self.raw_ND_count = full_df['LoD_count'].sum()
            self.raw_frac_ND = full_df['LoD_count'].mean()
            self.raw_perc_ND = self.raw_frac_ND*100
            group_cols = ['HERO_ID'] + self.fixed_effects
            
            frac_missing = full_df.groupby(group_cols)['LoD_count'].mean()
            full_df_group = full_df.set_index(group_cols)
            frac_idx = frac_missing[frac_missing < 0.4].index
            # Drop the LoD entries that are part of the study/covariate that has < 40% LoD
            full_df_group = full_df_group.loc[~(full_df_group.index.isin(frac_idx) & full_df_group.LoD_count.isin([1]))]
            full_df = full_df_group.reset_index()
            full_df.replace('LoD', self.default_LoD, inplace=True) # Replace remaining LoD with low value
        
            self.fitted_ND_count = self.raw_ND_count
            self.fitted_perc_ND = self.raw_perc_ND
        elif handle_zeros == 'study_consistent':
            self.raw_ND_count = full_df['LoD_count'].sum()
            self.raw_frac_ND = full_df['LoD_count'].mean()
            self.raw_perc_ND = self.raw_frac_ND*100

            group_cols = ['HERO_ID'] + self.fixed_effects
            full_df[self.PCBs] = full_df[self.PCBs].astype(str) # Make sure they're all strings for picking out LoD
            study_missing = full_df.groupby('HERO_ID')[self.PCBs].max()
            study_drop_LoD = study_missing[study_missing == 'LoD'].dropna(how='any', axis=0).index.values
            full_df = full_df[~(full_df['HERO_ID'].isin(study_drop_LoD) & full_df.LoD_count==1)]
            #full_df['N_sites'] = full_df.groupby('HERO_ID')['sample_name'].transform('count')
            #full_df.replace('LoD', 0.000001, inplace=True) # Replace remaining LoD with low value ### ORIG
            for col in df_map.columns:
                full_df[col] = full_df.apply(lambda row: df_map.loc[row['HERO_ID'], col] if row[col] == 'LoD' else row[col], axis=1)

            self.full_df_map_fill = full_df.copy()
            self.fitted_ND_count = self.raw_ND_count
            self.fitted_perc_ND = self.raw_perc_ND
        elif handle_zeros == 'drop_ND':
            self.raw_ND_count = full_df['LoD_count'].sum()
            self.raw_frac_ND = full_df['LoD_count'].mean()
            self.raw_perc_ND = self.raw_frac_ND*100
            
            full_df.replace('LoD', np.nan, inplace=True)
            full_df.dropna(how='any', axis=0, inplace=True, subset=self.PCBs)
            self.fitted_ND_count = 0
            self.fitted_perc_ND = 'Drop ND'
        
        else:
            #TODO: Fill in other options for handling zeros
            pass
        #X = X.astype(float)
        #features_df = self.features_df.loc[X.index].copy()
        #full_df = pd.concat([features_df, X], axis=1)
        full_df[self.PCBs] = full_df[self.PCBs].replace(',', '', regex=True).astype(float) # Remove and commas in the numbers before converting to float
        
        self.tmp_full_df = full_df.copy()
        
        if clean_up:
            self.full_df = self._clean_up_data(full_df)
            #self.feature_list.append('N')
            check_site_data(self.full_df, 'site_info', ['site-level average', 'study average'], self.PCBs)
        
        X = self.full_df[self.PCBs].copy()
        self.features = self.full_df[self.feature_list].copy()
        site_numbers = self.features.groupby('HERO_ID').cumcount() + 1
        #self.features['site'] = self.features['HERO_ID'].astype(str) + '_' + site_numbers.astype(str)
        self.features['site'] = site_numbers
        self.features['LoD'] = self.full_df['LoD_count']
        self.X_norm = X.divide(X.sum(axis=1), axis=0)#.tail(20)
        #self.X_norm.loc[self.features.LoD==1] = np.round(self.X_norm.loc[self.features.LoD==1],1) # Change LoD to 1/0
        #self.features = features_df.loc[self.X_norm.index]        
        self.data = pd.concat([self.features, self.X_norm], axis=1)
        if set_max_prop:
            pass
            #self.data[self.PCBs] = self.data[self.PCBs].where(self.data[self.PCBs] < 0.99, 0.99)
            #self.data[self.PCBs] = self.data[self.PCBs].where(self.data[self.PCBs] > 0.01, 0.01)
            #LoD_replace = self.data.loc[self.data.LoD == 0, self.PCBs].median()
            #self.LoD_replace = LoD_replace
            #self.data.loc[self.data.LoD == 1, self.PCBs] = LoD_replace.values
        
        # Drop studies that only have one site
        site_level_df = self.data[self.data.site_info == 'site-level average'].copy()
        total_sites = site_level_df.groupby('HERO_ID')['N'].sum()
        hero_drop = list(total_sites[total_sites <= 1].index)
        self.data = self.data[~self.data['HERO_ID'].isin(hero_drop)]
        
        if self.site_level_only:
            self.data = self.data[self.data.site_info == 'site-level average']
        #else:
        #    df_repeated = self.data.loc[self.data['site_info'] == 'study average'].copy()
        #    df_repeated = df_repeated.loc[df_repeated.index.repeat(df_repeated['N'])].reset_index(drop=True)
        #    df_repeated = df_repeated.reset_index(drop=True)
        #    df_non_study = self.data.loc[self.data['site_info'] != 'study average'].copy()
        #    self.data = pd.concat([df_repeated, df_non_study], ignore_index=True)
        
    def fit_coelution(self, disp_summary=False):
        """Use brms to fit the coelution combination"""
        check_empty_df(self.data, self.PCBs)
            
        available_effects = self.data[self.fixed_effects].nunique()
        self.fixed_effects = [x for x in self.fixed_effects if available_effects[x] > 1]
        self._assign_model(self.fixed_effects)
        
        self.used_all = len(self.fixed_effects) == len(self.orig_effects)
        
        if len(self.data.HERO_ID.unique()) > 1:
            self.random_effects = True
        else:
            self.random_effects = False
        
        if self.randomize_y:
            # Randomize the response variable (PCB proportions)
            randomized_indices = np.random.permutation(self.data.index)
            self.data[self.PCBs] = self.data.loc[randomized_indices, self.PCBs].reset_index(drop=True)


        self.compile_R_code()
        
        all_pcbs = ', '.join(self.PCBs)
        #self.mod = brms.brm(formula=ro.r(formula), data=p2r.py2rpy(self.data), family=brms.Beta(), chains=1)
        self.mod = self.run_R.run_brms(p2r.py2rpy(self.data), self.loo_comparison)
        if disp_summary:
            print(base.summary(self.mod))
        self._assign_sig_effects()
    
    def model_summary(self):
        print(base.summary(self.mod))

    def _assign_phi(self, row, fixed_effects_df):
        phi = fixed_effects_df.loc['phi_Intercept', 'Estimate']
        for param in self.fixed_effects:
            if param in row.index:
                weight_index = f'phi_{param}{row[param]}'
                phi += fixed_effects_df.loc[weight_index, 'Estimate'] if weight_index in fixed_effects_df.index else 0
        return phi

    def create_coelution_str(self):
        def extract_integers(s):
            return int(re.search(r'\d+', s).group())
        integers_list = [extract_integers(item) for item in self.PCBs]
        return '+'.join(map(str, integers_list))
    def sample_posterior(self, col_order=[]):
        def assign_PCB(row, mapper):
            return mapper[row['.category']]
        
        self.ppc_pop = p2r.rpy2py(self.run_R.sample_ppc(self.mod, p2r.py2rpy(self.data[self.fixed_effects]), self.draw_type))
        self.fixed_effects_posterior = p2r.rpy2py(self.run_R.fixed_params(self.mod))
        
        mapper = dict(zip(self.ppc_pop['.category'].unique(), self.PCBs))
        self.ppc_pop['PCB'] = self.ppc_pop.apply(assign_PCB, args=(mapper,), axis=1)
        #self.r_summary = self.run_R.ppc_summary(self.mod, p2r.py2rpy(self.data[self.fixed_effects]))
        
        self.prop_summary = self.ppc_pop.groupby(self.fixed_effects + ['PCB'])[self.draw_type].apply(lambda x: az.summary(x.values, kind='stats', hdi_prob=0.95, skipna=True)).reset_index()#.drop(columns='level_3')#.sort_values(by='phase')
        self.prop_summary = self.prop_summary.filter(regex='^(?!level_)')
        self.prop_summary['phi'] = np.round(self.prop_summary.apply(self._assign_phi, args=(self.fixed_effects_posterior,), axis=1), 3) # If intercept model, done
        if not self.fixed_effects:
            self.prop_summary['sample_type'] = 'combined'
            self.prop_summary['N_sites'] = self.data.N.sum().astype(int)
            self.prop_summary['N_studies'] = self.data.HERO_ID.nunique()
        else:
            grouped_df3 = self.data.groupby(self.fixed_effects).agg({'N': 'sum', 'HERO_ID': 'nunique'}).reset_index()
            df1 = pd.merge(self.prop_summary, grouped_df3, on=self.fixed_effects, how='left')
            self.prop_summary['N_sites'] = df1['N'].fillna(0).astype(int)
            self.prop_summary['N_studies'] = df1['HERO_ID'].fillna(0).astype(int)
        
        self.prop_summary['raw_ND_count'] = self.raw_ND_count
        self.prop_summary['raw_perc_ND'] = self.raw_perc_ND

        self.prop_summary['fitted_ND_count'] = self.fitted_ND_count
        self.prop_summary['fitted_perc_ND'] = self.fitted_perc_ND
        #self.prop_summary['cv'] = self.prop_summary['sd']/self.prop_summary['mean']
        
        self.prop_summary['coelution'] = self.create_coelution_str()
        if col_order:
            self.prop_summary = self.prop_summary[col_order]

        
        #self.prop_summary['diff'] = self.prop_summary.iloc[:, -1] - self.prop_summary.iloc[:, -2]
        
        if self.random_effects and not self.nested:
            self.ppc_study = p2r.rpy2py(self.run_R.sample_ppc(self.mod, p2r.py2rpy(self.data[self.fixed_effects + ['HERO_ID']]), self.draw_type, study=True))
            self.ppc_study['PCB'] = self.ppc_study.apply(assign_PCB, args=(mapper,), axis=1)
            
            new_study_df = self.data[self.fixed_effects].copy()
            new_study_df['HERO_ID'] = 'Generic study'
            new_study_df['site'] = 'Generic site'
            self.new_study = p2r.rpy2py(self.run_R.sample_ppc_new_study(self.mod, p2r.py2rpy(new_study_df), self.draw_type))
            self.new_study['PCB'] = self.new_study.apply(assign_PCB, args=(mapper,), axis=1)
            
            self.new_summary = self.new_study.groupby(self.fixed_effects + ['PCB'])[self.draw_type].apply(lambda x: az.summary(x.values, kind='stats', hdi_prob=0.95)).reset_index()#.drop(columns='level_3')#.sort_values(by='phase')
            self.new_summary = self.new_summary.filter(regex='^(?!level_)')
    def plot_comparison(self, observed='indiv', prediction='pop', dodge=False, save=False, use_legend=True):
        """
        Plot the comparison between the posterior and the observed data

        Parameters
        ----------
        observed : TYPE, optional
            Observed data to plot. 'indiv' is the individual points while 
            'study' is the study-level mean. The default is 'indiv'.
        prediction : TYPE, optional
            Prediction distrubtion to plot

        Returns
        -------
        None.

        """
        data_plot = self.data.copy()
        #X_norm_plot['label'] = feature_design.apply(assign_label, axis=1)
        plot_vars = self.fixed_effects + ['HERO_ID']
        data_plot = pd.melt(data_plot, value_vars=self.PCBs, id_vars=plot_vars)
        data_plot['HERO_ID'] = data_plot['HERO_ID'].astype(str)
        self.data_plot = data_plot
        HERO_IDs = data_plot['HERO_ID'].unique()
        pal = sns.color_palette('tab10', len(HERO_IDs))
        palette = dict(zip(HERO_IDs, pal))
        if all([x in self.hero_map.hero_id.values for x in palette.keys()]) and use_legend:
            # Only change to short ciation if all hero_ids are in the mapper
            #palette = {self.hero_map.loc[self.hero_map.hero_id == k, 'short'].values[0]: v for k,v in tmp_palette.items()}
            hero_mapper = {k: self.hero_map.loc[self.hero_map.hero_id == k, 'short'].values[0] for k in palette.keys()}
        else:
            hero_mapper = None

        if prediction == 'study':
            plot_chain = self.ppc_study.copy()
            dodge=True
            plot_chain['HERO_ID'] = plot_chain['HERO_ID'].astype(str)
            plot_chain = plot_chain.merge(data_plot[plot_vars], on=plot_vars, how='inner')
        elif prediction == 'pop':
            plot_chain = self.ppc_pop.copy()
            #plot_chain = self.new_study.copy()
        
        self.data_plot = data_plot
        self.plot_chain = plot_chain
        n_effects = len(np.unique(plot_chain[self.fixed_effects].values))

        if n_effects == 0: # Intercept only
            fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(8,12))
            if prediction == 'pop':
                sns.violinplot(x=self.draw_type, y="PCB", data=plot_chain, ax=ax)
                #sns.swarmplot(x=self.draw_type, y="PCB", data=plot_chain, ax=ax)
            elif prediction == 'study':
                sns.violinplot(x=self.draw_type, y="PCB", hue = 'HERO_ID', hue_order=palette.keys(), data=plot_chain, ax=ax, palette=palette)
            plt.setp(ax.collections, alpha=.3)
            
            if observed == 'indiv':
                sns.swarmplot(x="value", y="variable", hue="HERO_ID", data=data_plot, ax=ax, dodge=dodge, palette=palette)#, size=3)
            elif observed == 'study':
                sns.swarmplot(x="value", y="variable", hue="HERO_ID", data=data_plot.groupby(['HERO_ID', 'variable'])['value'].mean().reset_index(), ax=ax, size=10, palette=palette, dodge=dodge)
            ax.set_title('Intercept Only', fontsize=18)
            ax.set_ylabel('')
            ax.set_xlabel('Proportion', fontsize=16)
            ax.xaxis.set_tick_params(labelsize = 14)
            ax.yaxis.set_tick_params(labelsize = 14)
            if hero_mapper is not None:
                #legend = ax.legend()
                #for handle, label in zip(legend.get_legend_handles_labels(), legend.get_texts()):
                handles, labels = ax.get_legend_handles_labels()
                for i, label in enumerate(labels):
                    citation = hero_mapper.get(label, label)
                    labels[i] = citation
                ax.legend(handles, labels)

            ax.set_xlim([0,1])
            # remove legend
            if not use_legend:
                ax.get_legend().remove()
        else:
            nrows = int(np.ceil(np.sqrt(n_effects)))
            ncols = int(np.ceil((n_effects)/float(nrows)))
            if (ncols*nrows > n_effects) and (n_effects < 4):
                ncols = 1
                nrows = n_effects
            
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(8,12))
            
            try:
                axes = axs.flatten()
            except:
                axes = [axs]
            data_groupby = data_plot.groupby(self.fixed_effects)
            #for ax, (n1, grp1), (n2,grp2) in zip(axes, plot_chain.groupby(self.fixed_effects), data_plot.groupby(self.fixed_effects)):
            for ax, (n1, grp1) in zip(axes, plot_chain.groupby(self.fixed_effects)):
                if prediction == 'pop':
                    sns.violinplot(x=self.draw_type, y="PCB", data=grp1, ax=ax)
                elif prediction == 'study':
                    sns.violinplot(x=self.draw_type, y="PCB", hue = 'HERO_ID', hue_order=palette.keys(), data=grp1, ax=ax, palette=palette)
                plt.setp(ax.collections, alpha=.3)
                if len(n1) == 1:
                    test = n1[0]
                else:
                    test = n1
                if test in data_groupby.groups.keys():
                    grp2 = data_groupby.get_group(test)
                    if observed == 'indiv':
                        # hue_order=palette.keys() if we need to get the right order
                        sns.swarmplot(x="value", y="variable", hue="HERO_ID", data=grp2, ax=ax, dodge=dodge, palette=palette)#, size=3)
                    elif observed == 'study':
                        sns.swarmplot(x="value", y="variable", hue="HERO_ID", data=grp2.groupby(['HERO_ID', 'variable'])['value'].mean().reset_index(), ax=ax, size=10, dodge=dodge, palette=palette)
                #ax.set_title(n1[0].capitalize(), fontsize=18)
                ax.set_title('-'.join(list(n1)), fontsize=18)
                if hero_mapper is not None:
                    handles, labels = ax.get_legend_handles_labels()
                    for i, label in enumerate(labels):
                        citation = hero_mapper.get(label, label)
                        labels[i] = citation
                    ax.legend(handles, labels)
                ax.set_ylabel('')
                ax.set_xlabel('Proportion', fontsize=16)
                ax.xaxis.set_tick_params(labelsize = 14)
                ax.yaxis.set_tick_params(labelsize = 14)
                ax.set_xlim([0,1])
                # remove legend
                if not use_legend:
                    ax.get_legend().remove()
        fig.tight_layout()
        
        if save:
            if self.fit_phi:
                phi_save = ''
            else:
                phi_save = '_const_phi'
            #fig.savefig('../training_fits_pdf/%s%s/%s.pdf'%(self.model_type, phi_save, '_'.join(self.PCBs)), bbox_inches='tight')
            fig.savefig('../training_fits_wLegend/%s%s/%s.pdf'%(self.model_type, phi_save, '_'.join(self.PCBs)), bbox_inches='tight')
        
        plt.show()
    
    def investigate_arochlor(self):
        df = self._import_data(arochlor=True)
        self.df = df

    def load_IADN(self):       
        df_IADN = pd.read_csv('../data/IADN_props.csv', index_col=0)
        df_IADN.rename(columns={'Chemical': 'PCB'}, inplace=True)
        self.df_IADN = df_IADN[df_IADN.coelution == '_'.join(self.PCBs)]
    
    def check_IADN_list(self, coelution_list):
        """Given a list of coelutions, return the ones we have IADN data on"""
        df_IADN = pd.read_csv('../data/IADN_props.csv', index_col=0)
        df_IADN.rename(columns={'Chemical': 'PCB'}, inplace=True)

        return [x for x in coelution_list if '_'.join(x) in df_IADN.coelution.to_list()]

    
    def compare_IADN(self, compare_type='violin', phase=None, save=False):
        # IADN data are sample_type='outdoor' and phase='gas'
        data_plot = self.df_IADN.copy()
        data_plot['label'] = 'IADN'
        #X_norm_plot['label'] = feature_design.apply(assign_label, axis=1)
        
        if self.model_type == 'sample_only':
            plot_chain = self.ppc_pop.loc[self.ppc_pop.sample_type=='outdoor'].copy()
        elif self.model_type == 'intercept_only':
            plot_chain = self.ppc_pop.copy()
        else:
            if phase is None:
                check_for_gas(self.ppc_pop, 'phase', 'gas')
                use_phase = 'gas'
            else:
                use_phase = phase
            
            plot_chain = self.ppc_pop.loc[(self.ppc_pop.sample_type=='outdoor') & (self.ppc_pop.phase==use_phase)].copy()
        
        
        plot_chain['label'] = 'Prediction'
        
        
        if compare_type == 'violin':
            fig, ax = plt.subplots(figsize=(8,12))
            #fig, axes = plt.subplots(nrows=nrows, sharex=True, sharey=True, figsize=(8,12))
            tmp_data = pd.concat([plot_chain.rename(columns={self.draw_type: 'proportion'}), data_plot])
            self.tmp_data = tmp_data
            #sns.violinplot(x=self.draw_type, y="PCB", data=plot_chain, ax=ax)
            #sns.swarmplot(x="value", y="variable", hue="HERO_ID", data=data_plot.groupby(['Site'])['value'].mean().reset_index(), ax=ax, size=10)
            #sns.swarmplot(x="proportion", y="PCB", hue="Site", data=data_plot, ax=ax, dodge=False)
            #sns.swarmplot(x="proportion", y="PCB", data=data_plot, ax=ax, size=1)
            sns.violinplot(x="proportion", y="PCB", data=tmp_data, ax=ax, hue='label', dodge=False)
            plt.setp(ax.collections, alpha=.4)
            plt.show()
        elif compare_type == 'density':
            full_df = pd.concat([plot_chain.rename(columns={self.draw_type:'proportion'}), data_plot])
            sns.displot(x='proportion', col='PCB', data=full_df, hue='label',kind="kde")#, common_norm=True)
            #ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        self.tmp_data = tmp_data
        if save:
            #fig.savefig('../IADN_fits_intercept/%s.png'%'_'.join(self.PCBs), bbox_inches='tight')
            fig.savefig('../IADN_fits_intercept_pdf/%s.pdf'%'_'.join(self.PCBs), bbox_inches='tight')
    def _calc_summary(self, group):
        return az.summary(group[self.draw_type], hdi_prob=0.95)

    def get_posterior_predictions(self):
        return self.ppc_study.groupby(['PCB']+self.fixed_effects).apply(self._calc_summary)

    def investigate_IADN(self, PCB):
        obs = self.df_IADN.loc[self.df_IADN.PCB==PCB]
        sns.pointplot(x='Year', y='proportion', data=obs, hue='Site')
        #plt.yscale('log')
        plt.ylabel('%s/%s proportion'%(PCB, '_'.join(self.PCBs)))
        plt.ylim([0,1])
        _ = plt.xticks(rotation=90)

    def calc_cohens(self, PCB, phase=None, method='non_param_d', QQ_plot=False):
        def cohens_d(array1, array2):
            # This is actually Hedges-G
            # calculate means
            mean1, mean2 = np.mean(array1), np.mean(array2)
            
            # Calculate pooled variances
            n1, n2 = len(array1), len(array2)
            var1, var2 = np.var(array1, ddof=1), np.var(array2, ddof=1)
            pooled_sd = np.sqrt(((n1-1)*var1 + (n2-1)*var2)/(n1 + n2 -2))
            #pooled_sd = np.sqrt((np.std(array1)**2 + np.std(array2)**2))/2
            
            return (mean1 - mean2)/pooled_sd
        def non_param_d(array1, array2, quantile=0.5):
            val1, val2 = np.quantile(array1, quantile), np.quantile(array2, quantile)

            # Calculate pooled MAD
            n1, n2 = len(array1), len(array2)
            mad1, mad2 = scs.median_abs_deviation(array1), scs.median_abs_deviation(array2)
            PMAD = np.sqrt(((n1-1)*mad1**2 + (n2-1)*mad2**2)/(n1 + n2 - 2))
            return (val1 - val2)/PMAD

        if self.model_type == 'sample_only':
            pred = self.ppc_pop.loc[(self.ppc_pop.PCB == PCB) & (self.ppc_pop.sample_type=='outdoor'), self.draw_type].copy()
        elif self.model_type == 'intercept_only':
            pred = self.ppc_pop.loc[(self.ppc_pop.PCB == PCB), self.draw_type].copy()
        else:
        
            if phase is None:
                check_for_gas(self.ppc_pop, 'phase', 'gas')
                use_phase = 'gas'
            else:
                use_phase = phase
        
            pred = self.ppc_pop.loc[(self.ppc_pop.PCB == PCB) & (self.ppc_pop.sample_type=='outdoor') & (self.ppc_pop.phase==use_phase), self.draw_type]
        
        obs = self.df_IADN.loc[self.df_IADN.PCB==PCB, 'proportion']

        self.obs = obs.values
        self.pred = pred.values

        if QQ_plot:
            sm.qqplot_2samples(obs.values, pred.values, line='45')
        #obs = self.data.loc[self.data.sample_type=='outdoor', PCB]
        if method == 'cohens_d':
            return np.abs(cohens_d(pred.values, obs.values))
        elif method == 'non_param_d':
            return np.abs(non_param_d(pred.values, obs.values))
        elif method == 'KL_divergence':
            return scs.entropy(obs.values, pred.values)
        elif method == 'mutual':
            return skm.mutual_info_score(obs.values, pred.values)
        elif method == 'wasserstein':
            return scs.wasserstein_distance(obs.values, pred.values)
        elif method == 'overlap':
            return self.run_R.calc_overlap(n2r.py2rpy(pred.values), n2r.py2rpy(obs.values))[0]*100
        elif method == 'coverage_prob':
            low, high = az.hdi(pred.values)
            return np.mean((low < obs.values) & (obs.values < high))*100
        elif method == 'median':
            return scs.median_test(obs.values, pred.values)
        #print(cohens_d(pred.values, obs.values))
        
        #print(self.run_R.calc_overlap(n2r.py2rpy(pred.values), n2r.py2rpy(obs.values)))
    
            
            