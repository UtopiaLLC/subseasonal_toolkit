#!/usr/bin/env python
# coding: utf-8

# # MultiLLR: local linear regression with multitask feature selection

# In[ ]:


# Ensure notebook is being run from base repository directory
import os, sys
# Disable numpy multithreading
os.environ.update(
    OMP_NUM_THREADS = '1',
    OPENBLAS_NUM_THREADS = '1',
    NUMEXPR_NUM_THREADS = '1',
    MKL_NUM_THREADS = '1',
)
from subseasonal_toolkit.utils.notebook_util import isnotebook
if isnotebook():
    # Autoreload packages that are modified
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
else:
    from argparse import ArgumentParser
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import gc
import pickle

from ttictoc import tic, toc
from subseasonal_data.utils import get_measurement_variable
from subseasonal_toolkit.utils.general_util import printf, num_available_cpus, make_directories
from subseasonal_toolkit.utils.experiments_util import get_id_name, get_th_name, get_first_year, get_start_delta, clim_merge, month_day_subset
from subseasonal_toolkit.utils.fit_and_predict import apply_parallel
from subseasonal_toolkit.utils.skill import skill_report, skill_report_summary_stats_multicol
from subseasonal_toolkit.models.multillr.stepwise_util import *
from subseasonal_toolkit.utils.models_util import (get_submodel_name, start_logger, log_params, get_forecast_filename,
                                                   save_forecasts)
from subseasonal_toolkit.utils.eval_util import get_target_dates
from subseasonal_data import data_loaders


# In[ ]:


#
# Specify model parameters
#
model_name = "multillr"

if not isnotebook():
    # If notebook run as a script, parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument("pos_vars",nargs="*")  # gt_id and horizon                                                                                  
    parser.add_argument('--target_dates', '-t', default="std_test")
    # margin_in_days: regression will train on all dates within margin_in_days 
    #  of the target day and month in each training year
    # Set to 0 to include only target month-day combo
    # Set to "None" to include entire year
    parser.add_argument('--margin_in_days', '-m', default=56)
    # criterion: string criterion to use for backward stepwise (use "mean" to reproduce results in paper).
    #  Choose from: mean, mean_over_sd, similar_mean, similar_mean_over_sd, similar_quantile_0.5,
    #  similar_quantile_0.25, similar_quantile_0.1 or create your own in the function
    #  skill_report_summary_stats in skill.py
    parser.add_argument('--criterion', default="mean") 
    parser.add_argument('--metric', default="rmse",
                        help="metric for assessing improvement in "
                             "\{\'cos\', \'rmse\', \'mse\'\}")
    parser.add_argument('--num_cores', default=num_available_cpus(),
                       help="number of cores to use in execution")
    parser.add_argument('--date_order_seed', default="None",
                       help="None or integer determining random order in which "
                            "target dates are processed; if None, target dates "
                            "are sorted by day of the week")
    args, opt = parser.parse_known_args()
    
    # Assign variables
    gt_id = get_id_name(args.pos_vars[0]) # "contest_precip" or "contest_tmp2m"                                                                            
    horizon = get_th_name(args.pos_vars[1]) # "34w" or "56w"                                                                                        
    target_dates = args.target_dates
    if args.margin_in_days == "None":
        margin_in_days = None
    else:
        margin_in_days = int(args.margin_in_days)
    criterion = args.criterion
    metric = args.metric
    num_cores = args.num_cores
    if args.date_order_seed == "None":
        date_order_seed = None
    else:
        date_order_seed = int(args.date_order_seed)
else:
    # Otherwise, specify arguments interactively 
    gt_id = "contest_tmp2m" 
    horizon = "56w" 
    target_dates = "20210223"
    margin_in_days = 56
    criterion = 'mean' 
    metric = 'rmse'
    num_cores = num_available_cpus()
    date_order_seed = None

#
# Process model parameters
#

# Ensure metric is valid
valid_metrics = ['rmse','mse','cos']
if metric not in valid_metrics:
    raise ValueError(f"Unknown metric {metric}. Please choose from {valid_metrics}.")

# Get list of target date objects
target_date_objs = pd.Series(get_target_dates(date_str=target_dates,horizon=horizon))
if date_order_seed is None:
    # Sort target_date_objs by day of week
    target_date_objs = target_date_objs[target_date_objs.dt.weekday.argsort(kind='stable')]
else:
    # Sort target_date_objs in random order
    target_date_objs = target_date_objs.sample(frac=1, random_state=date_order_seed)

# Store delta between target date and forecast issuance date
# forecast_delta = timedelta(days=get_forecast_delta(horizon))
forecast_delta =  timedelta(days=get_start_delta(horizon, gt_id))


# Identify measurement variable name
measurement_variable = get_measurement_variable(gt_id) # 'tmp2m' or 'precip'

# column names for gt_col, clim_col and anom_col
gt_col = measurement_variable
clim_col = measurement_variable+"_clim"
anom_col = get_measurement_variable(gt_id)+"_anom" # 'tmp2m_anom' or 'precip_anom'

# anom_inv_std_col: column name of inverse standard deviation of anomalies for each start_date
anom_inv_std_col = anom_col+"_inv_std"

#
# Default stepwise parameter values
#
# Define candidate predictors
initial_candidate_x_cols = default_stepwise_candidate_predictors(gt_id, horizon)

### TODO: REMOVE THIS BLOCK OF CODE
if False:
    initial_candidate_x_cols = [col for col in initial_candidate_x_cols if not col.startswith("phase_")]

# Copy the list of candidates for later modification
### TODO: unnecessary to have both initial_candidate and candidate; get rid of one
candidate_x_cols = initial_candidate_x_cols[:]
# Tolerance for convergence: if performance hurt by more than tolerance, terminate.
tolerance = 0.001 if metric == 'rmse' else 0.01
# Whether to use margin days (days around the target date) in assessing
use_margin = False
# Whether metric should be maximized or minimized
maximize_metric = metric in {'cos','cos_margin'}


#
# Default regression parameter values
#
# anom_scale_col: multiply anom_col by this amount prior to prediction
# (e.g., 'ones' or anom_inv_std_col)
anom_scale_col = 'ones'
# pred_anom_scale_col: multiply predicted anomalies by this amount
# (e.g., 'ones' or anom_inv_std_col)
pred_anom_scale_col = 'ones'
# choose first year to use in training set
first_train_year = 1948 if gt_id == 'contest_precip' else 1979
# columns to group by when fitting regressions (a separate regression
# is fit for each group); use ['ones'] to fit a single regression to all points
group_by_cols = ['lat', 'lon']
# base_col: column which should be subtracted from gt_col prior to prediction
# (e.g., this might be clim_col or a baseline predictor like NMME);
# if None, subtract nothing
base_col = None
# Name the collection of supporting columns
# Discard None if it exists
supporting_cols = set([base_col,clim_col,anom_col]).difference([None])

#
# Record submodel names
#
### TODO: decide whether name_to_params json file should sit somewhere else
submodel_name = get_submodel_name(
    model_name, margin_in_days=margin_in_days, criterion=criterion,
    metric=metric, x_cols = initial_candidate_x_cols)

if not isnotebook():
    # Save output to log file
    logger = start_logger(model=model_name,submodel=submodel_name,gt_id=gt_id,
                          horizon=horizon,target_dates=target_dates)
    # Store parameter values in log                                                                                                                        
    params_names = ['gt_id', 'horizon', 'target_dates',
                    'margin_in_days', 'criterion',
                    'metric', 'initial_candidate_x_cols'
                   ]
    params_values = [eval(param) for param in params_names]
    log_params(params_names, params_values)

#
# Create directory for storing model outputs, like
# convergence indicators, prediction paths, and summary statistics
#
task = f"{gt_id}_{horizon}"
outdir = os.path.join('models', model_name, 'storage', submodel_name, task)
make_directories(outdir)


# In[ ]:


#
# Load and process dataset
#
### TODO: can we disable numpy multithreading after this step so feather can load data in parallel?
# Load relevant data columns, excluding clim_col
relevant_cols = supporting_cols.union(['start_date','lat','lon']+group_by_cols+
    candidate_x_cols+[anom_scale_col,pred_anom_scale_col]) - set([clim_col])
if gt_id.endswith("precip"):
    data = data_loaders.load_combined_data('all_data', gt_id, horizon, columns=relevant_cols - set(['tmp2m_clim']))
else:
    data = data_loaders.load_combined_data('all_data', gt_id, horizon, columns=relevant_cols)

last_target_date = target_date_objs.max()
printf(f"Restricting data to years >= {first_train_year}"
       f" and dates <= {last_target_date}")
tic()
data = data[(data.start_date <= last_target_date) &
            (data.start_date >= f"{first_train_year}-01-01")]
toc()

printf("Merging in climatology")
tic()
clim = data_loaders.get_climatology(gt_id).rename(columns={gt_col:clim_col})
data = clim_merge(data, clim)
toc()

printf(f"Adding sample weight and target columns")
tic()
### TODO: could define sample_weight_col = 'ones' in typical case
# To minimize the mean-squared error between predictions of the form
# (f(x_cols) + base_col - clim_col) * pred_anom_scale_col
# and a target of the form anom_col * anom_scale_col, we will
# estimate f using weighted least squares with datapoint weights
# pred_anom_scale_col^2 and effective target variable
# anom_col * anom_scale_col / pred_anom_scale_col + clim_col - base_col
data['sample_weight'] = data[pred_anom_scale_col]**2
# Ensure that we do not divide by zero when dividing by pred_anom_scale_col
data['target'] = (data[clim_col] - (0 if base_col is None else data[base_col]) +
                  data[anom_col] * data[anom_scale_col] /
                    (data[pred_anom_scale_col]+(data[pred_anom_scale_col]==0)))
# Add sample_weight and target to supporting cols
supporting_cols.update(['sample_weight','target'])
toc()

### TODO: Include tmp2m_clim in precip lat_lon_date_data
if gt_id.endswith("precip") and "tmp2m_clim" in candidate_x_cols:
    print("Adding tmp2m_clim to dataframe")
    tic()
    tmp_clim = data_loaders.get_climatology(gt_id.replace("precip","tmp2m")).rename(columns={'tmp2m':'tmp2m_clim'})
    data = data.merge(tmp_clim[['tmp2m_clim']], left_on=['lat','lon',data.start_date.dt.month,data.start_date.dt.day], 
                      right_on=[tmp_clim.lat, tmp_clim.lon, tmp_clim.start_date.dt.month, tmp_clim.start_date.dt.day],
                      how="left").drop(['key_2', 'key_3'], axis=1)
    toc()


# In[ ]:


if False:
    ###TODO: REMOVE ME
    cfsv2 = load_measurement("data/dataframes/subx-cfsv2-tmp2m-all_leads-4_periods_avg.h5", shift=30)
    cfsv2.set_index(['lat','lon','start_date'],inplace=True)
    data = pd.merge(data, cfsv2[["subx_cfsv2_tmp2m-0.5d_shift30","subx_cfsv2_tmp2m-28.5d_shift30"]],
                   left_on=['lat','lon','start_date'], right_index=True,
                   how="left")


# In[ ]:


def backward_rolling_linear_regression(X, y, sample_weight, t, threshold_date, 
                                       ridge=0.0, test_X=None, test_t=None):
    """Fits backward rolling weighted ridge regression without an intercept.  
    For the equivalent threshold date in each year, 
    forms 'hindcast' predictions based on leaving out one year
    of training data at a time.  Fits regression with each column of X withheld 
    and returns hindcast predictions for each model fit in a DataFrame with 
    columns corresponding to the withheld column name.
    
    Args:
       X: feature matrix
       y: target vector
       sample_weight: weight assigned to each datapoint
       t: DateTimeIndex corresponding to rows of X
       threshold_date: Cutoff used to determine holdout batch boundaries (must not be Feb. 29);
          each batch runs from threshold_date in one year (exclusive) to threshold_date
          in subsequent year (inclusive)
       ridge (optional): regularization parameter for ridge regression objective
          [sum_i (y_i - <w, x_i>)^2] + ridge ||w||_2^2
       test_X: test feature matrix; if None, use X as the test feature matrix
       test_t: DateTimeIndex corresponding to rows of test_X; if test_X is None, use t
    """
    # Process test set arguments
    if test_X is None:
        test_X = X
        test_t = t
    
    # Extract size of feature matrix
    (n, p) = X.shape
    
    # Multiply y and X by the sqrt of the sample weights
    if all(sample_weight == 1):
        wtd_y = y
        wtd_X = X
    else:
        sqrt_sample_weight = np.sqrt(sample_weight)
        wtd_y = y * sqrt_sample_weight
        wtd_X = X.multiply(sqrt_sample_weight, axis=0)
    
    # Maintain training sufficient statistics for linear regression
    Xty = np.dot(wtd_X.T, wtd_y)
    XtX = np.zeros((p,p))
    # Set diagonal of XtX to ridge regularization parameter
    if ridge != 0:
        np.fill_diagonal(XtX, ridge)
    # Add data component of XtX
    XtX += np.dot(wtd_X.T, wtd_X)

    # Store predictions associated with each model in dataframe
    preds = pd.DataFrame(index=test_X.index, columns=test_X.columns, 
                         dtype=np.float64)
    
    # Find range of years in dataset
    first_year = min(t.min().year, test_t.min().year)
    last_year = max(t.max().year, test_t.max().year)

    #
    # Produce hindcast predictions
    #            
    # Form hindcast predictions for all blocks
    #printf("Form hindcast predictions for each block with each column removed")
    #tic()
    for year in range(first_year,last_year+2):
        # Identify unweighted test data for prediction
        upper_threshold_date = threshold_date.replace(year = year)
        lower_threshold_date = threshold_date.replace(year = year-1)
        test_date_block = ((test_t <= upper_threshold_date) & 
                           (test_t > lower_threshold_date))
        test_X_slice = test_X[test_date_block]
        if test_X_slice.empty:
            # No predictions to make for this block
            continue
        # Find block of training dates for this year's threshold
        date_block = ((t <= upper_threshold_date) & 
                  (t > lower_threshold_date))
        # Form training data with current date block removed
        X_slice = wtd_X[date_block]
        y_slice = wtd_y[date_block]
        train_XtX = XtX-np.dot(X_slice.T,X_slice)
        train_Xty = Xty-np.dot(X_slice.T,y_slice)
        for col_ind, col in enumerate(X.columns):
            # Fit coefficients using all columns exept for col
            train_col_inds = list(range(col_ind)) + list(range(col_ind+1, p))
            # Find minimum norm solution (to be robust to poor conditioning)
            # Use numpy broadcast indexing to obtain submatrix of train_XtX
            coef = np.linalg.lstsq(train_XtX[np.ix_(train_col_inds,train_col_inds)], 
                                   train_Xty[train_col_inds], rcond=None)[0]
#             try:
#                 # Solve linear system when first argument is full rank
#                 # Use numpy broadcast indexing to obtain submatrix of train_XtX
#                 coef = np.linalg.solve(train_XtX[np.ix_(train_col_inds,train_col_inds)], 
#                                        train_Xty[train_col_inds])
#             except np.linalg.LinAlgError:
#                 #printf(f"Not full rank: {year}, {col}")
#                 # Otherwise, find minimum norm solution
#                 coef = np.linalg.lstsq(train_XtX[np.ix_(train_col_inds,train_col_inds)], 
#                                        train_Xty[train_col_inds], rcond=None)[0]
            
            # Store predictions on this block of dates for this candidate using unweighted data
            preds.loc[test_date_block,col] = np.dot(test_X_slice.iloc[:,train_col_inds], coef)
    #toc()
    # Return predictions
    return preds

def backward_rolling_linear_regression_wrapper(
    df, x_cols=None, base_col=None, clim_col=None, anom_col=None, 
    last_train_date=None, ridge=0.0, target_date=None):
    """Wrapper for backward_rolling_linear_regression that selects appropriate 
    training and test sets from df, associates sample weights with each training datapoint, 
    carries out rolling linear regression with each column removed, and for each test point
    returns hindcast anomalies per candidate column with ground truth anomalies and climatology
    
    Args:
        df: Dataframe with columns 'start_date', 'lat', 'lon', 
           clim_col, anom_col, x_cols, 'target', 'sample_weight'
        x_cols: Names of columns used as input features
        base_col: Name of column subtracted from target prior to prediction (or None
           if no column should be subtracted)
        clim_col: Name of climatology column in df
        anom_col: Name of ground truth anomaly column in df
        last_train_date: last date on which to train and cutoff used to determine 
           holdout batch boundaries (must not be Feb. 29); each batch runs from 
           last_train_date in one year (exclusive) to last_train_date
           in subsequent year (inclusive)
        ridge (optional): regularization parameter for ridge regression objective
           [sum_i (y_i - <w, x_i>)^2] + ridge ||w||_2^2
        target_date (optional): if not None, returns predictions only on points with the
           same month-day combination as target_date; otherwise, returns predictions
           on all points

    Returns predictions representing f(x_cols) + base_col - clim_col       
    """
    # Restrict to datapoints with valid features
    df = df.dropna(subset=x_cols)
    # Select training set by dropping training points after last_train_date
    # and points with invalid target or sample weights
    train_df = df.xs(
        slice(None, last_train_date), level='start_date', drop_level=False).dropna(
        subset=['target','sample_weight'])
    
    if target_date is None:
        # Test on all points if no target date given
        test_df = df
    else:
        # Test on points with same month-day combo as target
        dates = df.index.get_level_values(level='start_date')
        test_df = df[(dates.month == target_date.month) & (dates.day == target_date.day)]

    # Collect predictions and add base column minus climatology to predictions,
    if base_col is None:
        base_minus_clim = - test_df[clim_col].values
    else:
        base_minus_clim = test_df[base_col].values - test_df[clim_col].values
    preds = backward_rolling_linear_regression(
        train_df[x_cols],
        train_df['target'], 
        train_df['sample_weight'],
        train_df.index.get_level_values(level='start_date'),
        last_train_date, 
        ridge = ridge,
        test_X = test_df[x_cols],
        test_t = test_df.index.get_level_values(level='start_date')).add(
        base_minus_clim, axis = 'index')
    # Return dataframe with predicted anomalies, ground truth anomalies, and climatology
    preds = preds.assign(truth=test_df[anom_col].values,
                         clim=test_df[clim_col].values)
    return preds


# In[ ]:


### TODO: deal with leap days
### TODO: deal with fact that different predictors defined over different dates?
#
# Form predictions for each target date
#
for target_date_obj in target_date_objs: 
    # If any features are missing for target date, skip
    if data.loc[data.start_date == target_date_obj, candidate_x_cols].isnull().values.any():
        printf(f"warning: some features unavailable for target={target_date_obj}; skipping")
        printf("{}".format(data.loc[data.start_date == target_date_obj, candidate_x_cols].isnull().any()))
        continue
        
    target_date_str = datetime.strftime(target_date_obj, '%Y%m%d')
    printf(f'\ntarget={target_date_str}')
    
    # Check if algorithm has already converged for this target
    converged_outfile = os.path.join(outdir, 'converged-'+target_date_str)
    if os.path.exists(converged_outfile):
        # If algorithm has converged previously, skip
        printf('{} already exists; skipping.'.format(converged_outfile))
        continue

    #
    # Subset data for training
    #
    
    # Find the last observable training date for this target
    last_train_date = target_date_obj - forecast_delta
    if (last_train_date.day == 29) and (last_train_date.month == 2):
        # Treat Feb. 29 as Feb. 28
        last_train_date = last_train_date.replace(day = 28)

    printf("Restricting data to margin around target date")
    tic()
    if margin_in_days is not None:
        ### TODO: is there a faster way to do this?  maybe by indexing by start_date, lat, lon
        ### If we load dayofyear as a feature, could just measure proximity to target date's climpp?
        sub_data = month_day_subset(data, target_date_obj, margin_in_days)
    else:
        sub_data = data
    toc()
    printf("Dropping rows beyond target date and setting index")
    tic()
    sub_data = sub_data.loc[
        sub_data.start_date <= target_date_obj,:].set_index(
        ['lat','lon','start_date'])
    toc()

    #
    # Store target-date predictions and summary stats of each model on stepwise path
    #
    preds_outfile = os.path.join(outdir, 'preds-'+target_date_str+'.h5')
    stats_outfile = os.path.join(outdir, 'stats-'+target_date_str+'.pkl')
    if os.path.exists(preds_outfile) and os.path.exists(stats_outfile):
        # If preds and stats already exist on disk, load them and start from the latest model
        printf('Loading existing predictions and stats')
        tic()
        path_preds = pd.read_hdf(preds_outfile, key="data")
        path_stats = pickle.load( open( stats_outfile, "rb" ) )
        # The set of predictors in the latest model is specified by the last column name in
        # path_preds
        model_str = path_preds.columns[-1]
        current_x_col_set = eval(model_str)
        x_cols_current = list(current_x_col_set)
        printf(x_cols_current)
        # Enumerate non-model columns in path_preds
        non_model_cols = ['lat','lon','start_date','truth','clim']
        # Find the last feature removed from x cols
        best_x_col = (
            set(candidate_x_cols).symmetric_difference(current_x_col_set).pop() 
            if (len(path_preds.columns)-len(non_model_cols)) == 1 
            else current_x_col_set.symmetric_difference(
                eval(path_preds.columns[-2])).pop())
        printf(best_x_col)
        # Reconstruct the current best criterion
        best_criterion_current = path_stats[model_str][criterion][best_x_col]
        printf(best_criterion_current)
        # Last completed model round
        model_round = len(path_preds.columns) - 5
        toc()
    else:
        path_preds = data.loc[data.start_date == target_date_obj,
                              ['lat','lon','start_date',anom_col,clim_col]].copy()
        path_preds = path_preds.rename(index=str, columns={anom_col: "truth", clim_col: "clim"})
        path_stats = {}
        x_cols_current = candidate_x_cols[:]
        ### TODO: eventually replace with the score of training the full model
        best_criterion_current = -np.inf if maximize_metric else np.inf
        model_round = 0
        ### TODO: eventually get rid of set call and sort list of values?
        model_str = str(set(x_cols_current))
    
    #
    # Fit model
    #
    tic()
    printf("\nFitting backward stepwise regression")
    converged = False
    while not converged:
        model_round += 1
        printf(f"\nBackward stepwise regression; round {model_round}")
        tt = time.time()
        gc.collect()
        criteria = {}
        ### TODO: appropriately handle the empty set of predictors case (just predict 0)
        if not x_cols_current:
            converged = True
            break

        relevant_cols = supporting_cols.union(x_cols_current)
        printf("Fitting model with core predictors {}".format(x_cols_current))
        tic()
        preds = apply_parallel(sub_data.loc[:,relevant_cols].groupby(group_by_cols),
            backward_rolling_linear_regression_wrapper, num_cores,
#         preds = sub_data.loc[:,relevant_cols].groupby(group_by_cols).apply(
#             backward_rolling_linear_regression_wrapper, 
            x_cols=x_cols_current,
            base_col=base_col,
            clim_col=clim_col,
            anom_col=anom_col,
            last_train_date=last_train_date,
            target_date=target_date_obj)
        toc()
        # Ensure raw precipitation predictions are never less than zero
        if gt_id.endswith("precip"):
            printf("Ensure predicted precipitation >= 0")
            tic()
            preds[x_cols_current] = np.maximum(
                preds[x_cols_current].add(preds.clim, axis=0),0).subtract(
                preds.clim, axis=0)
            toc()
        preds = preds.reset_index()
        # Assess prediction quality
        printf("Getting skills")
        tic()
        skills = skill_report(preds, target_date_obj,
                              pred_cols=x_cols_current,
                              gt_anom_col='truth',
                              clim_col='clim',
                              include_trunc0 = False,
                              include_cos = metric == 'cos',
                              include_cos_margin = metric == 'cos_margin',
                              include_mse_by_latlon = False,
                              verbose = False)
        # Remove the target year from the skills dataframe so it isn't used in evaluation
        skills[metric] = skills[metric][skills[metric].index != target_date_obj]
        toc()
        summary_stats = skill_report_summary_stats_multicol(
            skills, metric = metric, use_margin = metric=='cos_margin')
        # Pick best column based on summary stats, and
        # compute difference from current best performance
        criteria = summary_stats[criterion]
        if maximize_metric:
            # Larger criterion values are better
            best_x_col = criteria.idxmax()
            improvement = criteria[best_x_col] - best_criterion_current
        else: 
            # Smaller criterion values are better
            best_x_col = criteria.idxmin()
            improvement = best_criterion_current - criteria[best_x_col]
        printf("-old criterion = {}, new criterion = {}, best_x_col = {}".format(
            best_criterion_current, criteria[best_x_col], best_x_col))
        printf("-improvement = {}, tolerance = -{}".format(improvement, tolerance))

        if improvement <= -tolerance:
            # Removing predictor is too costly; we're done
            converged = True
            printf("Model has converged")
            printf("--round {} elapsed time: {}s".format(model_round, time.time() - tt))
            break

        # Otherwise, performance hit is within tolerance:
        # Remove from model
        x_cols_current.remove(best_x_col)
        best_criterion_current = criteria[best_x_col]
        printf("Removing {} from model, current criterion is {}".format(best_x_col, best_criterion_current))
        # Store the predictions of the selected model
        printf("Storing the predictions of the selected model")
        tic()
        path_preds = pd.merge(path_preds,
                              preds.loc[preds.start_date == target_date_obj,
                                        ['lat','lon',best_x_col]],
                              on=["lat","lon"], how="left");
        # Rename added column to reflect the set of predictors in the model
        model_str = str(set(x_cols_current))
        path_preds = path_preds.rename(
            index=str, columns = { best_x_col : model_str })
        # Store summary stats of the selected model
        path_stats[model_str] = summary_stats
        toc()
        #
        # Save predictions and summary stats to disk after each round
        printf(f"Saving predictions and summary stats to disk after round {model_round}")
        # Write path predictions to file
        tic()
        path_preds.to_hdf(preds_outfile, key="data", mode="w", format="table")
        # Write path stats to file
        f = open(stats_outfile,"wb")
        pickle.dump(path_stats,f)
        f.close()
        toc()
        printf("Model hasn't converged")
        printf("--round {} elapsed time: {}s".format(model_round, time.time() - tt))
    toc()
    # Save prediction of selected model to file in standard format
    # after converting predicted anomalies into raw predictions
    path_preds[model_str] = path_preds[model_str].add(path_preds.clim)
    save_forecasts(
        path_preds[['lat','lon','start_date',model_str]].rename(
            columns={model_str:'pred'}),
        model=model_name, submodel=submodel_name, 
        gt_id=gt_id, horizon=horizon, 
        target_date_str=target_date_str)
    # Mark convergence by creating converged file
    printf('\nSaving ' + converged_outfile)
    open(converged_outfile, 'w').close()


# In[ ]:




