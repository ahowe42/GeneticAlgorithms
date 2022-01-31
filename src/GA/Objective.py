''' This module holds objective functions for use by the GA. '''
import math
import numpy as np
import pandas as pd
import sys
import ipdb
import scipy.stats as stt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.metrics import accuracy_score, f1_score

sys.path.append('../')
from Utils.Utils import *



def MinFunctionValue(func, params, data=None):
    '''
    Compute the value of a specified classical optimization test function, several
    of which are listed here: https://en.wikipedia.org/wiki/Test_functions_for_optimization.
    The input vector is incorrectly named "params" for compatibility with the GA. All
    functions require at least 2 input dimensions. Those that can accept > 2 are: Rastrigin,
    Sphere, Rosenbrock, Styblinsky-Tang.
    running function, and the data argument is not used.
    :param func: function name; one of 'Rastrigin', 'Ackley', 'Sphere', 'Rosenbrock',
        'Goldstein-price', 'Booth', 'Bukin6', 'Matyas', 'Levi13', 'Himmelblau', 'Easom',
        'Styblinsky-Tang'
    :param params: array_like vector of values for evaluation
    :param data: unused, here for compatibility with the GA runner
    :return val: single element list of value of the specified function at the
        specified input vector
    '''
    
    # get some aliases
    x = params[0]
    try:
        y = params[1]
        xy = x*y
    except IndexError as err:
        pass
    
    # get the number of dimensions
    n = len(params)
    
    if func == 'Rastrigin':
        val = 10*n + sum([x**2 - 10*math.cos(2*np.pi*x) for x in params])
    elif func == 'Ackley':
        val = -20*math.exp(-0.2*math.sqrt(0.5*(x**2 + y**2))) + \
            -math.exp(0.5*(math.cos(2*np.pi*x) + math.cos(2*np.pi*y))) + \
            math.exp(1) + 20
    elif func == 'Sphere':
        val = sum([x**2 for x in params])
    elif func == 'Rosenbrock':
        val = sum([100*(params[i+1]-params[i]**2)**2 + (1-params[i])**2 for i in range(len(params)-1)])
    elif func == 'Beale':
        val = (1.5 - x + xy)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)
    elif func == 'Goldstein-price':
        val = (1 + ((x + y + 1)**2)*(19 - 14*x + 3*x**2 - 14*y + 6*xy + 3*y**2))*\
            (30 + ((2*x - 3*y)**2)*(18 - 32*x + 12*x**2 + 48*y - 36*xy + 27*y**2))
    elif func == 'Booth':
        val = (x + 2*y - 7)**2 + (2*x + y - 5)**2
    elif func == 'Bukin6':
        val = 100*math.sqrt(abs(y - 0.01*x**2)) + 0.01*abs(x + 10)
    elif func == 'Matyas':
        val = 0.26*(x**2 + y**2) - 0.48*xy
    elif func == 'Levi13':
        val = math.sin(3*np.pi*x)**2 + ((x-1)**2)*(1 + math.sin(3*np.pi*y)**2) + \
            ((y-1)**2)*(1 + math.sin(2*np.pi*y)**2)
    elif func == 'Himmelblau':
        val = (x**2 + y - 11)**2 + (x + y**2 - 7)**2
    elif func == 'Easom':
        val = -math.cos(x)*math.cos(y)*math.exp(-((x-np.pi)**2 + (y-np.pi)**2))
    elif func == 'Styblinsky-Tang':
        val = sum([x**4 - 16*x**2 + 5*x for x in params])/2
    return [val]


def ComputeLikelihood(data, params, dist):
    '''
    For a specified dataset and parameters, compute the probability
    densities for the observations and also the log likelihood for
    the given parameters.
    :param data: array_like vector of data
    :param dist: name of distribution; the choices and the 
        expected relevant parameters are listed here; even if
        there is only a single parameter for a distribution, it
        should be in a tuple or list:
        NRM - Gaussian: location, scale
        GAM - Gamma: location, scale, shape
        LOG - Lognormal: Location, scale
        EXP - Exponential: scale
        CHI - Chi-Squared: degrees of freedom
        STU - Student's t: degrees of freedom
        CAU - Cauchy: location
        LPL - Laplace: location
        PAR - Pareto: shape
    :return loglike: scalar value for the log likelihood
    :return probs: array_like of the same size of data holding the
        probability densities
    '''

    if dist == 'NRM': # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
        probs = stt.norm.pdf(data, loc=params[0], scale=params[1])
    elif dist == 'GAM': # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html
        probs = stt.gamma.pdf(data, loc=params[0], scale=params[1], a=params[2])
    elif dist == 'LOG': # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html
        probs = stt.lognorm.pdf(data, s=1, loc=params[0], scale=params[1])
    elif dist == 'EXP': # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.expon.html
        probs = stt.expon.pdf(data, scale=params[0])
    elif dist == 'CHI': # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html
        probs = stt.chi2.pdf(data, df=params[0])
    elif dist == 'STU': # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
        probs = stt.t.pdf(data, df=params[0])
    elif dist == 'CAU': # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.cauchy.html
        probs = stt.cauchy.pdf(data, loc=params[0])
    elif dist == 'LPL': # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.laplace.html
        probs = stt.laplace.pdf(data, loc=params[0])
    elif dist == 'PAR': # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pareto.html
        probs = stt.pareto.pdf(data, b=params[0])
    else:
        raise ValueError('%s = Invalid distribution, please see docstring'%dist)
     
    # finally, compute the log-likelihood
    loglike = np.log(probs).sum()
    
    return loglike, probs
    
    
def RegressionMetric(data, subset, estim=None, metric='RMSE', optimGoal=-1):
    '''
    For a given dataset and subset, evalute the subset on the
    dataset, and assess the relationship between the results and the
    target in the data, returning the specified metric. Metric choices
    include 'RMSE', 'MSE', 'MAPE', 'R^2', or you can pass in a callable
    with 'y_true' and 'y_pred' arguments.
    :param data: dataframe of data; the first column should be
        'target', with the rest of the columns the features
    :param subset: array_like of boolean indices for feature inclusion
    :param estim: optional (default = linear Regression) model estimator
    :param metric: optional (default='RMSE') metric to compute
    :param optimGoal: flag indicating what to do with the metric
        (1 = maximize, -1 = minimize); this is only used to put a sign
        on a returned np.inf, in the case of an error
    :return metricVal: single-element tuple holding the value of the
        specified metric from a regression fit between the subset of
        features and the target data column; if an error occurs np.inf is returned
    :return preds: array-like of predictions using specified regression model
    :return estim: fit regression estimator
    '''
    
    # set the estimator if needed
    if estim is None:
        estim = LinearRegression(fit_intercept=False)

    # get the subset of features columns
    feats = data.columns[1:]
    keep = data[[f for b, f in zip(subset, feats) if b]]
    
    # model between target and the tree results
    try:
        estim.fit(X=keep.values, y=data['target'].values)
        preds = estim.predict(X=keep.values)
        # compute the metric
        if metric == 'RMSE':
            metricVal = mean_squared_error(y_true=data['target'].values, y_pred=preds, squared=False)
        elif metric == 'MSE':
            metricVal = mean_squared_error(y_true=data['target'].values, y_pred=preds, squared=True)
        elif metric == 'MAPE':
            metricVal = mean_absolute_percentage_error(y_true=data['target'].values, y_pred=preds)
        elif metric == 'R^2':
            metricVal = r2_score(y_true=data['target'].values, y_pred=preds)
        elif not isinstance(metric, str):
            metricVal = metric(y_true=data['target'].values, y_pred=preds)
    except Exception as err:
        #ipdb.set_trace()
        # don't know, but perhaps an error with the metric function
        print('Unkonwn error: %s'%err)
        preds = [np.nan]*len(data)
        metricVal = np.inf*optimGoal*-1
    
    return (metricVal, preds, estim)


def ClassificationMetric(data, subset, estim=None, metric='accuracy', optimGoal=-1):
    '''
    For a given dataset and subset, evalute the subset on the
    dataset, and assess the classification relationship between
    the results and the target in the data, returning the specified
    metric. Metric choices include 'accuracy', 'F1', 'wF1', 
    or you can pass in a callable with 'y_true' and 'y_pred' arguments.
    :param data: dataframe of data; columns should include
        'target', and 'X0', 'X1', ...
    :param subset: array_like of boolean indices for feature inclusion
    :param estim: optional (default = Decision Tree) model estimator
    :param metric: optional (default='accuracy') metric to compute
    :param optimGoal: flag indicating what to do with the metric
        (1 = maximize, -1 = minimize); this is only used to put a sign
        on a returned np.inf, in the case of an error
    :return metricVal: single-element tuple holding the value of the
        specified metric from a classification model fit between the
        subset of fetures and the target data column; if an error occurs
        np.inf is returned
    :return preds: array-like of predictions using specified classification model
    :return estim: fit classifiction estimator
    '''
    
    # set the estimator if needed
    if estim is None:
        estim = DecisionTreeClassifier(max_depth=5, min_samples_leaf=20)

    # get the subset of features columns
    feats = data.columns[1:]
    keep = data[[f for b, f in zip(subset, feats) if b]]
    
    # model between target and the tree results
    try:
        estim.fit(X=keep.values, y=data['target'].values)
        preds = estim.predict(X=keep.values)
        # compute the metric
        if metric == 'accuracy':
            metricVal = accuracy_score(y_true=data['target'].values, y_pred=preds)
        elif metric == 'F1':
            metricVal = f1_score(y_true=data['target'].values, y_pred=preds)
        elif metric == 'wF1':
            metricVal = f1_score(y_true=data['target'].values, y_pred=preds, average='weighted')
        elif not isinstance(metric, str):
            metricVal = metric(y_true=data['target'].values, y_pred=preds)
    except Exception as err:
        # don't know, but perhaps an error with the metric function
        print('Unkonwn error: %s'%err)
        preds = [np.nan]*len(data)
        metricVal = np.inf*optimGoal*-1
    
    return (metricVal, preds, estim)