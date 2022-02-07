''' This module holds generic utility functions. '''
import math
import numpy as np
import pandas as pd
import time
from itertools import chain
import scipy.stats as stt

import chart_studio.plotly as ply
import chart_studio.tools as plytool
import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly.offline as plyoff
import plotly.subplots as plysub

from sklearn.linear_model import LinearRegression


    
def PDFParamRanges(data, dist, scale=3):
    '''
    Computes, for a specified probability density, the range in which
    the PDF parameters would be expected to be found, given a dataset.
    :param data: array_like vector of data
    :param dist: name of distribution; the choices and their 
        relevant parameters are:
        NRM - Gaussian: location, scale
        GAM - Gamma: location, scale, shape
        LOG - Lognormal: Location, scale
        EXP - Exponential: scale
        CHI - Chi-Squared: degrees of freedom
        STU - Student's t: degrees of freedom
        CAU - Cauchy: location
        LPL - Laplace: location
        PAR - Pareto: shape
    :param scale: integer value to inflate estimated ranges
    :return lb: lower bound of parameters
    :return ub: upper bound of parameters
    '''
    
    # setup some stuff
    n = len(data)
    xbar = np.mean(data)
    s = np.std(data)
    logx = np.log(data)
    inflate = np.array([-scale, scale])
    sqn = math.sqrt(n)

    if dist == 'NRM':
        mnci = xbar + inflate*s/sqn
        lb = [min(mnci), 0.0001]
        ub = [max(mnci), s*1.5]
    elif dist == 'GAM':
        phat = stt.gamma.fit(data)
        phat = phat[1:] + (phat[0],)
        lb = [0.5*p for p in phat]
        ub = [1.5*p for p in phat]
    elif dist == 'LOG':
        mnci = xbar + inflate*s/sqn
        lb = [min(mnci), 0.0001]
        ub = [max(mnci), s*1.5]
    elif dist == 'EXP':
        lambdaci = xbar + inflate*(xbar**2)/sqn
        lb = [max(0, min(lambdaci))]
        ub = [max(lambdaci)]
    elif dist == 'CHI':
        lb = [0]
        ub = [xbar + scale*2*xbar/sqn]
    elif dist == 'STU':
        lb = [0]
        ub = [abs(1.5*(-2*s**2)/(1 - s**2))]
    elif dist == 'CAU':
        mnci = xbar + inflate*s/sqn
        lb = [min(mnci)]
        ub = [max(mnci)]
    elif dist == 'LPL':
        mnci = xbar + inflate*s/sqn
        lb = [min(mnci), 0.0001]
        ub = [max(mnci), s*1.5]
    elif dist == 'PAR':
        phat = stt.pareto.fit(data)[0]
        lb = [0]
        ub = [inflate[1]*phat]
        #ub = [1.5*n/np.sum(np.log(1 + data))]
    else:
        raise ValueError('%s = Invalid distribution, please see docstring'%dist)
        
    return lb, ub


def EncodeBinaryReal(inputType, inputValue, bits, lowerBounds, upperBounds):
    '''
    Using a specified number of bits, and lower & upper real value bounds,
    encode a list of real values as a list binary string, or a list binary
    string as a list of real values.
    :param inputType: 'r' = list of real values input; 'b' = list of
        binary values
    :param inputValue: either a list of n real values, or a single
        list of the binary representation of n real value using the
        number of bits indicated
    :param bits: n-length array_like with number of bits used to encode
        real values
    :param lowerBounds: n-length array_like with lower bound of range
        for real values
    :param upperBounds: n-length array_like with upper bound of range
        for real values
    :return outVal: if inputType is 'r', n*sum(bits)-length list of binary
        values; if inputType is 'b', n-length list of real values
    '''
    
    if inputType == 'b': # binary in, so real out
        # get the limits in the list of the individual values
        binLims = [0]+np.cumsum(bits).tolist()
        # iterate over binary strings
        outVal = [0]*len(bits)
        for indx, (low, hig, bt, lb, ub) in enumerate(zip(binLims[:-1],
            binLims[1:], bits, lowerBounds, upperBounds)):
            # get this real value's binary representation
            binV = inputValue[low:hig]
            # get the powers of 2 & max value
            exps = [2**b for b in range(bt-1,-1, -1)]
            mx = 2**len(bits)
            # compute the real value
            realV = sum([b*e for (b, e) in zip(binV, exps)])
            realV = lb + (ub - lb)*realV/mx
            outVal[indx] = realV
    elif inputType == 'r': # real in, so binary out
        # iterate over real values
        reBinVal = [None]*len(inputValue)
        for indx, (realV, bt, lb, ub) in enumerate(zip(inputValue, bits, lowerBounds, upperBounds)):
            # stepsize in range
            steps = (ub - lb)/(2**bt-1)
            # values distance from lower
            dist = int((realV - lb)/steps)
            # encode the distance into binary
            binV = [int(b) for b in bin(int(dist))[2:].zfill(bt)]
            reBinVal[indx] = binV
        outVal = list(chain.from_iterable(reBinVal))
    else:
        raise TypeError('Input type may only be "b" for "r"')
    
    return outVal


def BinaryStr(subset):
		'''
		Make a nice binary string of the form '1010' from an input
		array_like of binary values.
		:param subset: array_like of binary values
		:return strbinary: string of the binary values
		'''
		return ''.join([str(int(flg)) for flg in subset])


def RandomWeightedSelect(keys, wats, randSeed=None):
    '''
    Randomly select an item from a list, according to a set of
    specified weights.
    :param keys: array-like of items from which to select
    :param wats: array-like of weights associated with the input
        keys; must be sorted in descending weight
    :param randSeed: optional random seed for np.random; if no
        randomization is desired, pass 0
    :return selection: selected item
    :return randSeed: random seed used
    '''
    
    # ranodmize, perhaps
    if randSeed != 0:
        if randSeed is None:
            randSeed = int(str(time.time()).split('.')[1])
        np.random.seed(randSeed)
    
    # ensure weights sum to 1
    totWats = sum(wats)
    if totWats != 1:
        wats = [v/totWats for v in wats]
    
    # get the cumulative weights
    cumWats = np.cumsum(wats)
    # get the indices of where the random [0,1] is < the cum weight
    rnd = np.random.rand()
    seld = rnd < cumWats
    
    return [k for (k,s) in zip(keys, seld) if s][0], randSeed


def ResultsPlots(data, sequenceCol, responseCol, predCol, resdCol, colorCol, overall_title, plot_colors=('red',)*4):
    '''
    This creates and returns a quad-plot of results from a model. The four plots are: a) prediction vs response,
    b) histogram of residuals, c) residuals by sequence, d) residuals by response. They are arranged as
    [[a,b],[c,d]].
    :param data: dataframe holding the sequence, response, prediction, and residual columns
    :param sequenceCol: column in the dataframe holding the time or sequence counter; if None, a counter will be used
    :param responseCol: column in the dataframe holding the response variable
    :param predCol: column in the dataframe holding the model predictions
    :param resdCol: column in the dataframe holding the model residuals
    :param colorCol: optional column in the dataframe holding the color for each observation for the plots
    :param overall_title: title to go on top of the quad plot
    :param plot_colors (default=(red, red, red, red)): optional tuple of colors for each plot; only used if colorCol
        not passed
    :return fig: plotly plot figure
    '''
    
    # copy the input dataframe
    data = data.copy(deep=True)
    
    if colorCol is None:
        colorCol = 'NOCOLOR'
    
    # setup the subplot
    figRes = plysub.make_subplots(rows=2, cols=2, subplot_titles=['Predictions vs Responses', 'Residuals Distribution','Residuals by Sequence', 'Residuals vs Responses'])
    
    # for the actual vs preds plot, build the fit line
    actual = data[responseCol].values.reshape(-1,1)
    lindat = np.linspace(actual.min(), actual.max(), 10).reshape(-1, 1)
    fitlin = LinearRegression(n_jobs=-1)
    fitlin.fit(X=actual, y=data[predCol])
    actpred = fitlin.predict(X=lindat)
    r2 = np.corrcoef(actual.squeeze(), data[predCol].values)[0][1]
    # create the trace and annotation
    r2trc = go.Scatter(x=lindat.squeeze(), y=actpred, mode='lines', name='fit', line={'color':'black','width':1}, showlegend=False)
    r2ann = dict(x=lindat[5][0], y=actpred[5], xref='x1', yref='y1', text='$\\rho=%0.3f$'%(r2), showarrow=False, bgcolor='#ffffff')
    
    # actuals vs resids plot
    if colorCol == 'NOCOLOR':
        data[colorCol] = plot_colors[0]
    figRes.add_trace(go.Scatter(x=data[responseCol], y=data[predCol], mode='markers', marker={'color':data[colorCol]}, showlegend=False), 1,1)
    figRes.add_trace(r2trc, 1, 1)
    figRes['layout']['xaxis1'].update(title=responseCol)
    figRes['layout']['yaxis1'].update(title=predCol)
    
    # residuals histogram
    if colorCol == 'NOCOLOR':
        data[colorCol] = plot_colors[1]
    figRes.add_trace(go.Histogram(x=data[resdCol], histnorm='', marker={'color':data[colorCol]}, showlegend=False), 1,2)
    figRes['layout']['xaxis2'].update(title=resdCol)
    figRes['layout']['yaxis2'].update(title='count')
    
    # get the time variable
    if sequenceCol is None:
        seq = list(range(len(data)))
        seqNam = 'sequence'
    else:
        seq = data[sequenceCol]
        seqNam = sequenceCol
    
    # residuals by time plot
    if colorCol == 'NOCOLOR':
        data[colorCol] = plot_colors[2]
    figRes.add_trace(go.Scatter(x=seq, y=data[resdCol], mode='markers', marker={'color':data[colorCol]}, showlegend=False), 2, 1)
    figRes['layout']['xaxis3'].update(title=seqNam)
    figRes['layout']['yaxis3'].update(title=resdCol)
    
    # residuals by response plot
    if colorCol == 'NOCOLOR':
        data[colorCol] = plot_colors[3]
    figRes.add_trace(go.Scatter(x=data[responseCol], y=data[resdCol], mode='markers', marker={'color':data[colorCol]}, showlegend=False), 2,2)
    figRes['layout']['xaxis4'].update(title=responseCol)
    figRes['layout']['yaxis4'].update(title=resdCol)
    
    # update layout
    figRes['layout'].update(title=overall_title, height=1000)
    anns = list(figRes['layout']['annotations'])
    anns.append(r2ann)
    figRes['layout']['annotations'] = anns
    
    return figRes


def correlationsPlot(rhoYourBoat, plotTitl='Feature Correlations Plot', trcLims=(0.0, 1.0), tweaks=(20, None, None, 1.1)):
    '''
    This creates and returns a bubble plot visualization of a correlation matrix. The
    sizes of the bubbles are proportional to the absolute magnitude of the correlations.
    Positive correlations are only plotted in the upper triangle, with colors ranging
    from green (0) to red (+1). Negative correlations are plotted in the lower triangle,
    with colors ranging from green (0) to blue (-1). Perfect correlations are indicated
    with black bubbles.
    :param rhoYourBoat: dataframe of correlation matrix, probably created with pd.DataFrame.corr()
    :param plotTitl: optional (default='Feature Correlations Plot') plot title
    :param trcLims: (default=(0.0,1.0)) = ordered tuple of "buckets" in which to place absolute correlations for
        plotting traces; must include at least 0 and 1
    :param tweaks (default=(20,None,None,1.1)): tuple of position & size tweaking values for plotly; maximum size of
        bubbles, plot width, plot height, y position of legend
    :return fig: plotly correlation plot figure
    '''

    # set the granulatrity of the colors
    n = 101  # must be odd so in the middle at correlation = 0 is just green

    # number features
    p = len(rhoYourBoat.columns)
    ps = list(range(p))

    # positive correltions are red>green
    scl = np.linspace(1.0, 0.0, n)
    redsP = np.round(255 * scl)
    grnsP = 255 - redsP
    blusP = [0.0] * n

    # negative correlations are blue>green
    scl = scl[:-1]
    blusN = np.round(255 * scl)
    grnsN = 255 - blusN
    redsN = [0.0] * n

    # adding 2 more to make the endpoints for perfect correlations
    scl = np.linspace(-1.0, 1.0, 2 * n - 1 + 2)

    # make the colormap - perfectly uncorrelated and perfectly correlated are black
    rgb = ['rgb(0,0,0)']
    rgb.extend(['rgb(%d,%d,%d)' % (r, g, b) for r, g, b in
                zip(np.r_[redsN, redsP[::-1]], np.r_[grnsN, grnsP[::-1]], np.r_[blusN, blusP[::-1]])])
    rgb.append('rgb(0,0,0)')

    # now map correlations to colors - unhappy that I have to do this double loop :-(
    vals = rhoYourBoat.values
    cols = np.zeros(shape=vals.shape, dtype=object)
    for i in ps:
        for j in ps:
            v = vals[i, j]
            mni = np.argmin(np.abs(v - scl))
            mnv = scl[mni]
            cols[i, j] = rgb[mni]
            # print('%0.5f,%d,%0.5f,%s'%(v,mni,mnv,cols[i,j]))

    # filter data so the upper triangle is (+) correlations and lower triangle is (-) correlations
    y = np.tile(ps, (p, 1))
    x = y.T
    x = x.flatten()
    y = y.flatten()
    vals = vals.flatten()
    cols = cols.flatten()
    keepind = ((y > x) & (vals > 0)) | ((x > y) & (vals < 0))
    x = x[keepind]
    y = y[keepind]
    vals = vals[keepind]
    cols = cols[keepind]
    absVals = np.abs(vals)

    # set a minimum bubble size
    minBub = 0.09 * tweaks[0]

    # put together the figure - make multiple traces
    trc = [go.Scatter(
        {'x': ps, 'y': ps, 'mode': 'lines', 'line': {'color': 'black'}, 'showlegend': False, 'hoverinfo': 'skip'})]
    for i, t in enumerate(trcLims):
        # build the index for the traces
        if i == 0:
            continue
        elif i == 1:
            indx = (absVals <= t) & (absVals >= trcLims[0])
            trcName = '$\\vert\\rho\\vert\\in[%0.2f,%0.2f]$' % (trcLims[0], t)
        else:
            indx = (absVals <= t) & (absVals > trcLims[i - 1])
            trcName = '$\\vert\\rho\\vert\\in(%0.2f,%0.2f]$' % (trcLims[i - 1], t)
        # create & add the trace
        trc.append(go.Scatter({'x': x[indx], 'y': y[indx], 'mode': 'markers', 'text': ['%0.4f' % v for v in vals[indx]],
                               'name': trcName, 'hoverinfo': 'x+y+text',
                               'marker': {'color': cols[indx], 'line': {'color': cols[indx]},
                                          'size': np.maximum(tweaks[0] * absVals[indx], minBub)}}))

    # finalize the layout
    lout = go.Layout({'title': plotTitl, 'legend': {'orientation': 'h', 'x': 0, 'y': tweaks[-1]},
                      'xaxis': {'ticklen': 1, 'tickvals': ps, 'ticktext': rhoYourBoat.columns.values,
                                'mirror': True, 'showgrid': False, 'range': [-1, p], 'linecolor': 'black',
                                'linewidth': 0.5, 'zeroline': False, 'tickangle': 90},
                      'yaxis': {'ticklen': 1, 'tickvals': ps, 'ticktext': rhoYourBoat.index.values,
                                'mirror': True, 'showgrid': False, 'range': [-1, p], 'linecolor': 'black',
                                'linewidth': 0.5, 'zeroline': False}})
    if tweaks[1] is not None:
        lout['width'] = tweaks[1]
    if tweaks[2] is not None:
        lout['height'] = tweaks[2]

    return go.Figure(data=trc, layout=lout) 
