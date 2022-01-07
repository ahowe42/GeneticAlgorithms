''' This module holds a runner and operation functions for a feature selection genetic algorithm. '''
import numpy as np
import pandas as pd
import datetime as dt
import ipdb
import time
from itertools import chain
import copy
import sys

import chart_studio.plotly as ply
import chart_studio.tools as plytool
import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly.offline as plyoff
import plotly.subplots as plysub

sys.path.append('../')
from Utils.Utils import *

pd.set_option('display.max_columns', None)



def OP_GAEngineering(currBest, prevBest, population, probEngineer):
    '''
    This implements GA engineering, the point of which is to limit the
    variability between GA runs. It takes the best solution from the
    previous generation and the best solution from the current generation,
    and finds the difference. Where they are different, the bits from the
    previous best are inserted into the offspring of the current generation
    with the specified probability. This should only be called if the previous
    best is better than the current best.
    :param currBest: p-length array_like holding the best solution from the
        current generation
    :param prevBest: p-length array_like holding the best solution from the
        previous generation
    :param population: (n,p) array of n GA solutions of size p each; this is
        the offspring for the next generation
    probEngineer: scalar float probability of GA engineering (in range [0,1])
    offspring: (n,p) population with bits engineered
    '''

    # duck arrays
    currBest = np.array(currBest, ndmin=1, dtype=bool, copy=False).flatten()
    prevBest = np.array(prevBest, ndmin=1, dtype=bool, copy=False).flatten()
    population = np.array(population, ndmin=2, dtype=bool, copy=False)
    (n,p) = population.shape

    # first get where they are different and the actual different values
    difflocs = np.logical_xor(currBest, prevBest)
    diffvals = prevBest[difflocs]
    # if no differences, just exit
    if np.sum(difflocs) == 0:
      return population

    # generate the randoms to decide who gets engineered
    engme = engineer_rate > np.random.rand(n)
    # if none selected to engineer, just exit
    if np.sum(engme) == 0:
      return population

    # GA engineer - population[engme, difflocs] = diffvals doesn't work :-(
    newPop = np.zeros((n, p), dtype=bool)
    rows = np.arange(n)
    # first build the output by adding the population elements that won't be changed
    rows_nochg = rows[~engme]
    newPop[rows_nochg, :] = population[~engme, :]
    # now edit the rest of the population
    jnk = population[engme, :]
    jnk[:, difflocs] = diffvals
    newPop[rows[engme], :] = jnk

    return newPop
  
  
def OP_Crossover(population, parents, probXover):
    '''
    Performs crossover on the current generation of a GA after the solutions
    have been selected and paired for mating. Crossover is selected to occur
    with (probXover)% probability - if no crossover, the results are genetic
    replicates. For single-and double-point crossover, the points are selected
    uniformly randomly.
    :param population: (n,p) bool array of n GA solutions of size p each
    :param parents: (n/2,2) array indicating pairs of solutions to mate; if n is odd,
        parents is of size ((n-1)/2,2)
    :param probXover: scalar float probability of crossover (in range [0,1])
    :return newPop: (n,p) array of next generation's population
    '''
    
    # population and parent arrays: duck you! JAH 20120920
    population = np.array(population, ndmin=2, dtype=bool, copy=False)
    parents = np.array(parents, ndmin=2, copy=False)

    # now get the dimensions JAH 20120920
    (n,p) = parents.shape
    matepairs = parents.shape[0] # number couples, should be n/2

    offspring1 = np.zeros((matepairs, p))
    offspring2 = offspring1.copy()
    # perform the crossovers (maybe)
    if xover_type == 1:     # single-point
      for matecnt in range(matepairs):
        dad = population[parents[matecnt,0],:]
        mom = population[parents[matecnt,1],:]
        if xover_rate > np.random.rand():     # crossover
          # point randomly selected - endpoints allowed (since :p excludes p)
          xoverpoint = np.random.random_integers(1,p-1)
          offspring1[matecnt,:] = np.concatenate((dad[:xoverpoint], mom[xoverpoint:]))
          offspring2[matecnt,:] = np.concatenate((mom[:xoverpoint], dad[xoverpoint:]))
        else:              # genetic replication
          offspring1[matecnt,:] = dad.copy()
          offspring2[matecnt,:] = mom.copy()
    elif xover_type == 2:   # double-point
      for matecnt in range(matepairs):
        dad = population[parents[matecnt,0],:]
        mom = population[parents[matecnt,1],:]
        if xover_rate > np.random.rand():     # crossover
          # select 2 points randomly without replacement in inclusive range [1,p-2]
          xoverpoints = np.sort(np.random.permutation(p-1)[:2]+1)
          offspring1[matecnt,:] = np.concatenate((dad[:xoverpoints[0]],
            mom[xoverpoints[0]:xoverpoints[1]],dad[xoverpoints[1]:]))
          offspring2[matecnt,:] = np.concatenate((mom[:xoverpoints[0]],
            dad[xoverpoints[0]:xoverpoints[1]],mom[xoverpoints[1]:]))
        else:                           # genetic replication
          offspring1[matecnt,:] = dad.copy()
          offspring2[matecnt,:] = mom.copy()
    elif xover_type == 3:   # uniform
      for matecnt in range(matepairs):
        dad = population[parents[matecnt,0],:]
        mom = population[parents[matecnt,1],:]
        if xover_rate > np.random.rand():     # crossover
          xoverpoints = xover_rate > np.random.rand(p)
          offspring1[matecnt,:] = dad*xoverpoints + mom*~xoverpoints
          offspring2[matecnt,:] = dad*~xoverpoints + mom*xoverpoints
        else:                           # genetic replictaion
          offspring1[matecnt,:] = dad.copy()
          offspring2[matecnt,:] = mom.copy()

    return np.vstack((offspring1,offspring2)) == 1


def OP_Mutate(population, prob):
    '''
    Perform random mutation or pruning on a population; this would usually
    be performed after the previous generation has mated and produced the
    next generation.
    :param population: (n,p) array of n GA solutions of size p each
    :param prob: scalar float probability of mutation or pruning (in range [0,1])
    :return newPop: (n,p) array of mutated population
    '''
    
    # ducktype population
    population = np.array(population, ndmin=2, dtype=bool, copy=False)
    (n,p) = population.shape

    # get the index of the elements that will mutate
    mutators = (mutat_rate > np.random.rand(n,p))
    # now prepare the mutated population
    newPop = population.copy()
    newPop[mutators] = ~(newPop[mutators])

    return newPop


def OP_MateSelect(popFitness, optimGoal, meth):
    '''    
    Generate an index array into the population showing which members to mate.
    If the population size is uneven, the solutions will all be mated as instructed,
    but then at the end, one of the mating pairs will be randomly culled from mating.
    The random selection is proportional to the pairs' average fitnesses.
    :param popFitness: (n,) data array_like of fitness scores from a population of size n
    :param optimGoal: 1 = maximize, -1 = minimize
    :param meth: 1 = sorted, 2 = roulette
    :return parents: (n/2,2) array indicating pairs of solutions to mate; if n is odd,
        parents is of size ((n-1)/2,2)
    '''
    
    # duck pop_fitness into 1d array JAH 20120920
    popFitness = np.array(popFitness, ndmin=1, copy=False)
    populSize = popFitness.size

    # sort fitness scores, if optimGoal is positive, this will sort the scores
    # descending, with the lowest at the front, if optimGoal is negative, it is
    # essentially sorting ascending, with the largest at the front, either way,
    # the best chromosomes are associated with largest roulette bins
    stdIndex = np.argsort(popFitness*-optimGoal)

    # do the work
    if meth == 1:
        # simply mate pairwise
        if populSize % 2 == 1:
            # odd number, so reinsert the best; we want to insert as 3rd so best doesn't mate with self
            stdIndex = np.insert(stdIndex, 2, stdIndex[0])
            populSize += 1
        parents = np.reshape(stdIndex, (populSize//2, 2))
    else:
        # roulette method
        # prepare bins for roulette - bigger bins at the beginning with lower scores
        bins = np.cumsum(np.linspace(populSize, 1, populSize)/(populSize*(populSize + 1.0)/2))
        # first n random numbers to each bin to find each bin that is a lower bound for each rand
        rands_in_bins = np.repeat(np.random.rand(populSize), populSize) >= np.tile(bins, populSize)
        # summing all lower bound flags for each random gives the bin it falls into (since 0-based)
        newPop = np.sum(np.reshape(rands_in_bins, [populSize]*2), axis=1)
        # now index into the stdindex to get parents
        parents = stdIndex[newPop]
        # odd number, so reinsert the best; will randomly permute order, so doesn't matter where
        if populSize % 2 == 1:
            parents = np.insert(parents, 0, stdIndex[0])
            populSize += 1
        # randomly resort then pair up
        parents = np.reshape(parents[np.random.permutation(populSize)], (populSize/2, 2))

        # 20160225 JAH don't want an uneven population size (often from Elitism) to result in
        # such fast population growth, so randomly cull one mating pair, with frequency relative
        # to their average scores (worst avg score is most likely to be culled)
        if popSitness.size % 2 == 1:
            # compute the parent's avg scores, then get the best<>worst sorted 1-based indexes
            srtAvgScs = np.argsort(np.sum(popFitness[parents], axis = 1)*(-optimGoal))+1
            # create the (0,1] bin upper bounds
            cumProbs = np.cumsum(srtAvgScs/np.sum(srtAvgScs))
            # pick which mating pair is culled - it's the last bin upper bound that's <= the random
            cullMe = (srtAvgScs[np.random.rand() <= cumProbs])[0] - 1
            parents = np.hstack((parents[:cullMe], parents[(cullMe+1):]))
            
    return parents


def RunGASubset(params, data, objective, seedSubs=[], verbose=False, randSeed=None):
    '''
    Run the genetic algorithm for a subset selection on a specified dataset.
    Parameters for the objective function must be passed, along with their names
    in the objective. While only params['showTopSubs'] results will be displayed,
    the unique best solutions from all generations will be returned.
    :param params: dictionary of GA parameters:
        'initPerc': float percent of initial population filled (features selected)
         'forceVars': array_like of feature index numbers up to p-1 to force to stay
            in all solutions
        'showTopSubs': integer number of best solutions to show
        'populSize': integer population size
        'numGens': integer number of generations
        'noChangeTerm': integer number generations with insufficient
            improvement before early termination
        'convgcrit': float convergence criteria
        'elitism': True = on, False = off
        'matetype': 1 = sorted, 2 = roulette
        'probXover': float probability of crossover
        'probMutate': float probability of mutation
        'probEngineer': float probability of GA engineering
        'optimGoal': 1 = maximize, -1 = minimize
        'plotFlag': True = on, False = off
        'printFreq': integer number of generations by which the GA will print
            progress
    :param data: dictionary expected to hold two items:
        'data': pandas dataframe of data
        'name': name (descriptive or perhaps filename) of data
    :param objective: dictionary of objective function parameters:
        'function': string function to execute whatever modeling is required and
            return the objective score; it can return multiple items, but the
            first must be the score; if a single item is returned, it should be
            in a tuple
        'arguments': dictionary of arguments to pass to the objective function;
            should include at least 'data' and 'subset'
    :param seedSubs: optional array of p-length binary vectors with which
        to seed the initial population
    :param verbose: optional (default = false) flag to print extra info
    :param randSeed: optional (default = none) seed for randomizer; if not passed,
        this will be generated and printed
    :return bestSubset: the best subset overall
    :return bestScore: the score of the best overall subset
    :return genBest: array of the best subset from each generation
    :return genScores: array of the best score and the average (of finite) scores
        from each generation
    :return randSeed: the random seed used
    :return tstamp: string timestamp of the GA run
    :return fig: plotly figure of the GA progress
    '''

    # start time and timestamp
    stt = dt.datetime.now()
    sttT = time.perf_counter()
    tstamp = re.sub('[^0-9]', '', stt.isoformat()[:19])
    
    # parse GA parameters
    populSize = int(params['populSize'])
    numGens = int(params['numGens'])
    noChangeTerm = int(params['noChangeTerm'])
    convgCrit = float(params['convgCrit'])
    elitism = bool(params['elitism'])
    probXover = float(params['probXover'])
    mateType = int(params['mateType'])
    probMutate = float(params['probMutate'])
    optimGoal = int(params['optimGoal'])
    plotFlag = bool(params['plotFlag'])
    printFreq = int(params['printFreq'])
    showTopSubs = int(params['showTopSubs'])
    initPerc = float(params['initPerc'])
    forceVars = float(params['forceVars'])
    probEngineer = float(params['probEngineer'])
    
    # parse the data
    dataName = data['name']
    data = data['data']
    n, p = data.shape
    # these are used later on
    varis = np.arange(p) + 1
    bin_to_dec = 2**(varis - 1)
    
    # parse the objective
    objFunc = objective['function']
    objArgs = objective['arguments'] 
    objArgs['data'] = data
    objStr = '%s(%s)'%(objFunc, ', '.join(['%s=%r'%(key, val) for (key, val) in objArgs.items()\
        if key not in ['data', 'subset']]))
    
    # set the random state
    if randSeed is None:
        randSeed = int(str(time.time()).split('.')[1])
        print('Random Seed = %d'%randSeed)
    np.random.seed(randSeed)
        
    # display parameters
    dispLine = '#'*42
    print('%s\nGA Started on %s\n%s'%(dispLine, stt.isoformat(), dispLine))
    print('Data: %s(n=%d, p=%d)'%(dataName, n, p))
    print('Random Seed: %d'%randSeed)
    print('Maximum # Generations: %d\nMininum # of Generations: %d\nConvergence Criteria: %0.8f'%(numGens,noChangeTerm,convgCrit))
    if populSize % 2 == 1:
        populSize += 1
        print('!!Population Size Increased By 1 to be Even!!')
    print('Population Size: %d'%populSize)
    print('Initial Fill Percentage: %0.2f'%initPerc)
    print('Features Forced in all Models: %r'%forceVars)
    print('Initial Population Seeded with %d Subsets'%len(seedSubs))
    print('Mutation Rate: %0.2f\nCrossover Rate: %0.2f'%(probMutate, probXover))
    print('Mating Method: %s'%['SORTED','ROULETTE'][mateType - 1])
    print('Elitism is: %s'%['OFF','ON'][elitism])
    if (elitism) and (probEngineer > 0.0):
      print('!!With Elitism ON, the probability of GA engineering has been set to 0.00!!')
      probEngineer == 0.0
    else:
      print('GA Engineering Rate: %0.2f'%probEngineer)
    print(dispLine)
    if optimGoal == 1:
        print('Objective: MAXIMIZE')
    else:
        print('Objective: MINIMIZE')
    print('Objective Function: %s'%objStr)
    print(dispLine)
    
    ''' setup first population '''
    # initialize the population with initPerc% 1s, then ...
    population = (initPerc >= np.random.rand(popul_size, p))
    # ... force everything in force_vars to be included
    if forceVars is not None:
      population[:, forceVars] = True    
    # ... then add any seed solutions to the end ...
    if seedSubs is not None:
      seedSubs = np.array(seedSubs, ndmin=2, copy=False)
      scnt = min(seedSubs.shape[0], popul_size)
      population[-scnt:,v:] = seedSubs
    # ... but find any solutions that are entirely 0 and replace them
    allZero = np.where(np.sum(population, axis=1)==0)[0]
    population[allZero, np.random.randint(0, p, len(allZero))] = True
    # talk
    if verbose:
        for indx in range(populSize):
            print('%0d\n%r'%(indx, population[indx, :]))
    
    # now initialize more things
    # save results by generation
    genScores = np.zeros((numGens, 2), dtype=float)
    genBest = np.zeros((numGens ,p), dtype=bool)
    # current generation's best
    bestSubset = np.zeros((1, p), dtype=bool)
    bestScore = optimGoal*-1*np.Inf
    # previous generation's best - used for GA Engineering
    prevGenBestSubset = bestSubset.copy()
    prevGenBestScore = bestScore
    # generations with no improvement termination counter
    termCount = 0
    
    # Begin GA Algorithm Whoo Hoo!
    for genCnt in range(numGens):        
        ''' compute or lookup objective function values '''
        popFitness = np.ones(populSize, dtype=float)*np.Inf
        for popCnt in range(populSize):
            if genCnt > 0:
                # check if this subset already evaluated, to save time
                prevEval = np.nonzero(population[popCnt].function == allSubsets)[0]
                if prevEval.size == 0:
                    # evalaute
                    objArgs['subset'] = population[popCnt].function
                    popFitness[popCnt] = globals()[objFunc](**objArgs)[0]
                else:
                    # look up existing score
                    popFitness[popCnt] = allScores[prevEval]
            else:
                objArgs['subset'] = population[popCnt].function
                popFitness[popCnt] = globals()[objFunc](**objArgs)[0]
                
        # If optimGoal is (+), this will not change the scores, so the true max will be taken.
        # If optimGoal is (-), the signs will all be changed, and since max(X) = min(-X),
        # taking the maximum will really be taking the minimum.
        optInd = np.argmax(optimGoal*popFitness)
        optVal = popFitness[optInd]
        
        # save some stuff before moving along
        genScores[genCnt,:] = (optVal, np.mean(popFitness[np.isfinite(popFitness)]))
        genBest[genCnt] = population[optInd]
        
        ''' save all unique subsets & their scores '''
        # must first convert population to decimal representation so can unique
        tmp = np.sum(population*bin_to_dec, axis=1)
        # now can get the indices of the unique values; yes, I know this sorts them first - and I don't really care
        _, ind = np.unique(tmp, return_index=True)
        if genCnt == 0:
            # first generation, so create these arrays here
            allScores = popFitness[ind]
            allSubsets = population[ind, :]
        else:
            # not first generation, so append to the existing arrays
            allScores = np.append(allScores, popFitness[ind])
            allSubsets = np.vstack((allchroms, population[ind,:]))
        # now just get the unique subsets
        uniq, ind = np.unique(allSubsets, return_index=True)
        allScores = allScores[ind]
        allSubsets = uniq
        
        ''' check for early termination '''
        if optimGoal*genScores[genCnt, 0] > optimGoal*bestScore:
            # this is a better score, so save it and reset the counter
            bestScore = genScores[genCnt, 0]
            bestSubset = genBest[genCnt]
            termCount = 1
        #elif (optimGoal*genScores[genCnt, 0] < optimGoal*bestScore) and (elitism == False):
        #    # if elitism is off, we can still do early termination with this
        #    termcount += 1
        elif abs(optimGoal*genScores[genCnt, 0] - optimGoal*bestScore) < convgCrit:
            # "no" improvement
            termCount += 1
        elif (elitism  == True):
            # with elitism on and a deterministic objective, performance is monotonically non-decreasing
            termCount += 1

        if termCount >= noChangeTerm:
            print('Early Termination On Generation %d of %d'%(genCnt + 1, numGens))
            genScores = genScores[:(genCnt + 1), :] # keep only up to genCnt spaces (inclusive)
            genBest = genBest[:(genCnt + 1)]
            break
            
        # don't bother with the next generation
        if genCnt == (numGens - 1):
            break
    
        ''' create the next generation '''
        # select parents for the next generation
        parents = OP_MateSelect(popFitness, optimGoal, mateType)
        # crossover
        newPop = OP_Crossover(population, parents, probXover)
        # mutation
        newPop = OP_Mutate(newPop, probMutate)
        # engineering
        if probEngineer > 0:
          # I check for probEngineer > 0 because I can turn this off by
          # setting it to 0. Below we use genscores and gencnt, because
          # bestScore and bestSubset won't hold the current best if the
          # current best solution is worse than the overall best, and elitism
          # is off.
          if (gencnt > 0) and (optim_goal*prevGenBestScore > optim_goal*bestScore):
            # only call GAengineering if the previous generation best is better
            newPop = OP_GAEngineering(genBest[gencnt,:], prevGenBestSubset, newPop, probEngineer)
            prevGenBestSubset = bestSubset
            prevGenBestScore = bestScore
        # fix all-zero chromosomes
        allZero = np.where(np.sum(newPop, axis=1)==0)[0]
        newPop[allZero, np.random.randint(0, p, len(allZero))] = True

        ''' finalize new population & convey best individual into it if not already there '''
        if elitism:
          # check if best is currently in new_pop
          if np.where(np.sum(newPop == bestSubset, axis=1)==p)[0].size == 0:
            population = np.vstack((newPop, bestSubset))
          else:
            population = newPop.copy()
        else:
          population = newPop.copy()
        # readjust in case population grew
        populSize = len(newPop)
    
        # talk, maybe
        if genCnt % printFreq == 0:
            print('Generation %d of %d: Best Score = %0.4f, Early Termination = %d\n\t%s'%\
                (genCnt + 1, numGens, bestScore, termCount, BinaryStr(bestSubset)))
            
    # Finish GP Algorithm Whoo Hoo!
    print('Generation %d of %d: Best Score = %0.4f, Early Termination = %d\n\t%s'%\
        (genCnt + 1, numGens, bestScore, termCount, BinaryStr(bestSubset)))
    
    ''' plot GP progress '''
    fig = plysub.make_subplots(rows=2, cols=1, print_grid=False, subplot_titles=['Best Score', 'Average Score'])
    # build traces
    gens = len(genBest)
    xs = list(range(gens))
    fig.add_trace(go.Scatter(x=xs, y=genScores[:,0], mode='markers+lines', name='Best Score',
        text=['%s = %0.5f'%(BinaryStr(subset), score) for (subset, score) in zip(genBest, genScores[:,0])]), 1, 1)
    fig.add_trace(go.Scatter(x=xs, y=genScores[:,1], mode='markers+lines', name='Average Score'), 2, 1)
    # annotate the best solution
    bestAnn = dict(x=gens-1, y=np.min(genScores[:,0]), xref='x1', yref='y1', text='%s = %0.5f'%(BinaryStr(bestSubset), bestScore),
                   showarrow=True, bordercolor="#c7c7c7", borderwidth=2, borderpad=4, bgcolor="#6d72f1", opacity=0.8,
                   font={'color':'#ffffff'}, align="center", arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#636363")
    # update layout
    anns = list(fig['layout']['annotations'])
    anns.append(bestAnn)    
    fig.update_layout(title='GA Progress Resuls by Generation (%s, %s, %s)'%(dataName, objStr, tstamp),
        annotations=anns)
    fig['layout']['xaxis3'].update(title='Generation')
    if plotFlag:
        plyoff.plot(fig, filename='../output/GAProgress_%s_%s_%s.html'%(tstamp, re.sub('[^0-9A-Za-z_]', '_', dataName), objFunc),
            auto_open=True, include_mathjax='cdn')
    
    ''' summarize results: GA_BEST '''
    # must first convert generation best solutions to decimal representation so can unique
    genBestDec = np.sum(genBest*bin_to_dec, axis=1)
    # now can get the indices of the unique values; yes, I know this sorts them first - and I don't really care
    _, ind = np.unique(genBestDec, return_index=True)
    # combine the unique best scores and solutions from each generation
    GA_BEST = np.hstack((genScores[ind, 0], genBest[ind, :]))
    # now sort so best is at top
    GA_BEST = GA_BEST[np.argsort(-1*optim_goal*GA_BEST[:, 0]), :]   
    # build the results dataframe
    indx = [BinaryStr(subset) for subset in GA_BEST[:, 1:]]
    GA_BEST = pd.DataFrame(index=indx, data=GA_BEST, columns=['Score']+data.columns.tolist())
    # compute the subset bests frequencies, add them in
    freqs = pd.DataFrame(data=GA_BEST.index.value_counts()/len(genBest), columns=['Frequency'])
    GA_BEST = GA_BEST.join(freqs)
    # drop duplicates
    GA_BEST = GA_BEST.drop_duplicates()
    
    # show results
    print('%s\nGA Complete\n\tUnique Subsets Evaluated - %d\nTop %d Solutions'%(dispLine, len(allScores), showTopSubs))
    display(GA_BEST.head(showTopSubs))
    
    # stop time
    stp = dt.datetime.now()
    stpT = time.perf_counter()
    print('GA: Started on %s\n\tFinished on %s\n\tElapsed Time = %0.3f(m)'%(stt.isoformat(), stp.isoformat(), (stpT-sttT)/60))
    
    return bestSubset, bestScore, genBest, genScores, randSeed, tstamp, fig