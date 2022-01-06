''' This module holds a runner and operation functions for a genetic programming algorithm. '''
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

from sklearn.linear_model import LinearRegression

sys.path.append('../')
from GP.FunctionTree import *
from GP.Objective import *
from util.Utils import *

pd.set_option('display.max_columns', None)



def TreesCrossover(this, that, verbose=False):
    '''
    Cross two trees at a node selected at random.
    :param this: first tree to cross
    :param that: second tree to cross
    :param verbose: optional (default=False) flag to print the crossover node
    :return this_new: new crossed-over tree
    :return that_new: new crossed-over tree
    '''
    
    # first create copies of the input trees
    thisC = copy.deepcopy(this)
    thatC = copy.deepcopy(that)
    
    # get the random crossover points
    thisXoverNode = np.random.permutation(list(chain.from_iterable(thisC.struct.values())))[0]
    if verbose:
        print('This crossover point: %s '%thisXoverNode)
    thatXoverNode = np.random.permutation(list(chain.from_iterable(thatC.struct.values())))[0]
    if verbose:
        print('That crossover point: %s '%thatXoverNode)
    
    # reassign the children
    try:
        if thisXoverNode.parent.right == thisXoverNode:
            thisXoverNode.parent.setRight(thatXoverNode)
        elif thisXoverNode.parent.left == thisXoverNode:
            thisXoverNode.parent.setLeft(thatXoverNode)
    except AttributeError:
        # this is a root node, so there is no parent, so no left or right
        pass
    try:    
        if thatXoverNode.parent.right == thatXoverNode:
            thatXoverNode.parent.setRight(thisXoverNode)
        elif thatXoverNode.parent.left == thatXoverNode:
            thatXoverNode.parent.setLeft(thisXoverNode)
    except AttributeError:
        # that is a root node, so there is no parent, so no left or right
        pass
    
    # reassign the parents
    thisXoverNode.parent, thatXoverNode.parent = thatXoverNode.parent, thisXoverNode.parent
    
    # if either is an orphan from having been crossed with a root, make it a new tree;
    # otherwise, just rebuild the structure
    if thisXoverNode.parent is None:
        thatC = Tree(thisXoverNode, thisC.depth + thatC.depth)
    else:
        thatC.GenStruct()
    if thatXoverNode.parent is None:
        thisC = Tree(thatXoverNode, thisC.depth + thatC.depth)
    else:
        thisC.GenStruct()

    return thisC, thatC

def OP_Crossover(population, parents, probXover):
    '''
    Performs tree crossover on the current generation of a GP after the solutions
    have been selected and paired for mating. Crossover is selected to occur
    with (probXover)% probability - if no crossover, the results are genetic replicates.
    For single-and double-point crossover, the points are selected uniformly.
    :param population: array_like of the current population
    :param parents: (n/2,2) array indicating pairs of solutions to mate; if n is odd,
        parents is of size ((n-1)/2,2)
    :param probXover: scalar float probability of crossover (in range [0,1])
    :return newPop: (n,p) array of next generation's population
    '''
    
    # initialize the container for the new population
    newPop = []
    
    # iterate over pairs of trees
    for pair in parents:
        # get the parents
        this = population[pair[0]]
        that = population[pair[1]]
        if probXover > np.random.rand():
            # cross them over
            offspring = TreesCrossover(this, that)
            newPop.extend(offspring)
        else:
            # just copy them
            newPop.extend((copy.deepcopy(this), copy.deepcopy(that)))
    
    return np.array(newPop)


def TreeMutate(this, maxDepth, nodeMeta, verbose=False):
    '''
    Mutate a tree at a randomly-selected node.
    :param this: tree to mutate
    :param maxDepth: integer maximum depth allowed for the tree (including the root)
    :param nodeMeta: dictionary holding the a tuple of a list of the node values
        allowed, the number of node values allowed, and node weight for random
        selection; keys are node types of 'ops, 'feat', and 'const'
    :param verbose: optional (default = false) flag to print some info
    :return this_new: the mutated tree
    '''
    # first create a copy of the input tree
    thisC = copy.deepcopy(this)
    
    # get the random mutation point
    thisMutePoint = np.random.permutation(list(chain.from_iterable(thisC.struct.values())))[0]
    if verbose:
        print('This mutation point: %s '%thisMutePoint)
    
    # mutating this node can have big consequences; potentially as big as creating
    # a new tree - so just do that; but it should be shorter than a full tree
    muty = BuildTree(np.random.randint(1, maxDepth), nodeMeta, verbose)
    if verbose:
        print(muty)
    
    # mutate the input tree
    if thisMutePoint.parent is None:
        # mutation point is the root, so just pass the new tree
        thisC = muty
    else:
        muty = muty.root
        # now graft in the new tree
        if thisMutePoint.parent.right == thisMutePoint:
            thisMutePoint.parent.setRight(muty)
        elif thisMutePoint.parent.left == thisMutePoint:
            thisMutePoint.parent.setLeft(muty)
        muty.parent = thisMutePoint.parent
        thisC.GenStruct()

    return thisC

def TreePrune(this, nodeMeta, verbose=False):
    '''
    Prune a tree at a randomly-selected node.
    :param this: tree to prune
    :param nodeMeta: dictionary holding the a tuple of a list of the node values
        allowed, the number of node values allowed, and node weight for random
        selection; keys are node types of 'ops, 'feat', and 'const'
    :param verbose: optional (default = false) flag to print some info
    :return this_new: the pruned tree
    '''
    # first create a copy of the input tree
    thisC = copy.deepcopy(this)
    
    # get the random pruning point - find the first non-root op node
    for thisPrunePoint in np.random.permutation(list(chain.from_iterable(thisC.struct.values()))):
        if (thisPrunePoint.type == 'op') and (thisC.root != thisPrunePoint):
            if verbose:
                print('This pruning point: %s '%thisPrunePoint)
            break
            
    # tree is too short to have a non-root op node, so just do nothing
    if thisPrunePoint == thisC.root:
        pass
    else:
        # create a replacement non-op node
        noOpsK = [k for k in nodeMeta.keys() if k != 'op']
        noOpsW = [nodeMeta[t][2] for t in noOpsK]
        nodeType, _ = RandomWeightedSelect(noOpsK, noOpsW, 0)
        nodeValu = nodeMeta[nodeType][0][np.random.randint(nodeMeta[nodeType][1])]
        muty = Node(nodeType, nodeValu, None)

        # prune the tree
        if thisPrunePoint.parent.right == thisPrunePoint:
            thisPrunePoint.parent.setRight(muty)
        elif thisPrunePoint.parent.left == thisPrunePoint:
            thisPrunePoint.parent.setLeft(muty)
        muty.parent = thisPrunePoint.parent
        thisC.GenStruct()

    return thisC

def OP_MutatePrune(population, prob, maxDepth, nodeMeta, mutePrune):
    '''
    Perform random mutation or pruning on a population of trees, this would
    usually be performed after the previous generation has mated and produced
    the next generation.
    :param population: array_like of the mated population
    :param prob: scalar float probability of mutation or pruning (in range [0,1])
    :param maxDepth: integer maximum depth allowed for the tree (including the root)
    :param nodeMeta: dictionary holding the a tuple of a list of the node values
        allowed, the number of node values allowed, and node weight for random
        selection; keys are node types of 'ops, 'feat', and 'const'
    :param mutePrune: 'm' for mutation, 'p' for pruning
    :return newPop: array of mutated population
    '''
    
    # initialize the container for the new population
    n = len(population)
    newPop = np.array([None]*n)
    
    # get the index of the trees that will mutate
    mutators = (prob > np.random.rand(n))
    
    # save the non-mutators
    newPop[~mutators] = population[~mutators]
    
    # iterate over the trees to mutate and save
    for muteMe in np.nonzero(mutators)[0]:
        if mutePrune == 'm':
            newPop[muteMe] = TreeMutate(population[muteMe], maxDepth, nodeMeta)
        elif mutePrune == 'p':
            newPop[muteMe] = TreePrune(population[muteMe], nodeMeta)
    
    return newPop


def OP_Simplify(population, prob):
    '''
    Perform randomsimplification on a population of trees, this would
    usually be performed after the previous generation has mated and
    and been mutated to produce the next generation. Note that while
    this returns the simplified population, it changes the trees in
    population in place.
    :param population: array_like of the mated population
    :param prob: scalar float probability of simplification (in range [0,1])
    :return newPop: array of mutated population
    '''
    
    # get the index of the trees that will be simplified
    simplers = (prob > np.random.rand(len(population)))
    
    # iterate over the trees to simplify & save
    for simpMe in np.nonzero(simplers)[0]:
         population[simpMe].Simplify()

    return population


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
        rands_in_bins = np.repeat(rnd.rand(populSize), populSize) >= np.tile(bins, populSize)
        # summing all lower bound flags for each random gives the bin it falls into (since 0-based)
        newPop = np.sum(np.reshape(rands_in_bins, [populSize]*2), axis=1)
        # now index into the stdindex to get parents
        parents = stdIndex[newPop]
        # odd number, so reinsert the best; will randomly permute order, so doesn't matter where
        if populSize % 2 == 1:
            parents = np.insert(parents, 0, stdIndex[0])
            populSize += 1
        # randomly resort then pair up
        parents = np.reshape(parents[rnd.permutation(populSize)], (populSize/2, 2))

        # 20160225 JAH don't want an uneven population size (often from Elitism) to result in
        # such fast population growth, so randomly cull one mating pair, with frequency relative
        # to their average scores (worst avg score is most likely to be culled)
        if popSitness.size % 2 == 1:
            # compute the parent's avg scores, then get the best<>worst sorted 1-based indexes
            srtAvgScs = np.argsort(np.sum(popFitness[parents], axis = 1)*(-optimGoal))+1
            # create the (0,1] bin upper bounds
            cumProbs = np.cumsum(srtAvgScs/np.sum(srtAvgScs))
            # pick which mating pair is culled - it's the last bin upper bound that's <= the random
            cullMe = (srtAvgScs[rnd.rand() <= cumProbs])[0] - 1
            parents = np.hstack((parents[:cullMe], parents[(cullMe+1):]))
            
    return parents


def RunGP(params, data, objective, nodeMeta, seedTrees=[], verbose=False, randSeed=None):
    '''
    Run the GP algorithm for a specified dataset. Parameters for the objective
    function must be passed, along with their names in the objective. While
    only params['showTopSubs'] results will be display, the unique best solutions
    from all generations will be returned.
    :param params: dictionary of GP parameters:
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
        'probPrune': float probability of tree pruning
        'probSimp': float probability of tree simplification
        'optimGoal': 1 = maximize, -1 = minimize
        'plotFlag': True = on, False = off
        'printFreq': integer number of generations by which the GP will print
            progress
        'maxDepth': integer maximum depth allowed for trees (including the root)
    :param data: dictionary expected to hold two items:
        'data': pandas dataframe of data; columns should be X0, X1, ...
        'name': name (descriptive or perhaps file) of data
    :param objective: dictionary of objective function parameters:
        'function': string function to execute whatever modeling is required and
            return the objective score; it can return multiple items, but the
            first must be the score; if a single item is returned, it should be
            in a tuple
        'arguments': dictionary of arguments to pass to the objective function;
            should include at least 'data' and 'tree'
    :param nodeMeta: dictionary holding the a tuple of a list of the node values
        allowed, the number of node values allowed, and node weight for random
        selection; keys are node types of 'op', 'feat', and 'const'
    :param seedTrees: optional list of tree objects to seed the initial population
    :param verbose: optional (default = false) flag to print extra info
    :param randSeed: optional (default = none) seed for randomizer; if not passed,
        this will be generated and printed
    :return bestTree: the best tree overall
    :return bestScore: the score of the best overall tree
    :return genBest: array of the best tree from each generation
    :return genScores: array of the best score and the average (of finite) scores
        from each generation
    :return randSeed: the random seed used
    :return tstamp: string timestamp of the GP run
    :return fig: plotly figure of the GP progress
    '''

    # start time and timestamp
    stt = dt.datetime.now()
    sttT = time.perf_counter()
    tstamp = re.sub('[^0-9]', '', stt.isoformat()[:19])
    
    # parse GP parameters
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
    maxDepth = int(params['maxDepth'])
    showTopSubs = int(params['showTopSubs'])
    probPrune = float(params['probPrune'])
    probSimp = float(params['probSimp'])
    
    # parse the data
    dataName = data['name']
    data = data['data']
    n, p = data.shape
    
    # parse the objective
    objFunc = objective['function']
    objArgs = objective['arguments'] 
    objArgs['data'] = data
    objStr = '%s(%s)'%(objFunc, ', '.join(['%s=%r'%(key, val) for (key, val) in objArgs.items()\
        if key not in ['data', 'tree', 'feats']]))
    
    # set the random state
    if randSeed is None:
        randSeed = int(str(time.time()).split('.')[1])
        print('Random Seed = %d'%randSeed)
    np.random.seed(randSeed)
        
    # display parameters
    dispLine = '#'*42
    print('%s\nGP Started on %s\n%s'%(dispLine, stt.isoformat(), dispLine))
    print('Data: %s(n=%d, p=%d)'%(dataName, n, p))
    print('Random Seed: %d'%randSeed)
    print('Maximum # Generations: %d\nMininum # of Generations: %d\nConvergence Criteria: %0.8f'%(numGens,noChangeTerm,convgCrit))
    if populSize % 2 == 1:
        populSize += 1
        print('!!Population Size Increased By 1 to be Even!!')
    print('Population Size: %d'%populSize)
    print('Initial Population Seeded with %d Trees'%len(seedTrees))
    print('Mutation Rate: %0.2f\nPrune Rate: %0.2f\nCrossover Rate: %0.2f\nSimplification Rate: %0.2f'\
          %(probMutate, probPrune, probXover, probSimp))
    print('Mating Method: %s'%['SORTED','ROULETTE'][mateType - 1])
    print('Elitism is: %s'%['OFF','ON'][elitism])
    print(dispLine)
    if optimGoal == 1:
        print('Objective: MAXIMIZE')
    else:
        print('Objective: MINIMIZE')
    print('Objective Function: %s'%objStr)
    print(dispLine)
    
    # randomly initialize the population of trees
    population = np.array([None]*populSize)
    for indx in range(populSize-len(seedTrees)):
        population[indx] = BuildTree(maxDepth, nodeMeta, True)
    # add the seed trees
    for indx in range(-1*len(seedTrees), 0, 1):
        population[indx] = seedTrees[indx]
        if verbose:
            print('Seeded Initial Population with \n%s'%population[indx])
    # tree simplification
    population = OP_Simplify(population, probSimp)
    # talk
    if verbose:
        for indx in range(populSize):
            print('%0d\n%s'%(indx, population[indx]))
    
    # now initialize more things
    # save results by generation
    genScores = np.zeros((numGens, 2), dtype=float)
    genBest = np.array([None]*numGens)
    # current generation's best
    bestTree = None
    bestScore = optimGoal*-1*np.Inf
    # generations with no improvement termination counter
    termCount = 0
    
    # Begin GP Algorithm Whoo Hoo!
    for genCnt in range(numGens):        
        ''' compute or lookup objective function values '''
        popFitness = np.ones(populSize, dtype=float)*np.Inf
        for popCnt in range(populSize):
            if genCnt > 0:
                # check if this tree already evaluated, to save time
                prevEval = np.nonzero(population[popCnt].function == allTrees)[0]
                if prevEval.size == 0:
                    # evalaute
                    objArgs['tree'] = population[popCnt].function
                    popFitness[popCnt] = globals()[objFunc](**objArgs)[0]
                else:
                    # look up existing score
                    popFitness[popCnt] = allScores[prevEval]
            else:
                objArgs['tree'] = population[popCnt].function
                popFitness[popCnt] = globals()[objFunc](**objArgs)[0]
                
        # If optimGoal is (+), this will not change the scores, so the true max will be taken.
        # If optimGoal is (-), the signs will all be changed, and since max(X) = min(-X),
        # taking the maximum will really be taking the minimum.
        optInd = np.argmax(optimGoal*popFitness)
        optVal = popFitness[optInd]
        
        # save some stuff before moving along
        genScores[genCnt,:] = (optVal, np.mean(popFitness[np.isfinite(popFitness)]))
        genBest[genCnt] = population[optInd]
        
        ''' save all unique trees & their scores '''
        if genCnt == 0:
            # first generation, so create these arrays here
            allScores = popFitness
            allTrees = [tree.function for tree in population]
        else:
            # not first generation, so append to the existing arrays
            allScores = np.append(allScores, popFitness)
            allTrees = np.hstack((allTrees, [tree.function for tree in population]))
        # now just get the unique trees
        uniq, ind = np.unique(allTrees, return_index=True)
        allScores = allScores[ind]
        allTrees = uniq
        
        ''' check for early termination '''
        if optimGoal*genScores[genCnt, 0] > optimGoal*bestScore:
            # this is a better score, so save it and reset the counter
            bestScore = genScores[genCnt, 0]
            bestTree = genBest[genCnt]
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
        # mate them
        newPop = OP_Crossover(population, parents, probXover)
        # tree pruning
        newPop = OP_MutatePrune(newPop, probMutate, maxDepth, nodeMeta, 'p')
        # tree mutation
        newPop = OP_MutatePrune(newPop, probMutate, maxDepth, nodeMeta, 'm')
        # tree simplification
        newPop = OP_Simplify(newPop, probSimp)

        ''' finalize new population & convey best individual into it if not already there '''
        if elitism:
            # check if best is currently in newPop
            if np.nonzero(bestTree == newPop)[0].size == 0:
                population = np.hstack((newPop, bestTree))
            else:
                population = newPop.copy()
        else:
            population = newPop.copy()
        # readjust in case population grew
        populSize = len(population)
    
        # talk, maybe
        if genCnt % printFreq == 0:
            print('Generation %d of %d: Best Score = %0.4f, Early Termination = %d\n\t%s'%\
                (genCnt + 1, numGens, bestScore, termCount, bestTree.function))
            
    # Finish GP Algorithm Whoo Hoo!
    print('Generation %d of %d: Best Score = %0.4f, Early Termination = %d\n\t%s'%\
        (genCnt + 1, numGens, bestScore, termCount, bestTree.function))
    
    ''' simplify the best trees '''
    print('Simplifying Generation Best Trees')
    bestTree.Simplify()
    for tree in genBest:
        tree.Simplify()
    
    ''' plot GP progress '''
    fig = plysub.make_subplots(rows=3, cols=1, print_grid=False, subplot_titles=['Best Score', 'Average Score', 'Solution Length'])
    # build traces
    gens = len(genBest)
    xs = list(range(gens))
    fig.add_trace(go.Scatter(x=xs, y=genScores[:,0], mode='markers+lines', name='Best Score', text=['%s = %0.5f'%(tree.function, score) for (tree, score) in zip(genBest, genScores[:,0])]), 1, 1)
    fig.add_trace(go.Scatter(x=xs, y=genScores[:,1], mode='markers+lines', name='Average Score'), 2, 1)
    fig.add_trace(go.Scatter(x=xs, y=[len(tree.function) for tree in genBest], mode='markers+lines', name='Solution Length'), 3, 1)
    # annotate the best solution
    bestAnn = dict(x=gens-1, y=np.min(genScores[:,0]), xref='x1', yref='y1', text='%s = %0.5f'%(bestTree.function, bestScore),
                   showarrow=True, bordercolor="#c7c7c7", borderwidth=2, borderpad=4, bgcolor="#6d72f1", opacity=0.8,
                   font={'color':'#ffffff'}, align="center", arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#636363")
    # update layout
    anns = list(fig['layout']['annotations'])
    anns.append(bestAnn)    
    fig.update_layout(title='GP Progress Resuls by Generation (%s, %s, %s)'%(dataName, objStr, tstamp), annotations=anns)
    fig['layout']['xaxis3'].update(title='Generation')
    if plotFlag:
        plyoff.plot(fig, filename='../output/GPProgress_%s_%s_%s.html'%(tstamp, re.sub('[^0-9A-Za-z_]', '_', dataName), objFunc), auto_open=True, include_mathjax='cdn')
    
    ''' summaryize results: GA_BEST '''
    # build the results dataframe
    tmp = np.array([[tree.function for tree in genBest], genScores[:,0]]).T
    GA_BEST = pd.DataFrame(data=tmp, columns=['Tree Function', 'Tree Score'])
    # compute the tree bests frequencies, add them in
    freqs = pd.DataFrame(data=GA_BEST['Tree Function'].value_counts()/len(genBest)).reset_index()\
        .rename(columns={'Tree Function':'Frequency', 'index':'Tree Function'})
    GA_BEST = GA_BEST.merge(freqs, on=['Tree Function'], how='inner')
    # drop duplicates and move the function to the index
    GA_BEST = GA_BEST.drop_duplicates().set_index('Tree Function')
    # sort by a temporary column so the best are at the top
    GA_BEST['tmp'] = -1*optimGoal*GA_BEST['Tree Score']
    GA_BEST = GA_BEST.sort_values(by='tmp').drop(columns='tmp', inplace=False)
    
    # show results
    print('%s\nGP Complete\n\tUnique Trees Evaluated - %d\nTop %d Solutions'%(dispLine, len(allScores), showTopSubs))
    display(GA_BEST.head(showTopSubs))
    
    # stop time
    stp = dt.datetime.now()
    stpT = time.perf_counter()
    print('GP: Started on %s\n\tFinished on %s\n\tElapsed Time = %0.3f(m)'%(stt.isoformat(), stp.isoformat(), (stpT-sttT)/60))
    
    return bestTree, bestScore, genBest, genScores, randSeed, tstamp, fig