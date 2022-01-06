#QREDIS_GA
"""
QREDIS genetic algorithm subsetting functions
----------------------
Crossover - 20120919 - perform GA crossover
Engineer - 20121012 - perform GA engineering
MateSelect - 20120917 - select pairs from a GA population for mating
Mutate - 20120919 - perform GA mutation
RunGA - 20121030 - run the GA for subset modeling
ReRunGA - 20170405 - rerun RunGA with the seed from a previous run
----------------------
JAH 20140118 everything has been tested and seems to be working fine with python 3
"""

import datetime as dat
import string
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import QREDIS_Basic as QB

def MateSelect(pop_fitness, optim_goal, meth):
	"""
	Generate an index array into the population showing which members to mate.
	If the population size is uneven, the solutions will all be mated as instructed,
	but then at the end, one of the mating pairs will be randomly culled from mating.
	The random selection is proportional to the pairs' average fitnesses.
	---
	Usage: parents = MateSelect(pop_fitness, optim_goal, meth)
	---
	pop_fitness: (n,) data array_like of fitness scores from a population of size n
	optim_goal: 1 = maximize, -1 = minimize
	meth: 1 = sorted, 2 = roulette
	parents: (n/2,2) array indicating pairs of solutions to mate; if n is odd,
		parents is of size ((n-1)/2,2)
    ---
	ex: par = QG.MateSelect(rnd.rand(10),-1,1); print(par)
	JAH 20120917
	"""
	# duck pop_fitness into 1d array JAH 20120920
	pop_fitness = np.array(pop_fitness,ndmin=1,copy=False)
	popul_size = pop_fitness.size

	# ensure proper arguments
	if (type(optim_goal) is not int) or (type(meth) is not int):
		raise TypeError("Variable(s) are wrong type: %s"%MateSelect.__doc__)
	if not(optim_goal in [1,-1]):
		raise ValueError("Objective must be +1 or -1!: %s"%MateSelect.__doc__)
	if not(meth in [1,2]):
		raise ValueError("Selection method must be 1 or 2!: %s"%MateSelect.__doc__)
	if (pop_fitness.ndim != 1) and (1 not in pop_fitness.shape): # JAH 20121005 can be 2d if vector
		raise TypeError("Fitness scores must be column or row vector: %s"%MateSelect.__doc__)

	# sort fitness scores, if optim_goal is positive, this will sort the scores
	# descending, with the lowest at the front, if optim_goal is negative, it is
	# essentially sorting ascending, with the largest at the front, either way,
	# the best chromosomes are associated with largest roulette bins
	stdindex = np.argsort(pop_fitness*-optim_goal)

	# do the work
	if meth == 1:	# simply mate pairwise
		# odd number, so reinsert the best; we want to insert as 3rd so best doesn't mate with self
		if popul_size % 2 == 1:
			stdindex = np.insert(stdindex, 2, stdindex[0])
			popul_size +=1
		parents = np.reshape(stdindex,(popul_size/2,2))
	else:               # roulette method
		# prepare bins for roulette - bigger bins at the beginning with lower scores
		bins = np.cumsum(np.linspace(popul_size,1,popul_size)/(popul_size*(popul_size + 1.0)/2))
		# first n random numbers to each bin to find each bin that is a lower bound for each rand
		rands_in_bins = np.repeat(rnd.rand(popul_size),popul_size) >= np.tile(bins,popul_size)
		# summing all lower bound flags for each random gives the bin it falls into (since 0-based)
		new_pop = np.sum(np.reshape(rands_in_bins,[popul_size]*2),axis=1)
		# now index into the stdindex to get parents
		parents = stdindex[new_pop]
		# odd number, so reinsert the best; will randomly permute order, so doens't matter where
		if popul_size % 2 == 1:
			parents = np.insert(parents,0,stdindex[0])
			popul_size += 1
		# randomly resort then pair up
		parents = np.reshape(parents[rnd.permutation(popul_size)],(popul_size/2,2))

		# 20160225 JAH don't want an uneven population size (often from Elitism) to result in
		# such fast population growth, so randomly cull one mating pair, with frequency relative
		# to their average scores (worst avg score is most likely to be culled)
		if pop_fitness.size%2 == 1:
			# compute the parent's avg scores, then get the best<>worst sorted 1-based indexes
			srtavgscs = np.argsort(np.sum(pop_fitness[parents],axis = 1)*(-optim_goal))+1
			# create the (0,1] bin upper bounds
			cumprobs = np.cumsum(srtavgscs/np.sum(srtavgscs))
			# pick which mating pair is culled - it's the last bin upper bound that's <= the random
			cullme = (srtavgscs[rnd.rand() <= cumprobs])[0] - 1
			parents = np.vstack((parents[:cullme,:],parents[(cullme+1):,:]))
	return parents

def Crossover(popul, parents, xover_rate, xover_type):
	"""
	Performs chromosomal crossover on the current generation of a GA after the solutions
	have been selected and paired for mating. First, crossover is selected to occur with
	(xover rate)% probability - if no crossover, the results are genetic replicates. For
	single-and double-point crossover, the points are selected uniformly. For uniform
	crossover, each point is selected to crossover or not using the xover rate.
	---
	Usage: offspring = Crossover(popul, parents, xover_rate, xover_type)
	---
	popul: (n,p) bool array of n GA solutions of size p each
	parents: (>=n/2,2) array indicating pairs of solutions to mate from MateSelect
		if elitism is on, parents.shape[0] may be > n/2
	xover_rate: scalar float probability of crossover (in range [0,1])
	xover_type: 1 = single point, 2 = dual point, 3 = uniform
	offspring: (n,p) array of next iteration population
	---
	ex:
	JAH 20120919
	"""

	# pop and parent arrays: duck you! JAH 20120920
	popul = np.array(popul,ndmin=2,dtype=bool,copy=False)
	parents = np.array(parents,ndmin=2,copy=False)

	# now get the dimensions JAH 20120920
	(n,p) = popul.shape
	matepairs = parents.shape[0]		# number couples, should be n/2

	# ensure proper arguments JAH 20120920
	if (type(xover_rate) is not float) or (type(xover_type) is not int):
		raise TypeError("Variable(s) are wrong type: %s"%Crossover.__doc__)
	if (abs(xover_rate) < 0) or (xover_rate > 1):
		raise ValueError("Crossover rate must be a float probability in [0,1]: %s"%Crossover.__doc__)
	if not(xover_type in [1,2,3]):
		raise ValueError("Crossover method must be 1 or 2!: %s"%Crossover.__doc__)
	if parents.ndim > 2:
		raise ValueError("Parents must be (>=n/2,2) array: %s"%Crossover.__doc__)
	if popul.ndim > 2:
		raise ValueError("Population must be (n,p) array: %s"%Crossover.__doc__)

	offspring1 = np.zeros((matepairs,p)); offspring2 = offspring1.copy()
	# perform the crossovers (maybe)
	if xover_type == 1:     # single-point
		for matecnt in range(matepairs):
			dad = popul[parents[matecnt,0],:]
			mom = popul[parents[matecnt,1],:]
			if xover_rate > rnd.rand():     # crossover
				# point randomly selected - endpoints allowed (since :p excludes p)
				xoverpoint = rnd.random_integers(1,p-1)
				offspring1[matecnt,:] = np.concatenate((dad[:xoverpoint], mom[xoverpoint:]))
				offspring2[matecnt,:] = np.concatenate((mom[:xoverpoint], dad[xoverpoint:]))
			else:							# genetic replication
				offspring1[matecnt,:] = dad.copy()
				offspring2[matecnt,:] = mom.copy()
	elif xover_type == 2:   # double-point
		for matecnt in range(matepairs):
			dad = popul[parents[matecnt,0],:]
			mom = popul[parents[matecnt,1],:]
			if xover_rate > rnd.rand():     # crossover
				# select 2 points randomly without replacement in inclusive range [1,p-2]
				xoverpoints = np.sort(rnd.permutation(p-1)[:2]+1)
				offspring1[matecnt,:] = np.concatenate((dad[:xoverpoints[0]],\
					mom[xoverpoints[0]:xoverpoints[1]],dad[xoverpoints[1]:]))
				offspring2[matecnt,:] = np.concatenate((mom[:xoverpoints[0]],\
					dad[xoverpoints[0]:xoverpoints[1]],mom[xoverpoints[1]:]))
			else:                           # genetic replication
				offspring1[matecnt,:] = dad.copy()
				offspring2[matecnt,:] = mom.copy()
	elif xover_type == 3:   # uniform
		for matecnt in range(matepairs):
			dad = popul[parents[matecnt,0],:]
			mom = popul[parents[matecnt,1],:]
			if xover_rate > rnd.rand():     # crossover
				xoverpoints = xover_rate > rnd.rand(p)
				offspring1[matecnt,:] = dad*xoverpoints + mom*~xoverpoints
				offspring2[matecnt,:] = dad*~xoverpoints + mom*xoverpoints
			else:                           # genetic replictaion
				offspring1[matecnt,:] = dad.copy()
				offspring2[matecnt,:] = mom.copy()

	return np.vstack((offspring1,offspring2)) == 1

def Mutate(popul, mutat_rate):
	"""
	Perform random mutation on a population of chromosomes, this would usually be performed
	after the previous generation has mated and produced the next generation.
	---
	Usage: mutated = Mutate(popul, mutat_rate)
	---
	popul: (n,p) array of n GA solutions of size p each
	mutat_rate: scalar float probability of mutation (in range [0,1])
	mutated: (n,p) array of mutated population
	---
	ex: mut = QG.Mutate(np.eye(10,dtype=bool),0.2); print(mut)
	JAH 20120919
	"""

	# ducktype population
	popul = np.array(popul,ndmin=2,dtype=bool,copy=False)
	(n,p) = popul.shape

	# ensure proper arguments
	if (abs(mutat_rate) < 0) or (mutat_rate > 1) or (type(mutat_rate) is not float):
		raise ValueError("Mutation rate must be a float probability in [0,1]: %s"%Mutate.__doc__)

	# get the index of the elements that will mutate
	mutators = (mutat_rate > rnd.rand(n,p))
	# now prepare the mutated population
	new_pop = popul.copy()
	new_pop[mutators] = ~(new_pop[mutators])

	return new_pop

def Engineer(current_best, previous_best, current_offspring, engineer_rate):
	"""
	This implements GA engineering, the point of which is to limit the variability between
	GA runs. It takes the best solution from the previous generation and the best solution
	from the current generation, and finds the difference. Where they are different, the
	bits from the previous best are inserted into the offspring of the current generation
	with the specified probability. This should only be called if the previous best is
	better than the current best.
	---
	Usage: offspring = Engineer(current_best, previous_best, current_offspring, engineer_rate)
	---
	previous_best: p-length array_like holding the best solution from the previous generation
	current_best: p-length array_like holding the best solution from the current generation
	current_offspring: (n,p) array of n GA solutions of size p each; this is the offspring for the next generation
	engineer_rate: scalar float probability of GA engineering (in range [0,1])
	offspring: (n,p) current_offspring with bits engineered
	---
	ex:
	JAH 20121012
	"""

	# duck arrays
	current_best = np.array(current_best,ndmin=1,dtype=bool,copy=False).flatten()
	previous_best = np.array(previous_best,ndmin=1,dtype=bool,copy=False).flatten()
	current_offspring = np.array(current_offspring,ndmin=2,dtype=bool,copy=False)
	(n,p) = current_offspring.shape

	# ensure proper arguments
	if (abs(engineer_rate) < 0) or (engineer_rate > 1) or (type(engineer_rate) is not float):
		raise ValueError("GA Engineering rate must be a float probability in [0,1]: %s"%Engineer.__doc__)
	if (p != current_best.size) or (current_best.size != previous_best.size):
		raise ValueError("Offspring and best arrays should be of same length: %s"%Engineer.__doc__)

	# first get where they are different and the actual different values
	difflocs = np.logical_xor(current_best, previous_best)
	diffvals = previous_best[difflocs]
	# if no differences, just exit
	if np.sum(difflocs) == 0:
		return current_offspring

	# generate the randoms to decide who gets engineered
	engme = engineer_rate > rnd.rand(n)
	# if none selected to engineer, just exit
	if np.sum(engme) == 0:
		return current_offspring

	# GA engineer - current_offspring[engme,difflocs] = diffvals doesn't work :-(
	newoffspring = np.zeros((n,p),dtype=bool)
	rows = np.arange(n)
	# first build the output by adding the population elements that won't be changed
	rows_nochg = rows[~engme]
	newoffspring[rows_nochg,:] = current_offspring[~engme,:]
	# now edit the rest of the population
	jnk = current_offspring[engme,:]
	jnk[:,difflocs] = diffvals
	newoffspring[rows[engme],:] = jnk

	return newoffspring

def RunGA(data_parms, objec_parms, params, out_file=None):
	"""
	Run the genetic algorithm for subsetting for a specified model and dataset.  If an
	output path and file is given, it will save the console output and plot (if generated).
	Parameters for the objective function must be passed, along with their names in the
	objec_parms.	While only params['showtopsubs'] results will be printed on the
	console, the unique best solutions from all generations will be passed back out in
	best_solutions. If you have less than, say, 7 variables, complete enumeration
	(with VarSubset in QREDIS_Basic) is probably better.
	---
	Usage: best_solutions, out_file = RunGA(data_parms, objec_parms, params, out_file)
	---
	data_parms: dictionary expected to hold two items:
		data: (nxp) array of data to be subset
		data_name: filename of data
	objec_parms: dictionary of objective function parameters:
		function: string function to execute whatever modeling is required and return the
			objective score; it can return multiple items, but the first must be the score
		israndom: boolean flag indicating if the objective function is stochastic;
			True = solutions repeatedly seen are always evaluated, and the best score is taken;
			False* = solutions repeatedly seen are not re-evaluated
		data: name of the parameter in the function for the subset data
			anything else: other named parameters and values to pass to the objective function
	params: dictionary of GA parameters:
		init_perc: float percent of initial population filled
 		seed_vars: array_like of variable index numbers up to p-1 to seed in the
			initial population; will violate the init_perc; if no seed desired, use None
 		force_vars: array_like of variable index numbers up to p-1 to force to stay
			in all solutions; if none desired, use None
		seed_sols: array_like of binary strings with which to seed the initial population
			bypasses seeds_vars, force_vars, and init_perc; if none desired, use None
		showtopsubs: integer number of best subsets to show
		popul_size: integer population size
		num_generns: integer number of generations
		nochange_terminate: integer number generations with insufficient improvement
			before early termination
		convgcrit: float convergence criteria
		elitism: True = on, False = off
		mate_type: 1 = sorted, 2 = roulette
		prob_xover: float probability of crossover
		xover_type: 1 = single, 2 = dual, 3 = uniform
		prob_mutate: float probability of mutation
		prob_engineer: float probability of GA engineering
		optim_goal: 1 = maximize, -1 = minimize
		plotflag: True = on, False = off
		printfreq: integer GA prints something to screen every # generations
		randstate: None = let numpy randomize, else (integer) = random state
	out_file: full path and file name sans timestamp & extension where to save diary and
		plots; if None*, nothing is saved
	best_solutions: array holding the unique best scores, their GA frequencies, and the best
		solutions in order from best(best) to worst(best)
	out_file: full path and file name where diary and plots are saved
	---
	ex: data_parms = 'data = np.reshape(1+np.tile(np.arange(20),20),(20,20),order="C"); data_name = "428"'
		objec_parms = {'function':'np.sum','israndom':False,'data':'a','axis': None}
		params = {'seed_vars':[],'force_vars':[],'seed_sols':[],'init_perc':0.5,\
			'showtopsubs':5,'popul_size':25,'num_generns':100,'nochange_terminate':80,\
			'convgcrit':0.0001,'elitism':True,'mate_type':2,'prob_xover':0.75,\
			'xover_type':1,'prob_mutate':0.1,'prob_engineer':0.25,'optim_goal':1,\
			'plotflag':True,'printfreq':2,'randstate':42}
		out_file = '/home/ahowe42/QREDIS/out/demoGA'
		best_solutions = QG.RunGA(data_parms, objec_parms, params, out_file)[0]
		print(best_solutions[0,2:].T)
	JAH 20121030
	"""

	# close any open figures
	plt.close('all')

	# check inputs
	if (type(data_parms) is not dict) or (type(objec_parms) is not dict) or ((type(out_file) is not str) and (out_file is not None)):
		raise TypeError('Something wrong with inputs: %s'%RunGA.__doc__)

	# check and parse GA parameters
	try:
		if (type(params) is not dict) or (len(params) != 19):
			raise ValueError('Variable params must be a dictionary with 19 items: %s'%RunGA.__doc__)
		# parse and ducktype the params dictionary
		init_perc = float(params['init_perc'])
		seed_vars = params['seed_vars'] # JAH 20160225 added
		force_vars = params['force_vars'] # JAH 20160225 added
		seed_sols = params['seed_sols'] # JAH 20160304 added
		showtopsubs = int(params['showtopsubs'])
		popul_size = int(params['popul_size'])
		num_generns = int(params['num_generns'])
		nochange_terminate = int(params['nochange_terminate'])
		convgcrit = float(params['convgcrit'])
		elitism = bool(params['elitism'])
		mate_type = int(params['mate_type'])
		prob_xover = float(params['prob_xover'])
		xover_type = int(params['xover_type'])
		prob_mutate = float(params['prob_mutate'])
		prob_engineer = float(params['prob_engineer'])
		optim_goal = int(params['optim_goal'])
		plotflag = bool(params['plotflag'])
		printfreq = int(params['printfreq'])
		randstate = params['randstate']
		if (randstate is not None) and (type(randstate) is not int):
			raise TypeError('GA parameter randstate must be None or an integer: %s'%RunGA.__doc__)
	except ValueError:
		raise ValueError('A GA parmeter is the wrong type: %s'%RunGA.__doc__)
	except TypeError:
		raise TypeError('A GA parmeter is the wrong type: %s'%RunGA.__doc__)
	except KeyError:
		raise KeyError('A parameter might be missing from GA params dictionary: %s'%RunGA.__doc__)

	# start time and get save prefix for if we're Diarying
	stt = dat.datetime.now()
	save_prefix = '%s_%s'%(out_file,QB.TimeStamp(stt))
	# run data setup - we should now have variables: data, data_name
	data = data_parms['data']
	data_name = data_parms['data_name']
	(n,p) = data.shape
	# these are used later on
	varis = np.arange(p)+1
	bin_to_dec = 2**(varis-1)

	# parse the objective function params # JAH 20160307
	objec_func = objec_parms['function']
	objec_rand = objec_parms['israndom']
	# prepare the eval string
	objec_str = objec_func + '('
	for tmp in objec_parms.keys():
		if tmp == 'data':
			objec_str += objec_parms[tmp]+'=data[:,population[popcnt,:]],'
		elif (tmp != 'function') and (tmp != 'israndom') :
			objec_str += tmp+'=objec_parms["'+tmp+'"],'
	objec_str = objec_str[:-1]+')'

	# if randstate is None, this randomizes from the clock, otherwise, set a specific seed
	rnd.seed(randstate)
	# turn on diary, maybe
	if out_file is not None:
		QB.Diary(save_prefix+'.txt')

	# start talking
	disp_line = '#'*50
	print('GA Began on %s'%stt)

	# Display parameters, and do a few checks & corrections maybe
	print('%s\nRandom State: %10.0f'%(disp_line,rnd.get_state()[1][0]))
	print('Initial Fill Percentage: %0.2f'%init_perc)
	print('Initial Seed Vars: %r'%seed_vars) # JAH 20160225 added
	print('Vars Forced in all Models: %r'%force_vars) # JAH 20160225 added
	print('Seed Solutions: %s'%seed_sols) # JAH added 20160304
	print('Maximum # Generations: %0.0f\nMininum # of Generations: %0.0f\nConvergence Criteria: %0.8f'%(num_generns,nochange_terminate,convgcrit))
	if popul_size % 2 == 1:
		popul_size += 1
		print('!!Population Size Increased By 1 to be Even!!')
	print('Population Size: %0.0f'%popul_size)
	print('Mutation Rate: %0.2f\nCrossover Rate: %0.2f'%(prob_mutate,prob_xover))
	print('Crossover Method: %s'%['SINGLE','DUAL','UNIFORM'][xover_type - 1])
	print('Mating Method: %s'%['SORTED','ROULETTE'][mate_type - 1])
	print('Elitism is: %s'%['OFF','ON'][elitism])
	if (elitism) and (prob_engineer > 0.0):
		print('!!With Elitism ON, the probability of GA engineering has been set to 0.00!!')
		prob_engineer == 0.0
	else:
		print('GA Engineering Rate: %0.2f'%prob_engineer)
	if optim_goal == 1:
		print('Objective: MAXIMIZE')
	else:
		print('Objective: MINIMIZE')
	print('Data Name: %s\nObjective Function: %s (%s)\n%s'%\
		(data_name, objec_func,('deterministic','stochastic')[objec_rand],disp_line)) # JAH added istochastic flag 20160226

	# initialize the population with about 50% 1s, then ...
	population = (init_perc >= rnd.rand(popul_size,p))
	# ... let the first (up to) 5 rows be just first 5 subsets, and then ...
	for tmp in range(min(p,5)):
		population[tmp,:] = False
		population[tmp,:(tmp+1)] = True
	# ... force everything in seed_vars and force_vars to be included JAH 20160225...
	if seed_vars is not None:
		population[:,seed_vars] = True
	if force_vars is not None:
		population[:,force_vars] = True
	# ... then add any seed solutions to the end JAH 20160304 ...
	if seed_sols is not None:
		seed_sols = np.array(seed_sols,ndmin=2,copy=False)
		scnt = min(seed_sols.shape[0],popul_size)
		population[-scnt:,:] = seed_sols
	# ... but find any solutions that are entirely 0 and replace them
	all_zero = np.where(np.sum(population,axis=1)==0)[0]
	population[all_zero,rnd.randint(0,p,len(all_zero))] = True
	#import pdb; pdb.set_trace()

	# now initialize more things
	# save results by generation
	genscores = np.zeros((num_generns,2),dtype=float)
	genbest = np.zeros((num_generns,p),dtype=bool)
	# current generation's best
	best_chrom = np.zeros((1,p),dtype=bool)
	best_score = optim_goal*-1*np.Inf
	# previous generation's best
	prevgenbestchrom = best_chrom.copy()
	prevgenbestscore = best_score
	# generations with no improvement termination counter
	termcount = 0

	# Begin Genetic Algorithm Whoo Hoo!
	for gencnt in range(num_generns):
		# COMPUTE OBJECTIVE FUNCTION VALUES
		pop_fitness = np.ones(popul_size,dtype=float)*np.Inf
		for popcnt in range(popul_size):
			if gencnt > 0:
				# check if this chromosome already evaluated, to save time
				preveval = np.where(np.sum(allchroms == population[popcnt,:],axis=1)==p)[0]
				if preveval.size == 0:
					# this code handles if objec_func returns any number of outputs;
					# if only scalar is returned, it tuples then indexes it;
					# if multples are turned, tuple(a tuple) changes nothing
					pop_fitness[popcnt] = tuple(eval(objec_str))[0]
				else:
					# JAH 20160226 the different handling here driven by objec_rand added
					if objec_rand:		# evaluate then optimize with existing score
						tmp = tuple(eval(objec_str))[0]
						# get the best and save in both this population and history
						if max(optim_goal*allscores[preveval]) >= optim_goal*tmp:
							# one of the existing scores is not worst, so figure out which one
							# have to do this in case one existing score is same but other is worse
							# in that case, I need to update the worse
							tmp = allscores[preveval][np.argmax(optim_goal*allscores[preveval])]
						pop_fitness[popcnt] = tmp
						allscores[preveval] = tmp
					else:						# look up existing score
						pop_fitness[popcnt] = allscores[preveval][0]
			else:
				pop_fitness[popcnt] = tuple(eval(objec_str))[0]
		# If optim_goal is (+), this will not change the scores, so the true max will be taken.
		# If optim_goal is (-), the signs will all be changed, and since max(X) = min(-X),
		# taking the maximum will really be taking the minimum.
		optind = np.argmax(optim_goal*pop_fitness)
		optval = pop_fitness[optind]
		# save some stuff before moving along
		genscores[gencnt,:] = (optval, np.mean(pop_fitness[np.isfinite(pop_fitness)]))
		genbest[gencnt,:] = population[optind,:]

		# SAVE ALL UNIQUE CHROMOSOMES
		# must first convert population to decimal representation so can unique
		tmp = np.sum(population*bin_to_dec,axis=1)
		# now can get the unique values and their indices (only need the indices, since unique dec. values is useless to me)
		# yes, I know this sorts them first - and I don't really care
		tmp,ind = np.unique(tmp,return_index=True)
		# don't like this, but can't see a reasonable way around it
		if gencnt == 0:
			# first generation, so create these arrays here
			allscores = pop_fitness[ind]
			allchroms = population[ind,:]
		else:
			# not first generation, so append to the existing arrays
			allscores = np.append(allscores, pop_fitness[ind])
			allchroms = np.vstack((allchroms,population[ind,:]))

		# EARLY TERMINATION ALLOWED?
		if optim_goal*genscores[gencnt,0] > optim_goal*best_score:
			# this is a better score, so reset the counter
			best_score = genscores[gencnt,0]
			best_chrom = genbest[gencnt,:]
			termcount = 1
		elif (optim_goal*genscores[gencnt,0] < optim_goal*best_score) and (elitism == False):
			# if elitism is off, we can still do early termination with this
			termcount += 1
		elif abs(optim_goal*genscores[gencnt,0] - optim_goal*best_score) < convgcrit:
			# "no" improvement
			termcount += 1
		elif (elitism  == True):
			# NOTE: If the objective function is in anyway stochastic, it's possible that
			# even with elitism on, a generation's best score can be less optimal than the
			# global best, even though the global best solution is guaranteed to be in the pop
			# if this occurs, we still want to increase the early termination counter
			termcount += 1

		# don't bother with the next generation
		if gencnt == (num_generns-1):
			break

		if termcount >= nochange_terminate:
			print('Early Termination On Generation %d of %d'%(gencnt+1,num_generns))
			genscores = genscores[:(gencnt+1),:]	# keep only up to gencnt spaces (inclusive)
			genbest = genbest[:(gencnt+1),:]
			break

		# SELECTION OF NEXT GENERATION
		parents = MateSelect(pop_fitness, optim_goal, mate_type)
		# CROSSOVER OPERATION ON NEW POPULATION
		new_pop = Crossover(population, parents, prob_xover, xover_type)
		# CHROMOSOME MUTATION
		new_pop = Mutate(new_pop, prob_mutate)
		# GA ENGINEERING
		if prob_engineer > 0:
			# I check for prob_engineer > 0 because I can turn this off by
			# setting it to 0. Below we use genscores and gencnt, because
			# best_score and best_chrom won't hold the current best if the
			# current best solution is worse than the overall best, and elitism
			# is off.
			if (gencnt > 0) and (optim_goal*prevgenbestscore > optim_goal*best_score):
				# only call GAengineering if the previous generation best is better
				new_pop = GAengineering(genbest[gencnt,:], prevgenbestchrom, new_pop, prob_engineer)
			prevgenbestchrom = best_chrom
			prevgenbestscore = best_score

		# FIX ALL-ZERO CHROMOSOMES
		all_zero = np.where(np.sum(new_pop,axis=1)==0)[0]
		new_pop[all_zero,rnd.randint(0,p,len(all_zero))] = True

		# SETUP NEW POPULATION & CONVEY BEST INDIVIDUAL INTO NEW POPULATION IF NOT ALREADY THERE
		if elitism:
			# check if best is currently in new_pop
			if np.where(np.sum(new_pop == best_chrom,axis=1)==p)[0].size == 0:
				population = np.vstack((new_pop,best_chrom))
			else:
				population = new_pop.copy()
		else:
			population = new_pop.copy()
		# readjust in case population grew
		popul_size = population.shape[0]

		# finally, force any force_vars to be included JAH 20160225 added
		if force_vars is not None:
			population[:,force_vars] = True

		# talk, maybe
		if gencnt % printfreq == 0:
			# JAH 20160225 added best_chrome printing JAH 20160307 changed to show indexes 0-based
			print('Generation %d of %d: Best Score = %0.4f(%d), Early Termination = %d\n\t%r'%\
				(gencnt+1,num_generns,best_score,np.sum(best_chrom),termcount,(varis[best_chrom]-1).tolist()))
    # Finish Genetic Algorithm Whoo Hoo!
	# JAH 20160225 added best_chrome printing JAH 20160307 changed to show indexes 0-based
	print('Generation %d of %d: Best Score = %0.4f(%d), Early Termination = %d\n\t%r'%\
		(gencnt+1,num_generns,best_score,np.sum(best_chrom),termcount,(varis[best_chrom]-1).tolist()))

	# unique allchroms & allscores
	# must first convert population to decimal representation so can unique
	tmp = np.sum(allchroms*bin_to_dec,axis=1)
	# now can get the unique values and their indices (only need the indices, since unique dec. values is useless to me)
	# yes, I know this sorts them first - and I don't really care
	tmp,ind = np.unique(tmp,return_index=True)
	allchroms = allchroms[ind,:]
	allscores = allscores[ind]

	# create and save plot, maybe
	if plotflag:
		pltx = np.arange(gencnt+1)
		fhga = plt.figure()
		ax1 = fhga.add_subplot(1,1,1)
		ax1.plot(pltx,genscores[pltx,0],'bo-')
		ax1.set_ylabel('Optimum Value (o)',color='b')
		ax2 = ax1.twinx()
		ax2.plot(pltx,genscores[pltx,1],'r*-')
		ax2.set_ylabel('Average Value (*)',color='r')
		if (optim_goal == 1) and plotflag:
			plt.title('GA Progress: Maximize '+objec_func)
		else:
			plt.title('GA Progress: Minimize '+objec_func)
		plt.show()
		if out_file is not None:
			plt.savefig(save_prefix+'.eps')

	# SUMMARY: GA_BEST = (scores,frequencies,solutions)
	# must first convert generation best solutions to decimal representation so can unique
	genbest_dec = np.sum(genbest*bin_to_dec,axis=1)
	# now can get the unique values and their indices; yes, I know this sorts them first - and I don't really care
	tmp,ind = np.unique(genbest_dec,return_index=True)
	# combine the unique best scores and solutions from each gen + a placeholder column for frequencies in [1]
	# note that the frequencies placeholder is currently the gen averages from gen_scores; this will be overwritten
	GA_BEST = np.hstack((genscores[ind,:],genbest[ind,:]))
	# now sort so best is at top
	GA_BEST = GA_BEST[np.argsort(-1*optim_goal*GA_BEST[:,0]),:]
	# GA_BEST table info
	top_best = min(showtopsubs,ind.size)
	chromloc = range(2,(p + 2))

	# compute frequencies & prepare row headers for PrintTable
	rwhds = ['']*top_best
	for pcnt in range(ind.size):
		# compute frequency
		GA_BEST[pcnt,1] = np.sum(np.sum(GA_BEST[pcnt,chromloc]*bin_to_dec) == genbest_dec)
		# prepare row header if this will be printed
		if pcnt < top_best:
			rwhds[pcnt] = str(varis[True==GA_BEST[pcnt,chromloc]]-1) # JAH 20160307 changed to subtract 1 to show 0-based indexes

	# finally (almost) display the best chromosomes and scores
	disp_line = '='*60
	print('\n%s\nGA Complete\n\tUnique Solutions Evaluated - %d\n\tTotal Nontrivial Solutions Possible - %d'%(disp_line,allscores.size,2**p-1))
	tab = QB.PrintTable(GA_BEST[:top_best,:2],'%0.4f',[objec_func,'Gen. Freq.'],rwhds)
	print('%s\n%s'%(tab,disp_line))

	# finally (really) finish up
	stp = dat.datetime.now()
	print('GA Finished on %s\nExecution time: %s'%(stp,stp-stt))
	if out_file is not None:
		print('Files Saved with prefix %s'%save_prefix)
		QB.Diary(None)

	return (GA_BEST, save_prefix)  # JAH 20160215 added save_prefix output

def ReRunGA(prev_bests, seed_cnt, data_parms, objec_parms, params, out_file=None):
	"""
	With the GA, it's common to run many replications of the process, and view
	aggregate results over all to determine the optimal solution.  One way to
	do this is to seed the random population in each replication with some of
	the best solutions from the previous replication.  This function is a
	wrapper that automates this seeded replication by taking the specified number
	of top results and adding them into the seed_sols array in the params
	dict, then rerunning RunGA with the updated params.
	
	---
	Usage: best_solutions, out_file, new_params = ReRunGA(prev_bests, seed_cnt, data_parms, objec_parms, params, out_file)
	---
	prev_bests: best_solutions results array from the previous run
	seed_cnt: integer number of top best solutions from previous run to seed
	data_parms: see RunGA
	objec_parms: see RunGA
	params: see RunGA
	out_file: see RunGA
	best_solutions: see RunGA
	out_file: see RunGA
	new_params: params dict with the possibly updated seed_sols element
	---
	ex: data_parms = 'data = np.reshape(1+np.tile(np.arange(20),20),(20,20),order="C"); data_name = "428"'
		objec_parms = {'function':'np.sum','israndom':False,'data':'a','axis': None}
		params = {'seed_vars':[],'force_vars':[],'seed_sols':[],'init_perc':0.5,\
			'showtopsubs':5,'popul_size':25,'num_generns':100,'nochange_terminate':80,\
			'convgcrit':0.0001,'elitism':True,'mate_type':2,'prob_xover':0.75,\
			'xover_type':1,'prob_mutate':0.1,'prob_engineer':0.25,'optim_goal':1,\
			'plotflag':True,'printfreq':2,'randstate':42}
		out_file = '/home/ahowe42/QREDIS/out/demoGA'
		best_solutions = QG.RunGA(data_parms, objec_parms, params, out_file)[0]
		print(best_solutions[0,2:].T)
		new_best = QG.ReRunGA(best_solutions, 3, data_parms, objec_parms, params, out_file)[0]
		print(best_solutions[0,2:].T)
	JAH 20170405
	"""
	
	# ensure seed_cnt is a non-0 int
	if (type(seed_cnt) is not int) or not(seed_cnt > 0):
		raise ValueError("seed_cnt must be a positive integer!: %s"%ReRunGA.__doc__)
	# check inputs
	if (type(data_parms) is not dict) or (type(objec_parms) is not dict) or ((type(out_file) is not str) and (out_file is not None)):
		raise TypeError('Something wrong with RunGA inputs: %s'%RunGA.__doc__)		
	
	# get the top up to seed_cnt bool binary flags
	try:
		new_seed = prev_bests[:seed_cnt,2:]
	except TypeError:
		raise TypeError('prev_bests must be a 2-d array_like: %s'%ReRunGA.__doc__)
	except IndexError:
		raise IndexError('prev_bests must be a 2-d array_like: %s'%ReRunGA.__doc__)		
		
	# talk about it	
	varis = np.arange(new_seed.shape[1])+1
	print('Top up to %d Solutions (to be seeded)'%seed_cnt)
	for i in range(new_seed.shape[0]):
		print((varis[new_seed[i,:]==1]-1).tolist())
	
	# append these new seeds onto the existing seed_sols and unique it
	if (params['seed_sols'] is None):
		# seed was None
		seedSols = new_seed
	else:
		seedSols = np.vstack((params['seed_sols'],new_seed))
		# need to do all this to get unique rows
		tmp = np.ascontiguousarray(seedSols)
		uni = np.unique(tmp.view([('',tmp.dtype)]*tmp.shape[1]))
		seedSols = uni.view(seedSols.dtype).reshape((uni.shape[0],seedSols.shape[1]))
	
	# update params and rerun the GA
	params['seed_sols'] = seedSols
	(bestSolutions, saved) = RunGA(data_parms, objec_parms, params, out_file)
	return (bestSolutions, saved, params)
