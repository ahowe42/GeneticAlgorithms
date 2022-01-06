function parents = GAselect(pop_fitness, maxmin, meth)
% parents = GAselect(fitness scores, objective, method)
%  Use fitness scores to select chromosomes from a generation in preparation
%  for mating to produce the following generation.  If the roulette method
%  is selected, chromosomes are selected according to a biased roulette
%  such that the most fit individuals are overrepresented.  The selected
%  chromosomes are then randomly paired for mating.  If the sorted method
%  is selected, the chromosomes are paired for mating after being sorted by
%  their scores (mate(1,2), mate(3,4), etc...).
%
%  Where
%  fitness scores --- (n x 1) vector containing fitness scores for entire
%     population, n should be even; if it is not, the index of the most fit
%     chromosome will be inserted one extra time.
%  objective --- Scalar indicating type of optimization:
%     maximize the fitness score - -1
%     minimize the fitness score - 1
%  method --- 1: sorted, 2: roulette
%  parents --- (n/2 x 2) matrix indicating pairs of solution indices to mate
%
%  Example: fit = rand(10,1); GAselect(fit,-1,1)
%
%  See Also GAcrossover, GAmutation, GAengineering, GA10to2, GA2to10, GArealoptim
%
%  JAH 20071105
%  Copyright Prof. Hamparsum Bozdogan & J. Andrew Howe
%  All rights reserved, see LICENSE.TXT

if (nargin ~= 3) || (abs(maxmin) ~= 1) || (sum(meth == [1,2]) ~= 1)
    % wrong number arguments, objective not 1 or -1, method not 1 or 2
    fprintf('GAselect: INVALID USAGE-Please read the following instructions!\n'), help GAselect, return
end

popul_size = length(pop_fitness);    % population size

% sort fitness scores, if maxmin is positive, this will sort the scores
% descending, with the lowest at the front, if maxmin is negative, it is
% essentially sorting ascending, with the largest at the front, either way,
% the best chromosomes are associated with largest roulette bins
[val, stdindex] = sort(pop_fitness*maxmin);

if meth == 1    % sorted method
    % if population size is odd, insert the best in 3rd spot
    if rem(popul_size,2) == 1
        popul_size = popul_size + 1;
        stdindex = [stdindex([1,2]);stdindex(1);stdindex([3:end])];
    end
    parents = reshape(stdindex,2,popul_size/2)';
else            % roulette method
    % prepare bins for roulette - bigger bins at the beginning with lower scores
    bins =  cumsum([popul_size:-1:1]/(popul_size*(popul_size + 1)/2))';
    % roulette selection - find into which bin the random falls
    new_pop = sum(repmat(rand(1,popul_size),popul_size,1) >= repmat(bins,1,popul_size))+1;
    % if population size is odd, insert the best again
    if rem(popul_size,2) == 1
        popul_size = popul_size + 1;
        new_pop = [stdindex(1),new_pop];
    end
    % randomly permute rows, then reform to make parents
    [jnk,neword] = sort(rand(1,popul_size));
    parents = reshape(new_pop(neword),popul_size/2,2);
end