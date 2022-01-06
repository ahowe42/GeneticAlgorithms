function newpop = GAmutation(popul,mutat_rate);
% mutated = GAmutation(population, mutation rate);
%  Perform random mutation on a population of chromosomes, this would
%  usually be performed after the previous generation has mated and
%  produced the next generation.
%
%  Where
%  population --- (n x p) matrix of n offspring GA solutions
%  mutation rate --- Scalar probability of mutation in [0,1]
%  mutated --- Matrix of same size as population holding mutated offspring
%
%  Example: GAmutation([0,1,1,0,1;0,1,0,0,1],0.1)
%
%  See Also GAcrossover, GAselect, GAengineering, GA2to10, GA10to2, GArealoptim
%
%  J. Andrew Howe 20071103
%  Copyright Prof. Hamparsum Bozdogan & J. Andrew Howe
%  All rights reserved, see LICENSE.TXT

if (nargin ~= 2) || (mutat_rate < 0) || (mutat_rate > 1)
    % wrong number arguments, mutation rate not probability
    fprintf('GAmutation: INVALID USAGE-Please read the following instructions!\n'), help GAmutation, return
end

[num_chrom, p] = size(popul);
mutation_chances = mutat_rate > rand(num_chrom,p);
newpop = popul;
newpop(mutation_chances) = not(newpop(mutation_chances));