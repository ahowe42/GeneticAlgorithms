function offspring = GAcrossover(popul,parents,xover_rate,xover_type)
% offspring = GAcrossover(population,parents,xover rate,xover type)
%  Performs chromosomal crossover on the current generation of a GA after
%  the solutions have been selected and paired for mating.
%
%  Where
%  population --- (n x p) matrix of current GA population
%  parents --- (n/2 x 2) matrix indicating pairs of solution indices to mate
%  xover rate --- Scalar probability of crossover in [0,1]
%  xover type --- Scalar type of crossover:
%     1 - single point - chromosomes are traded at a randomly generated point
%     2 - dual point - chromosomes are traded at 2 randomly generated points
%     3 - uniform - chromosomes are traded at a random number of points,
%        the points are selected using the same probability as that for crossover
%  offspring --- (n x p) matrix of next GA generation
%
%  Example: GAcrossover([0,1,1,0,1;0,1,0,0,1],[1,2],0.8,1)
%
%  See Also GAselect, GAmutation, GAengineering, GA10to2, GA2to10, GArealoptim
%
%  JAH 20071104
%  Copyright Prof. Hamparsum Bozdogan & J. Andrew Howe
%  All rights reserved, see LICENSE.TXT

if (nargin ~= 4) || (xover_rate < 0) || (xover_rate > 1)
    % wrong number arguments, crossover rate not probability
    fprintf('GAcrossover: INVALID USAGE-Please read the following instructions!\n'), help GAcrossover, return
end

[popul_size,binlen] = size(popul);  % pop size, binary encoding length
matepairs = size(parents,1);        % mating pairs
offspring1 = zeros(matepairs,binlen); offspring2 = offspring1;

switch xover_type
    case 1      % randomized single point
        for matecnt = 1:matepairs
            dad = popul(parents(matecnt,1),:);
            mom = popul(parents(matecnt,2),:);
            if xover_rate > rand;
                % cross them over
                xoverpoint = unidrnd(binlen,1,1);   % point at which we xover
                offspring1(matecnt,:) = [dad([1:xoverpoint]),mom([(xoverpoint+1):end])];
                offspring2(matecnt,:) = [mom([1:xoverpoint]),dad([(xoverpoint+1):end])];
            else
                % just duplicate into next generation
                offspring1(matecnt,:) = dad;
                offspring2(matecnt,:) = mom;
            end
        end         % mating pairs loop
    case 2      % randomized dual point
        for matecnt = 1:matepairs
            dad = popul(parents(matecnt,1),:);
            mom = popul(parents(matecnt,2),:);
            if xover_rate > rand;
                % cross them over
                xoverpoints = sort(unidrnd(binlen,1,2));    % where we crossover
                offspring1(matecnt,:) = [dad([1:xoverpoints(1)]),...
                    mom([(xoverpoints(1)+1):xoverpoints(2)]),dad([(xoverpoints(2)+1):end])];
                offspring2(matecnt,:) = [mom([1:xoverpoints(1)]),...
                    dad([(xoverpoints(1)+1):xoverpoints(2)]),mom([(xoverpoints(2)+1):end])];
            else
                % just duplicate into next generation
                offspring1(matecnt,:) = dad;
                offspring2(matecnt,:) = mom;
            end
        end         % mating pairs loop
    case 3      % uniform
        for matecnt = 1:matepairs
            dad = popul(parents(matecnt,1),:);
            mom = popul(parents(matecnt,2),:);
            if xover_rate > rand;
                % cross them over
                % pick locations to xover at the same rate as that used for crossover
                xoverpoints = xover_rate > rand(1,binlen);
                offspring1(matecnt,:) = dad.*xoverpoints + mom.*not(xoverpoints);
                offspring2(matecnt,:) = mom.*xoverpoints + dad.*not(xoverpoints);
            else
                % just duplicate into next generation
                offspring1(matecnt,:) = dad;
                offspring2(matecnt,:) = mom;
            end
        end         % mating pairs loop        
end
offspring = [offspring1;offspring2] == 1;