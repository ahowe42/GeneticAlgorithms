function newoffspring = GAengineering(prevgenbest, currgenbest, currgenoff, eng_rate)
% new offspring = GAengineering(previous best, current best, current offspring, engineer rate)
%  This implements GA engineering, the point of which is to limit the
%  variability between GA runs. It takes the best solution from the
%  previous generation and the best solution from the current generation, and
%  finds the difference. Where they are different, the bits from the
%  previous best are inserted into the offspring of the current generation
%  with the specified probability. This should only be called if the best
%  from the previous generation is better than the current best.
%
%  Where
%  previous best --- (1 x p) vector chromosome of previous generation best
%  current best --- (1 x p) vector chromosome of current generation best
%  current offspring --- (n x p) matrix of all offspring from current generation
%  engineer rate --- Scalar probability of engineering in [0,1]
%  new offspring --- (n x p) matrix of engineered offspring from current generation
%
%  See Also GAcrossover, GAmutation, GAselect, GA10to2, GA2to10, GArealoptim
%
%  JAH 20071128
%  Copyright Prof. Hamparsum Bozdogan & J. Andrew Howe
%  All rights reserved, see LICENSE.TXT

if (nargin ~= 4) || (eng_rate < 0) || (eng_rate > 1)
    % wrong number arguments, engineering prob not probability
    fprintf('GAengineering: INVALID USAGE-Please read the following instructions!\n'), help GAengineering, return
end

popul_size = size(currgenoff,1);    % population size of next generation
newoffspring = currgenoff;

% get differences
diff = xor(prevgenbest,currgenbest);
if sum(diff) == 0; return; end;     % no change, so just exit
diffloc = find(diff);
diff = prevgenbest(diffloc);

% now do the engineering
eng_chances = eng_rate > rand(popul_size,1);    % chance of engineering
if sum(eng_chances) == 0; return; end;          % noone changing, so just exit
newoffspring(eng_chances,diffloc) = repmat(diff,sum(eng_chances),1);