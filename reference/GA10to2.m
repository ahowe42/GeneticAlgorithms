function binval = GA10to2(realval,bits,LB,UB)
% binary values = GA10to2(real values, number bits, lower bounds, upper bounds)
%  Convert a vector of real values into a binary vector that can be
%  operated on by the Genetic Algorithm. Use GA2to10 to decode.
%
%  Where
%  real values --- (1xp) vector holding real values to encode as a single binary vector
%  number bits --- (1xp) vector with number of bits used to encode each
%     real value (if all the same, just pass it in once)
%  lower bounds --- (1xp) vector with lower bound of range for each real value
%  upper bounds --- (1xp) vector with upper bound of range for each real value
%  binary values --- (1xsum(number bits)) vector holding all real values in one binary string
%
%  Example: GA10to2([0,0.63,1],16,[-1,-1,-1],[1,1,1])
%
%  See Also GAcrossover, GAselect, GAengineering, GAmutation, GA2to10, GArealoptim
%
%  J. Andrew Howe 20071103
%  Copyright Prof. Hamparsum Bozdogan & J. Andrew Howe
%  All rights reserved, see LICENSE.TXT

Ll = length(LB); Ul = length(UB); Rl = length(realval); Bl = length(bits);

if (nargin ~= 4) || (Ll ~= Ul) || (Rl ~= Ll) || ((Bl ~= Ll) && (Bl ~= 1))
    % wrong number arguments, sizes don't match
    fprintf('GA10to2: INVALID USAGE-Please read the following instructions!\n'), help GA10to2, return
end

if (Bl == 1); bits = repmat(bits,1,Ll); end;                    % replicated bits

binval = [];
for rcnt = 1:length(realval)
    stepamt = (UB(rcnt) - LB(rcnt))/(2^(bits(rcnt))-1);     % partition range per bit
    numsteps = (realval(rcnt) - LB(rcnt))/stepamt;          % get relative distance from lower bound
    binval = [binval,dec2bin(numsteps,bits(rcnt)) == '1'];  % encode
end                 % real values encoded as binary loop
binval = (binval == 1);