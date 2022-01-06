function realval = GA2to10(binval,bits,LB,UB)
% real values = GA2to10(binary values, number bits, lower bounds, upper bounds)
%  Convert a binary vector possibly encoding multiple real values back into
%  a vector of real values.  Use GA10to2 to encode.
%
%  Where
%  binary values --- (1xsum(number bits)) vector holding real values encoded as binary
%  number bits --- (1xp) vector with number of bits used to encode each real value
%  lower bounds --- (1xp) vector with lower bound of range for each real value
%  upper bounds --- (1xp) vector with upper bound of range for each real value
%  real values --- (1xp) vector of each decoded real value
%
%  Example: B = [16,16,16]; L = [-1,-1,-1]; U = -L; GA2to10(GA10to2([0,0.63,1],B,U,L),B,U,L)
%
%  See Also GAcrossover, GAselect, GAengineering, GAmutation, GA10to2, GArealoptim
%
%  J. Andrew Howe 20071103
%  Copyright Prof. Hamparsum Bozdogan & J. Andrew Howe
%  All rights reserved, see LICENSE.TXT

Ll = length(LB); Ul = length(UB); Bl = length(bits); Sl = length(binval);

if (nargin ~= 4) || (Ll ~= Ul) || (Ll ~= Bl) || (Ul ~= Bl) || (sum(bits) ~= Sl)
    % wrong number arguments, sizes don't match
    fprintf('GA2to10: INVALID USAGE-Please read the following instructions!\n'), help GA2to10, return
end

% separate the binary string into one binary string for each real value
binlims = cumsum(bits(:));
binlims = [[1;binlims([1:(end-1)])+1],binlims];

realval = zeros(1,Bl);
for rcnt = 1:Bl
    exps = (2.^[(bits(rcnt) - 1):-1:0]);    % get the power with each binary digit
    bin = binval([binlims(rcnt,1):binlims(rcnt,2)]);
    realval(rcnt) = LB(rcnt) + (UB(rcnt) - LB(rcnt))*sum(exps.*bin)/sum(exps);
end                 % real values encoded as binary loop