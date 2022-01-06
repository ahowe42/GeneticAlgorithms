function [lb,ub] = EstParamRanges(dist,data)
% [lower bound(s), upper bound(s)] = EstParamRanges(distribution,data)
%  This is a general function for computing the possible ranges for the
%  parameters of various distributions.  For the Gamma distribution, this
%  calls the MLE function in the Matlab statistics toolbox.
%
%  The exponential distribution uses the avg. lifetime parameterization:
%     f(x) = (1/b)*exp(-x/b) = exppdf(x,b)
%  The gamma has 1/b instead of the usual b:
%     f(x) = (x.^(a-1)).*exp(-x/b)/((b^a)*gamma(a)) = gampdf(x,a,b)
%  The weibull has a^-b instead of the usual a:
%     f(x) = a*b*(x.^(b-1)).*exp(-a*x.^b) = weibpdf(x,a,b)
%
%  Where
%  parameters --- Vector holding parameters for specified distribution.
%  data --- (n x 1) vector of data.
%  distribution  --- Character code from one of these:
%     'NRM' - Normal(mu,sigma)      'GAM' - Gamma(a,b)
%     'LOG' - Lognormal(mu,sigma)   'EXP' - Exponential(b)
%     'WEI' - Wiebull(a,b)          'CHI' - ChiSquare(nu)
%     'STU' - t(nu)                 'CAU' - Cauchy(theta)
%     'LPL' = Laplace(mu,sigma)     'PXP' = PowerExponential(mu,sigma,beta)
%     'PAR' = Pareto(c)
%  lower bound(s) --- (1 x p) vector holding the lower bounds for the parameters.
%  upper bound(s) --- (1 x p) vector holding the upper bounds for the parameters.
%
%  Example: EstParamRanges('NRM',SimDistData('NRM',500,[2,0.5],0))
%
%  See Also ComputeLogLike, ComputePDF, SimDistData
%
%  JAH 20071027
%  Copyright Prof. Hamparsum Bozdogan & J. Andrew Howe
%  All rights reserved, see LICENSE.TXT

if (nargin ~= 2) || (sum(size(data) == 1) == 0)
    % wrong number inputs, data a matrix
    fprintf('EstParamRanges: INVALID USAGE-Please read the following instructions!\n'), help EstParamRanges, return
end

n = length(data); xbar = mean(data); s = std(data);
switch upper(dist)
    case 'NRM'
        mnci = xbar + [-2,2]*s/sqrt(n);
        lb = [min(mnci),0.0001];
        ub = [max(mnci),s*1.5];
    case 'GAM'
        phat = mle(data,'distribution','gamma');
        lb = 0.5*phat; ub = 1.5*phat;
        %lb = [0.5*xbar/s,0.0001]; ub = [1.5*xbar/s,s*1.5];
    case 'LOG'
        mnci = mean(log(data)) + [-2,2]*std(log(data))/sqrt(n);
        lb = [min(mnci),0.0001];
        ub = [max(mnci),s*1.5];
    case 'EXP'      % this is the (1/b)exp(-x/b) parameterization
        lambdaci = xbar + [-2,2]*(xbar^2)/sqrt(n);
        lb = max([0,min(lambdaci)]); ub = max(lambdaci);
    case 'WEI'      % ab[x^(b-1)]exp(-ax^b) parameterization
        lb = [0.5/xbar,0.0001]; ub = [1.5/xbar,1.5/s];
    case 'CHI'
        lb = 0; ub = xbar + 2*2*xbar/sqrt(n);
    case 'STU'
        lb = 0; ub = abs(1.5*(-2*s^2)/(1 - s^2));
    case 'CAU'
        mnci = xbar + [-2,2]*s/sqrt(n);
        lb = min(mnci); ub = max(mnci);
    case 'LPL'
        mnci = xbar + [-2,2]*s/sqrt(n);
        lb = [min(mnci),0.0001];
        ub = [max(mnci),s*1.5];
    case 'PXP'
        mnci = xbar + [-2,2]*s/sqrt(n);
        lb = [min(mnci),0.0001,0.1];
        ub = [max(mnci),s*1.5,5];
    case 'PAR'
        lb = 0; ub = 1.5*n/sum(log(1 + data));
end