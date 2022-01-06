function [loglike,numparms] = ComputeLogLike(parms,data,dist)
% log likelihood = ComputeLogLike(parameters,data,distribution)
%  This is a general function for computing the log likelihood with respect
%  to a specified distribution, for some data. Note that though you can
%  generate uniform, beta, or F data with SimDistData, this function can't 
%  compute the log likelihood for fitting those distributions.
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
%  log likelihood --- log likelihood for distribution at parameters.
%
%  Example: ll = ComputeLogLike([0,1],SimDistData('NRM',1000,[0,1],-1),'NRM')
%  See Also ComputePDF, EstParamRanges, SimDistData
%
%  JAH 20071027
%  Copyright Prof. Hamparsum Bozdogan & J. Andrew Howe
%  All rights reserved, see LICENSE.TXT

if nargin == 0  % no inputs, pass out list of dists, [num parms, x support]
    loglike = {'NRM','mu,sigma';'GAM','a,b';'LOG','mu,sigma';...
        'EXP','b';'WEI','a,b';'CHI','nu';'STU','nu';'CAU','theta';...
        'LPL','mu,sigma';'PXP','mu,sigma,beta';'PAR','c'};
    % support: 0 = entire real line; 1 = positive real
    numparms = [2,0;2,1;2,1;1,1;2,1;1,1;1,0;1,0;2,0;3,0;1,1];
    return;
end

if (nargin ~= 3) || (sum(size(parms) == 1) == 0)
    % wrong number inputs, params a matrix
    fprintf('ComputeLogLike: INVALID USAGE-Please read the following instructions!\n'), help ComputeLogLike, return
end

if isequal(upper(dist),'LPL')
    % laplace is pe with beta = 0.5;
    parms = [parms,0.5]; dist = 'PXP';
end

n = length(data);
switch upper(dist)
    case 'NRM'
        if length(parms) ~= 2
            disp('Normal must have 2 parameters'), return;
        end
        mu = parms(1); sigsq = parms(2)^2;
        loglike = -0.5*(n*log(2*pi*sigsq) + (1/sigsq)*sum((data - mu).^2));
    case 'GAM'  % Matlab gamma has b^-1 instead of the usual b
        if length(parms) ~= 2
            disp('Gamma must have 2 parameters'), return;
        end
        a = parms(1); b = parms(2);
        loglike = -n*log((b^a)*gamma(a)) + (a-1)*sum(log(data)) - sum(data)/b;
    case 'LOG'
        if length(parms) ~= 2
            disp('Lognormal must have 2 parameters'), return;
        end
        mu = parms(1); sigsq = parms(2)^2;
        loglike = -0.5*(n*log(2*pi*sigsq) + 2*sum(log(data)) + (1/sigsq)*sum((log(data) - mu).^2));
    case 'EXP'      % this is the (1/b)exp(-x/b) parameterization
        if length(parms) ~= 1
            disp('Exponential must have 1 parameter'), return;
        end
        loglike = -n*log(parms) - sum(data)/parms;
    case 'WEI'      % ab[x^(b-1)]exp(-ax^b) parameterization
        if length(parms) ~= 2
            disp('Weibull must have 2 parameters'), return;
        end
        a = parms(1); b = parms(2);
        loglike = n*log(b/(a^b)) + (b-1)*sum(log(data)) - sum(data.^b)/(a^b);
    case 'CHI'
        if length(parms) ~= 1
            disp('Chi-squared must have 1 parameter'), return;
        end
        loglike = -n*log(gamma(parms/2)*2^(parms/2)) + (parms/2 - 1)*sum(log(data)) - sum(data)/2;
    case 'STU'
        if length(parms) ~= 1
            disp('Studen''ts t must have 1 parameter'), return;
        end
        nurat = (parms + 1)/2;
        loglike = n*log(gamma(nurat)/(sqrt(pi*parms)*gamma(parms/2))) - nurat*sum(log(1 + (data.^2)/parms));
    case 'CAU'
        if length(parms) ~= 1
            disp('Cauchy must have 1 parameter'), return;
        end
        loglike = -n*log(pi) - sum(log(1 + (data - parms).^2));
    case 'PXP'
        if length(parms) ~= 3
            disp('Power Exponential must have 3 parameters'), return;
        end
        mu = parms(1); sigma = parms(2); beta = parms(3);
        loglike = -n*log(sigma*gamma(1 + 1/(2*beta))*(2^(1 + 1/(2*beta)))) - 0.5*sum(abs((data - mu)/sigma).^(2*beta));
    case 'PAR'
        if length(parms) ~= 1
            disp('Pareto must have 1 parameter'), return;
        end
        loglike = n*log(parms) - (parms+1)*sum(log(1 + data));
end