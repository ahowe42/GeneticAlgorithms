function [X,rnd] = SimDistData(dist,n,parms,rnd)
% [data,state] = SimDistData(distribution,sample size,parameters,state)
%  This is a general function for simulating data, it calls other
%  Matlab random generators.
%
%  The exponential distribution uses the avg. lifetime parameterization:
%     f(x) = (1/b)*exp(-x/b) = exppdf(x,b)
%  The gamma has 1/b instead of the usual b:
%     f(x) = (x.^(a-1)).*exp(-x/b)/((b^a)*gamma(a)) = gampdf(x,a,b)
%  The weibull has a^-b instead of the usual a:
%     f(x) = a*b*(x.^(b-1)).*exp(-a*x.^b) = weibpdf(x,a,b)
%
%  Where
%  distribution  --- Character code from one of these:
%     'UNI' - Uniform(0,b)      'NRM' - Normal(mu,sigma)
%     'GAM' - Gamma(a,b)        'LOG' - Lognormal(mu,sigma)
%     'EXP' - Exponential(b)    'WEI' - Wiebull(a,b)
%     'CHI' - ChiSquare(nu)     'BET' - Beta(a,b)
%     'STU' - t(nu)             'CAU' - Cauchy(theta)
%     'LPL' = Laplace(mu,sigma) 'PXP' = PowerExponential(mu,sigma,beta)
%     'PAR' = Pareto(c)         'F' = F(nu1,nu2)
%  sample size --- Scalar number observations required.
%  parameters --- Vector holding parameters for specified distribution.
%  state --- Scalar randomizer state (optional):
%     -1 = no randomization
%      0 = automatic: state = sum(clock*1000000) (default)
%      ? = pass in your own value
%  data --- (n x 1) vector of data.
%  state --- State of randomizer used, if set.
%
%  Example: X = SimDistData('CHI',1000,3,-1); hist(X)
%  See Also ComputeLogLike, ComputePDF, EstParamRanges
%
%  JAH 20071027
%  Copyright Prof. Hamparsum Bozdogan & J. Andrew Howe
%  All rights reserved, see LICENSE.TXT

if nargin == 0  % no inputs, pass out list of dists, [num parms, x support]
    X = {'UNI','b';'NRM','mu,sigma';'GAM','a,b';'LOG','mu,sigma';...
        'EXP','b';'WEI','a,b';'CHI','nu';'BET','a,b';'STU','nu';...
        'CAU','theta';'LPL','mu,sigma';'PXP','mu,sigma,beta';...
        'PAR','c';'F','nu1,nu2'};
    % support: 0 = entire real line; 1 = positive real
    rnd = [1,1;2,0;2,1;2,1;1,1;2,1;1,1;2,1;1,0;1,0;2,0;3,0;1,1;2,1];
    return;
end

if ((nargin ~= 3) && (nargin ~= 4)) || (sum(size(parms) == 1) == 0)
    % wrong number inputs, params a matrix
    fprintf('SimDistData: INVALID USAGE-Please read the following instructions!\n'), help SimDistData, return
end

% randomize?
if nargin == 3; rnd = 0; end;
if rnd == -1
    % do nothing
elseif rnd == 0
    rnd = sum(clock*1000000);
    rand('state',rnd); randn('state',rnd);
else
    rand('state',rnd); randn('state',rnd);
end

if isequal(upper(dist),'LPL')
    % laplace is pe with beta = 0.5;
    parms = [parms,0.5]; dist = 'PXP';
end

switch upper(dist)
    case 'UNI'
        if length(parms) ~= 1
            disp('Uniform must have 1 parameter'), return;
        end
        X = parms*rand(n,1);
    case 'NRM'
        if length(parms) ~= 2
            disp('Normal must have 2 parameters'), return;
        end
        X = randn(n,1)*parms(2) + parms(1);
    case 'GAM'
        if length(parms) ~= 2
            disp('Gamma must have 2 parameters'), return;
        end
        X = gamrnd(parms(1),parms(2),n,1);
    case 'LOG'
        if length(parms) ~= 2
            disp('Lognormal must have 2 parameters'), return;
        end
        X = lognrnd(parms(1), parms(2),n,1);
    case 'EXP'      % this is the (1/b)exp(-x/b) parameterization
        if length(parms) ~= 1
            disp('Exponential must have 1 parameter'), return;
        end
        X = exprnd(parms,n,1);
    case 'WEI'      % ab[x^(b-1)]exp(-ax^b) parameterization
        if length(parms) ~= 2
            disp('Weibull must have 2 parameters'), return;
        end
        X = weibrnd(parms(1),parms(2),n,1);
    case 'CHI'
        if length(parms) ~= 1
            disp('Chi-squared must have 1 parameter'), return;
        end
        X = chi2rnd(parms,n,1);
    case 'BET'
        if length(parms) ~= 2
            disp('Beta must have 2 parameters'), return;
        end
        X = betarnd(parms(1),parms(2),n,1);
    case 'STU'
        if length(parms) ~= 1
            disp('Studen''ts t must have 1 parameter'), return;
        end
        X = trnd(parms,n,1);
    case 'CAU'
        if length(parms) ~= 1
            disp('Cauchy must have 1 parameter'), return;
        end
        p = rand(n,1);
        X = tan(pi*p + atan(-Inf - parms)) + parms;
    case 'PXP'
        if length(parms) ~= 3
            disp('Power Exponential must have 3 parameters'), return;
        end
        % this formulation requires the variance, not stdev
        mu = parms(1); sigmasq = parms(2)^2; beta = parms(3);
        contst = gamma(0.5)/(sqrt(pi)*gamma(1+1/(2*beta))*2^(1+1/(2*beta)));
        X = zeros(n,1); k = 0;
        while k < n
            tmp = 8*sigmasq*rand() + mu - 4*sigmasq;
            f1 = contst*sqrt(sigmasq)*exp((-1/2)*(((tmp - mu)^2)/sigmasq)^(beta));
            f2 = contst*rand();
            if f2 <= f1
                k = k + 1;          
                X(k) = tmp;
            end
        end
    case 'PAR'
        if length(parms) ~= 1
            disp('Pareto must have 1 parameter'), return;
        end
        p = rand(n,1);
        X = (1 - p).^(-1/parms) - 1;
    case 'F'
        if length(parms) ~= 2
            disp('F must have 2 parameters'), return;
        end
        X = frnd(parms(1),parms(2),n,1);
end