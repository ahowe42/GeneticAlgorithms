function [x,y] = ComputePDF(dist,parms,datminmax)
% [x,y] = ComputeLogLike(distribution,parameters,data min and max)
%  This is a general function for computing the densities with respect
%  to a specified distribution, for some data, such that the range output
%  in x encompasses the data and the distribution. Note that though you can
%  generate uniform, beta, or F data with SimDistData, this function can't 
%  compute the density for them.
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
%     'NRM' - Normal(mu,sigma)      'GAM' - Gamma(a,b)
%     'LOG' - Lognormal(mu,sigma)   'EXP' - Exponential(b)
%     'WEI' - Wiebull(a,b)          'CHI' - ChiSquare(nu)
%     'STU' - t(nu)                 'CAU' - Cauchy(theta)
%     'LPL' = Laplace(mu,sigma)     'PXP' = PowerExponential(mu,sigma,beta)
%     'PAR' = Pareto(c)
%  parameters --- Vector holding parameters for specified distribution.
%  data min and max --- (1 x 2) vector holding min and max of data.
%  x --- (100 x 1) vector of plotting range.
%  y --- (100 x 1) vector of densities for x.
%
%  Example: [x,y] = ComputePDF('NRM',[0,1],[-4,4]); plot(x,y)
%  See Also ComputeLogLike, EstParamRanges, SimDistData
%
%  JAH 20071027
%  Copyright Prof. Hamparsum Bozdogan & J. Andrew Howe
%  All rights reserved, see LICENSE.TXT

if (nargin ~= 3) || (sum(size(parms) == 1) == 0)
    % wrong number inputs, params a matrix
    fprintf('ComputePDF: INVALID USAGE-Please read the following instructions!\n'), help ComputePDF, return
end

mn = min(datminmax); mx = max(datminmax);
alpha = 0.01; ci = [alpha/2,1-alpha/2];

if isequal(upper(dist),'LPL')
    % laplace is pe with beta = 0.5;
    parms = [parms,0.5]; dist = 'PXP';
end

switch upper(dist)
    case 'NRM'
        if length(parms) ~= 2
            disp('Normal must have 2 parameters'), return;
        end
        drng = norminv(ci,parms(1),parms(2));
        x = linspace(min([drng,mn]),max([drng,mx]),500);
        y = normpdf(x,parms(1),parms(2));
    case 'GAM'
        if length(parms) ~= 2
            disp('Gamma must have 2 parameters'), return;
        end
        drng = gaminv(ci(2),parms(1),parms(2));
        x = linspace(0,max([drng,mx]),500);
        y = gampdf(x,parms(1),parms(2));
    case 'LOG'
        if length(parms) ~= 2
            disp('Lognormal must have 2 parameters'), return;
        end
        drng = logninv(ci(2),parms(1),parms(2));
        x = linspace(0,max([drng,mx]),500);
        y = lognpdf(x,parms(1),parms(2));
    case 'EXP'      % this is the (1/b)exp(-x/b) parameterization
        if length(parms) ~= 1
            disp('Exponential must have 1 parameter'), return;
        end
        drng = expinv(ci(2),parms);
        x = linspace(0,max([drng,mx]),500);
        y = exppdf(x,parms);
    case 'WEI'      % ab[x^(b-1)]exp(-ax^b) parameterization
        if length(parms) ~= 2
            disp('Weibull must have 2 parameters'), return;
        end
        drng = weibinv(ci(2),parms(1),parms(2));
        x = linspace(0,max([drng,mx]),500);
        y = weibpdf(x,parms(1),parms(2));
    case 'CHI'
        if length(parms) ~= 1
            disp('Chi-squared must have 1 parameter'), return;
        end
        drng = chi2inv(ci(2),parms);
        x = linspace(0,max([drng,mx]),500);
        y = chi2pdf(x,parms);
    case 'STU'
        if length(parms) ~= 1
            disp('Studen''ts t must have 1 parameter'), return;
        end
        drng = tinv(ci,parms);
        x = linspace(min([drng,mn]),max([drng,mx]),500);
        y = tpdf(x,parms);
    case 'CAU'
        if length(parms) ~= 1
            disp('Cauchy must have 1 parameter'), return;
        end
        drng = tan(pi*ci + atan(-Inf - parms)) + parms;
        x = linspace(min([drng,mn]),max([drng,mx]),500);
        y = 1./((1 + (x - parms).^2)*pi);
    case 'PXP'
        if length(parms) ~= 3
            disp('Power Exponential must have 3 parameters'), return;
        end
        mu = parms(1); sigma = parms(2); beta = parms(3);
        x = linspace(min([-6,mn]),max([6,mx]),500);
        y = exp(-0.5*abs((x - mu)/sigma).^(2*beta))/(sigma*gamma(1+1/(2*beta))*(2^(1+1/(2*beta))));
    case 'PAR'
        if length(parms) ~= 1
            disp('Pareto must have 1 parameter'), return;
        end
        drng = (1 - [0,ci(2)]).^(-1/parms) - 1;
        x = linspace(0,max([drng,mx]),500);
        y = parms*(1 + x).^(-(parms + 1));
end
