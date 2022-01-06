function loglike = MVN_Like_SigmaEncoded(parms, data)
%  Compute the log likelihood for a multivariate normal density fit to some
%  data, where the parameters are encoded as a vector with the first p
%  entries for the mean, and the remainder representing the covariance
%  matrix encoded using SigmaEncodeDecode
%
%  JAH 20100423
%  Copyright Prof. Hamparsum Bozdogan & J. Andrew Howe
%  All rights reserved, see LICENSE.TXT

% extract the parameters
[n,p] = size(data);
mean = parms([1:p]);
sigma = SigmaEncDec(parms([(p+1):end]),p);

% ensure sigma is positive definite
[R,p] = chol(sigma);

% compute the densities & the log likelihood
if p ~= 0;      % sigma is not positive definite, so just finish
    loglike = -inf;
else            % sigma is positive definite
    dens = mvnpdf(data,mean,sigma);
    loglike = sum(log(dens));
end