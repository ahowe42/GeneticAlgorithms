function maxloglike = MVGaussLogLike(subset)
% maximized likelihood = MVGaussLogLike(subset model)
%  Compute the maximized log likelihood assuming multivariate normality for
%  a subset model.
%
%  Where
%  subset --- (n x p) specific subset of the entire data matrix.
%  maximized likelihood --- Scalar value of likelihood evaluated at MLEs
%     for subset.
%
%  JAH 20071128
%  Copyright Prof. Hamparsum Bozdogan & J. Andrew Howe
%  All rights reserved, see LICENSE.TXT

[n,p] = size(subset);   % sizes
sigma = cov(subset);    % get the covariance matrix of the subset
if p == 1   % computation is slightly different if univariate
    maxloglike = -(n*log(2*pi*sigma) + (1/sigma)*sum((subset - mean(subset)).^2))/2;
else
    maxloglike = -n*(p*log(2*pi) + log(det(sigma)) + p)/2;
end