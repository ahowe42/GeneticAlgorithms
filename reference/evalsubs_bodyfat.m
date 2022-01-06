% usage: evalsubs_bodyfat
% Evaluate the multivariate gaussian log likelihood for all nonempty
% subsets of the bodyfat data, and display the best 10.
%
% JAH 20071128
% Copyright Prof. Hamparsum Bozdogan & J. Andrew Howe
% All rights reserved, see LICENSE.TXT

clc, clear
% get location
mydir = dbstack; mydir = mydir(end);
mydir = which(mydir.file);
tmp = strfind(mydir,'\');
mydir = mydir([1:(tmp(end)-1)]);

% get data
data_path = [mydir,'\data\'];
data_file = 'BodyFatData.m';
data = dlmread([data_path,data_file]);
[n,p] = size(data);

% get all subsets
b = varsubset(13);
sizes = b(:,1);
bins = (b(:,[2:end]) == 1);
numsubs = length(b);

% get scores
scores = zeros(numsubs,1);
for subcnt = 1:numsubs
    scores(subcnt) = MVGaussLogLike(data(:,bins(subcnt,:)));
end                 % subsets loop
scorebin = [scores,bins];
scorebin = sortrows(scorebin,-1);   % sort with best at top

% display best subsets and scores
top_best = 10;
binloc = [2:(p + 1)]; vars = [1:p]; rwhds = cell(top_best,1);
for pcnt = 1:top_best
    if pcnt <= top_best
        vrs = sprintf('%d,',find(vars.*scorebin(pcnt,binloc)));
        rwhds{pcnt,1} = ['{',vrs([1:(end-1)]),'}'];
    end
end                 % results loop
disp(table2str({'MVGaussLogLike'},scorebin([1:top_best],1),{'%0.3f'},0,rwhds));