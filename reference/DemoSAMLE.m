% usage: DemoSAMLE
% Demonstrate using adaptive simulated annealing to perform maximum
% likelihood estimation for 4 distributions: Gaussian, Chi-squared, Gamma,
% and Power Exponential.
%
% JAH 20071205
% Copyright Prof. Hamparsum Bozdogan & J. Andrew Howe
% All rights reserved, see LICENSE.TXT

% find and store where I am located
mydir = dbstack; mydir = mydir(end);
mydir = which(mydir.file);
tmp = strfind(mydir,'\');
mydir = mydir([1:(tmp(end)-1)]);
lin = repmat('-',1,75);
simstt = clock();
stt = sprintf('%4.0f%02.0f%02.0f_%02.0f%02.0f%02.0f',simstt);

% save output
dryname = ['SAMLE_',stt,'.out']; diary([mydir,'\',dryname]);
disp(repmat('@',1,75)),disp(['DemoSAMLE began on ',stt])


% normal
disp(' '), disp(lin), disp('Fitting To Normal Distribution')
DST = 'NRM';
true = [5,2];
X = SimDistData(DST,1000,true);
truemax = ComputeLogLike(true,X,DST);
% run the SA
[MLE,maxloglike] = SimAnneal_MLE([-20,0.25],[-1,1],X,DST,[100000,100,0,1,1]);
disp(table2str({'mu';'sigma'},[true;MLE],{'%0.2f'},0,{'True';'MLE'}))
disp(sprintf('Maximized Log Likelihood at\n\tTrue = %0.3f\n\tMLE = %0.3f',truemax,maxloglike))
% plot best distribution
figure
[e,c,f] = DensityHist(X,15,1);
[x,y] = ComputePDF(DST,MLE,[e(1),e(end)]);
hold on, plot(x,y,'r-'),hold off
title(['Data with Best Model: ',DST,'(',... 
    strrep(strtrim(sprintf('%0.2f ',MLE)),' ',','),')'])
disp(lin)

% chi-squared
disp(' '), disp(lin), disp('Fitting To Chi-Squared Distribution')
DST = 'CHI';
true = 3;
X = SimDistData(DST,1000,true);
truemax = ComputeLogLike(true,X,DST);
% run the SA
[MLE,maxloglike] = SimAnneal_MLE(50,1,X,DST,[10000,100,0,1,1]);
disp(table2str({'nu'},[true;MLE],{'%0.2f'},0,{'True';'MLE'}))
disp(sprintf('Maximized Log Likelihood at\n\tTrue = %0.3f\n\tMLE = %0.3f',truemax,maxloglike))
% plot best distribution
figure
[e,c,f] = DensityHist(X,15,1);
[x,y] = ComputePDF(DST,MLE,[e(1),e(end)]);
hold on, plot(x,y,'r-'),hold off
title(['Data with Best Model: ',DST,'(',... 
    strrep(strtrim(sprintf('%0.2f ',MLE)),' ',','),')'])
disp(lin)

% gamma
disp(' '), disp(lin), disp('Fitting To Gamma Distribution')
DST = 'GAM';
true = [2,0.5];
X = SimDistData(DST,1000,true);
truemax = ComputeLogLike(true,X,DST);
% run the SA
[MLE,maxloglike] = SimAnneal_MLE([5,5],[1,1],X,DST,[100000,100,0,1,1]);
disp(table2str({'a';'b'},[true;MLE],{'%0.2f'},0,{'True';'MLE'}))
disp(sprintf('Maximized Log Likelihood at\n\tTrue = %0.3f\n\tMLE = %0.3f',truemax,maxloglike))
% plot best distribution
figure
[e,c,f] = DensityHist(X,15,1);
[x,y] = ComputePDF(DST,MLE,[e(1),e(end)]);
hold on, plot(x,y,'r-'),hold off
title(['Data with Best Model: ',DST,'(',... 
    strrep(strtrim(sprintf('%0.2f ',MLE)),' ',','),')'])
disp(lin)

% power exponential - this just can't find it
disp(' '), disp(lin), disp('Fitting To Power Exponential Distribution')
DST = 'PXP';
true = [-3,2,0.5];
X = SimDistData(DST,1000,true);
truemax = ComputeLogLike(true,X,DST);
% run the SA
[MLE,maxloglike] = SimAnneal_MLE([-2,1.5,0.25],[-1,1,1],X,DST,[500000,500,0,1,1]);
disp(table2str({'mu';'sigma';'beta'},[true;MLE],{'%0.2f'},0,{'True';'MLE'}))
disp(sprintf('Maximized Log Likelihood at\n\tTrue = %0.3f\n\tMLE = %0.3f',truemax,maxloglike))
% plot best distribution
figure
[e,c,f] = DensityHist(X,15,1);
[x,y] = ComputePDF(DST,MLE,[e(1),e(end)]);
hold on, plot(x,y,'r-'),hold off
title(['Data with Best Model: ',DST,'(',... 
    strrep(strtrim(sprintf('%0.2f ',MLE)),' ',','),')'])
disp(lin)

tottim = etime(clock(),simstt);
disp(' ')
disp(sprintf('Modeling Completed in \n\t%1.4f Seconds\n\t%1.4f Minutes\n\t%1.4f Hours',tottim./[1,60,3600]));
disp(['Diary file saved:',dryname]),disp(repmat('@',1,75)),diary off