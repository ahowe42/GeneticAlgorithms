% usage: Demo_GAMultivariateMLE
% Demonstrate using the genetic algorithm to estimate parameters 
% for a multivariate normal distribution.
%
% JAH 20100423
% Copyright Prof. Hamparsum Bozdogan & J. Andrew Howe
% All rights reserved, see LICENSE.TXT

clc; close all;

% get the starting date/timestamp
lin = repmat('-',1,75);
stt = sprintf('%4.0f%02.0f%02.0f_%02.0f%02.0f%02.0f',clock());

% find and store where I am located
mydir = dbstack; mydir = mydir(end);
mydir = which(mydir.file);
tmp = strfind(mydir,'\');
mydir = mydir([1:(tmp(end)-1)]);

% save output
dryname = ['GA_MultMLE_',stt,'.out']; diary([mydir,'\',dryname]);
disp(repmat('@',1,75)),disp(['Demo_GAMultivariateMLE began on ',stt]);

% get GA parameters
prompt = {'Population Size','Number Generations','Number Generations for Early Termination',...
    'Convergence Criteria','Elitism','Generation Seeding type','Crossover Rate','Crossover Type',...
    'Mutation Rate','GA Engineering Rate','Progress Plot Flag','3d Surface Plot Flag',...
    'Screen Output Flag','Randomization','Encoding Rate','Use Existing Data Parms'};
defaults = {'50','50','10','1e-5','1','2','0.75','1','0.10','0.00','-1','-1','-1','0','32','0'};
answer = inputdlg(prompt,'GA Parameters (from GArealoptim)',1,defaults,'on');
options = zeros(1,15);
options(1) = str2num(answer{1});    options(2) = str2num(answer{2});
options(3) = str2num(answer{3});    options(4) = str2num(answer{4});
options(5) = str2num(answer{5});    options(6) = str2num(answer{6});
options(7) = str2num(answer{7});    options(8) = str2num(answer{8});
options(9) = str2num(answer{9});    options(10) = str2num(answer{10});
options(11) = 1;                    options(12) = str2num(answer{11});
options(13) = str2num(answer{12});  options(14) = str2num(answer{13});
options(15) = str2num(answer{14});
encrate = str2num(answer{15});
use_these = str2num(answer{16});

% randomize
if (options(14) == 0)
	randstate = sum(drvstt*1000000);
else
	randstate = options(14);
end
randn('state',randstate);

% simulate the data
% check if user has already defined variables
if use_these
    % ensure n, p, true_mu, and true_sigma exist
    if exist('n','var')*exist('p','var')*exist('true_mu','var')*...
        exist('true_sigma','var') == 0
        disp('Please Ensure n, p, true_mu, and true_sigma Exist')
        return;
    end
else
    % TRIVARIATE
    % n = 100; p = 3;
    % true_mu = [-3,-3,5];
    % true_sigma = [1,0,0.5;0,2,-0.5;0.5,-0.5,0.5];
    % BIVARIATE
    n = 100; p = 2;
    true_mu = [-3,5];
    true_sigma = [1,-0.75;-0.75,2];    
end
data = genrndmvnorm(n,p,true_mu,true_sigma);
true_mll = sum(log(mvnpdf(data,true_mu,true_sigma)));

% intitialize
simstt = clock();
numparms = p*(1 + (p+1)/2);         % number parameters to estimate
bits = repmat(encrate,1,numparms);	% encoding rate

% set up ranges for all variables
xbar = mean(data); S = cov(data,1);
mnci = repmat(xbar,2,1) + 3*repmat([-1;1],1,p).*repmat((diag(S)'/n).^(0.5),2,1);
% encode S
Sflat = SigmaEncDec(S,p);
% extract variances and covariances separately
varis = zeros(p*(p+1)/2,1);
varis(cumsum(ones(1,p) + [0,1:(p-1)])) = 1;
varis = (varis == 1);
% make the ranges for variances
sigmalow = varis*0.001;
sigmaupp = varis.*Sflat*2.25;
% make ranges for covariances
sigmalow(not(varis)) = -abs(Sflat(not(varis))*2.25^2);
sigmaupp(not(varis)) = abs(Sflat(not(varis))*2.25^2);
% make the bounds
vlb = [min(mnci),sigmalow'];
vub = [max(mnci),sigmaupp'];

% run the GA
[estparms,maxloglike] = GArealoptim(options,bits,vlb,vub,[],'MVN_Like_SigmaEncoded',data);
est_mu = estparms([1:p]);
est_sigma = SigmaEncDec(estparms([(p+1):end]),p);
tottim = etime(clock(),simstt);

% diplay stuff now
% display parameters and results
disp(repmat('=',1,75))
disp('-GA Parameters-')
disp(sprintf('Population Size: %0.0f',options(1)))
disp(sprintf('Maximum # Generations: %0.0f\nMinimum # of Generations: %0.0f',options([2,3])))
disp(['Convergence Criteria: ',num2str(options(4))]);
if options(5) == 1 disp('Elitism is: ON'); else; disp('Elitism is: OFF');
end
if options(6) == 1; disp('Mating Method: SORTED'); else; disp('Mating Method: ROULETTE'); end;
disp(sprintf('Mutation Rate: %0.2f\nCrossover Rate: %0.2f',options([9,7])))
if options(8) == 1;
    disp('Crossover Method: SINGLE-POINT')
elseif options(8) == 2
    disp('Crossover Method: DUAL-POINT')
else
    disp('Crossover Method: UNIFORM')
end    
disp(sprintf('GA Engineering Rate: %0.2f',options(10)))
disp(sprintf('Randomizer: %10.0f',options(15)))
disp(sprintf('Encoding Rate: %0.0f',encrate))
disp(repmat('=',1,75)),disp(' ')

% display the true parameters
disp(lin)
disp('TRUE DATA GENERATING PARAMETERS')
disp(sprintf('%d observations drawn from Multivariate Normal with Parameters',n))
disp(dispMeanCovar(true_mu,true_sigma,'%0.2f'))
disp(sprintf('Likelihood = %0.2f',true_mll))
disp(lin)

% now display the estimated paramters
disp('ESTIMATED DATA GENERATING PARAMETERS')
disp(dispMeanCovar(est_mu,est_sigma,'%0.2f'))
disp(sprintf('Likelihood = %0.2f',maxloglike))
disp(lin)

% if bivariate, show density contours for true & estimated pdfs
if (p == 2)
	% ranges for true distribution
	true_x1 = [true_mu(1)-3*sqrt(true_sigma(1,1)),true_mu(1)+3*sqrt(true_sigma(1,1))];
	true_x2 = [true_mu(2)-3*sqrt(true_sigma(2,2)),true_mu(2)+3*sqrt(true_sigma(2,2))];
	% ranges for est distribution
	est_x1 = [est_mu(1)-3*sqrt(est_sigma(1,1)),est_mu(1)+3*sqrt(est_sigma(1,1))];
	est_x2 = [est_mu(2)-3*sqrt(est_sigma(2,2)),est_mu(2)+3*sqrt(est_sigma(2,2))];
	% overall ranges
	x1 = linspace(min(true_x1(1),est_x1(1)),max(true_x1(2),est_x1(2)));
	x2 = linspace(min(true_x2(1),est_x2(1)),max(true_x2(2),est_x2(2)));
	% create the eval values
	[X1,X2] = meshgrid(x1, x2); mesh_size = length(x1); 
	X1flat = reshape(X1,mesh_size^2,1);
	X2flat = reshape(X2,mesh_size^2,1);
	X = [X1flat,X2flat];
	% compute densities for each distribution
	true_f = mvnpdf(X,true_mu,true_sigma);
	true_f = reshape(true_f,mesh_size,mesh_size);
	est_f = mvnpdf(X,est_mu,est_sigma);    
	est_f = reshape(est_f,mesh_size,mesh_size);

	% finally plot the contours
	limts = [min(x1),max(x1),min(x2),max(x2)];
	[c,h] = contour(X1,X2,true_f);
	set(h,'LineColor','k','LineStyle',':','LineWidth',2);
	hold on, contour(X1,X2,est_f), hold off
	axis(limts),legend('True','Estimated')
end

% finish up
disp(sprintf('Modeling Completed in \n\t%1.4f Seconds\n\t%1.4f Minutes\n\t%1.4f Hours',tottim./[1,60,3600]));
disp(['Diary file saved:',dryname]),disp(repmat('@',1,75)),diary off