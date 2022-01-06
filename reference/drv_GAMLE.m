% usage: drv_GAMLE
% Use the genetic algorithm to estimate parameters for various
% distributions using simulated or real data, then use information critiera
% to pick the best model. This uses functions:
% SimDistData, ComputeLogLike, EstParamRanges, ComputePDF.
%
% JAH 20071028
% Copyright Prof. Hamparsum Bozdogan & J. Andrew Howe
% All rights reserved, see LICENSE.TXT

clear; clc; close all;

% get the starting date/timestamp
lin = repmat('-',1,75);
stt = sprintf('%4.0f%02.0f%02.0f_%02.0f%02.0f%02.0f',clock());

% find and store where I am located
mydir = dbstack; mydir = mydir(end);
mydir = which(mydir.file);
tmp = strfind(mydir,'\');
mydir = mydir([1:(tmp(end)-1)]);

% save output
dryname = ['GAMLE_',stt,'.out']; diary([mydir,'\',dryname]);
disp(repmat('@',1,75)),disp(['drv_GAMLE began on ',stt]);

% fitting distributions stuff
[diststofit,jnka] = ComputeLogLike();
diststofitnumparms = jnka(:,1);
diststofitsupp = jnka(:,2);
numdists = size(diststofit,1);
% just fit some distributions?
fitsubset = questdlg('Do you want to fit just a subset of distributions?','drv_GAMLE','Yes','No','No');
if isequal(fitsubset ,'Yes')
    disp('Fitting Distributions:')
    disp([cellstr(num2str([1:numdists]')),diststofit])
    actualfit = input('Enter a Matlab vector ([1,2,3,...]) with the number(s) of the distribution(s) you want to fit.\n');
    disp(lin)
    disp('User requested to only fit:')
    disp(diststofit(actualfit,1))
    disp(lin)    
else
    actualfit = [1:numdists];
end

% get simulation and GA parameters
prompt = {'Population Size','Number Generations','Number Generations for Early Termination',...
    'Convergence Criteria','Elitism','Generation Seeding type','Crossover Rate','Crossover Type',...
    'Mutation Rate','GA Engineering Rate','Progress Plot Flag','3d Surface Plot Flag',...
    'Screen Output Flag','Randomization','Information Criteria (AIC,SBC)','Encoding Rate'};
defaults = {'50','50','10','1e-5','1','2','0.75','1','0.10','0.00','-1','-1','-1','0','AIC','32'};
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
ICuse = answer{15};
encrate = str2num(answer{16});
% this chunk of code from old version using genetic.m is at the bottom

% simulated data or real data?
data_type = 1+isequal('Real',questdlg('Will you use simulated or real data?',...
    'drv_GAMLE','Simulated','Real','Simulated'));    % 1=sim, 2 = real

if data_type == 1    % fit to simulated data
    prompt = {'Number Monte Carlo Trials','Sample Size'};
    answer = inputdlg(prompt,'Simulation Parameters',1,{'50','500'},'on');
    mcnum = str2num(answer{1});
    sampsize = str2num(answer{2});
    
    % get simulating distribution & parameters
    [simdists,jnka] = SimDistData();
    numparms = jnka(:,1); simsupport = jnka(:,2);
    disp('Simulating Distributions:')
    disp([cellstr(num2str([1:length(simdists)]')),simdists])
    simcnt = input('Which distribution (it''s number)? ');
    simdist = simdists{simcnt,1};               % distribution
    simprms = zeros(1,numparms(simcnt));        % parameter names
    simsupport = simsupport(simcnt);            % support for X
    disp(['Simulation Parameters (',simdists{simcnt,2},')'])
    for cnt = 1:numparms(simcnt)
        simprms(cnt) = input(['What is the value for parameter: ',num2str(cnt),'? ']);
    end
    
    % restrict distributions to fit?
    if sum(diststofitsupp <= simsupport) < numdists
        X = SimDistData(simdist,1000,simprms,0);
        if min(X) < 0
            % limit this to a subset of dists if user selected one
            fitsubset = (diststofitsupp <= simsupport);
            fitsubset = fitsubset(actualfit);
            actualfit = actualfit(fitsubset);
            disp(lin)
            disp('Dist. could include negative values, so limited X support - only fit:')
            disp(diststofit(actualfit,1))
            disp(lin)
        else
            % no restriction required
        end
    end    
else                % fit to real data
    mcnum = 1;
    % get the data file - this will just read the 1st column
    [data_file,pn] = uigetfile('*.m', 'Load Real Data File',mydir);
    fullfile = [pn,data_file];
    X = dlmread(fullfile); X = X(:,1);
    sampsize = size(X,1);
    
    % restrict distributions to fit?
    if min(X) < 0
        fitsubset = (diststofitsupp <= repmat(0,size(diststofitsupp)));
        fitsubset = fitsubset(actualfit);
        actualfit = actualfit(fitsubset);
        disp(lin)
        disp('Data includes negative values, so limited X support - only fit:')
        disp(diststofit(actualfit,1))
        disp(lin)
    else
        % no restriction required
    end
end

if isempty(actualfit)
    disp('!!!Fit Limitations Narrowed the List to None - Terminating!!!');
    diary off
    delete([mydir,'\',dryname]);
    return;    
end

% initialize
ICs = ones(mcnum,numdists)*Inf;  % row = simulation, column = distribution
estparms = cell(mcnum,numdists);
simstt = clock();
%global dst;
for mciter = 1:mcnum
    if data_type == 1
        disp(sprintf('Performing Iteration %0.0f of %0.0f',mciter,mcnum))
        % perform simulation
        X = SimDistData(simdist,sampsize,simprms,0);
    end
    disp(lin)    
    
    % do the work
    for distcnt = actualfit
        dst = diststofit{distcnt,1};
        p = diststofitnumparms(distcnt);
        bits = repmat(encrate,1,p);
        % upper and lower bounds for parameter(s)
        [vlb,vub] = EstParamRanges(dst,X);
        %[estparms{mciter,distcnt},jnka,jnkb,maxloglike] = genetic('ComputeLogLike',[],options,vlb,vub,bits,X,dst);
        [estparms{mciter,distcnt},maxloglike] = GArealoptim(options,bits,vlb,vub,dst,'ComputeLogLike',X,dst);
        if isequal(ICuse, 'AIC')
            ICs(mciter,distcnt) = -2*maxloglike + 2*p;
        elseif isequal(ICuse, 'SBC')
            ICs(mciter,distcnt) = -2*maxloglike + log(sampsize)*p;
        end
        % display
        disp([dst,'(',diststofit{distcnt,2},'): [',...
            strtrim(sprintf('%0.2f ',estparms{mciter,distcnt})),sprintf([']: ',ICuse,' = %0.4f'],...
            ICs(mciter,distcnt))])
    end             % distributions to fit loop
    [v,i] = min(ICs(mciter,:));
    disp(['BEST MODEL: ',diststofit{i}]),disp(lin)
end                 % simulations loop

% get the minimum values and related items
[minvals,mininds] = min(ICs,[],2); % min per iteration
% frequency with which each dist minimized IC
freqsel = 100*sum(ICs == repmat(minvals,1,numdists),1)/mcnum;
% get the overall best model
[absminv,absmini] = min(minvals);
bstdst = diststofit{mininds(absmini),1};        % distribution
bstparms = estparms{absmini,mininds(absmini)};  % parms
tottim = etime(clock(),simstt);

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
disp('-Data Parameters-')
if data_type == 1           % simulated
    disp(sprintf('Number Simulations: %0.0f',mcnum))
    disp([sprintf('Simulate %0.0f samples from ',sampsize),simdist,...
        '(',simdists{simcnt,2},'): [',strtrim(sprintf('%0.2f ',simprms)),']'])
else                        % real data
    disp(['Fit data in ',fullfile])
    disp(sprintf('Number Observations: %0.0f ',sampsize))
end
disp(repmat('=',1,75))

if data_type == 1           % simulated
    disp('Simulation Results'),disp([ICuse,' Values'])
    tab = table2str(diststofit(:,1),[ICs;freqsel],{'%0.2f'},0,[cellstr(num2str([1:mcnum]'));cellstr('%')]);
    tab = [tab([1:(end-2)],:);tab([2,3],:);tab([(end-1):end],:)]; disp(tab)
    disp(['The % row indicates the % of simulations in which ',ICuse,' was minimized with that distribution.'])
else
    tab = table2str(diststofit(:,1),ICs,{'%0.2f'},0);
    disp([ICuse,' Values']),disp(tab)
end
disp('Best Overall Model:')
disp([bstdst,'(',diststofit{mininds(absmini),2},'): [',...
            strtrim(sprintf('%0.2f ',bstparms)),sprintf([']: ',ICuse,' = %0.4f'],absminv)])
disp(sprintf('Modeling Completed in \n\t%1.4f Seconds\n\t%1.4f Minutes\n\t%1.4f Hours',tottim./[1,60,3600]));

% plot data
[e,c,f,bh] = DensityHist(X,15,1);
% overlay best fit distribution
[x,y] = ComputePDF(bstdst,bstparms,[e(1),e(end)]);
set(bh,'FaceColor','none')
hold on, plot(x,y,'r-'),hold off
% overlay true simulating distribution
if data_type == 1
    [xt,yt] = ComputePDF(simdist,simprms,[e(1),e(end)]);    
    hold on
    plot(xt,yt,'b:')
    hold off
    legend('Data','Best','True')
    title({['Data with Best Model: ',bstdst,'(',... 
        strrep(strtrim(sprintf('%0.2f ',bstparms)),' ',','),')'];...
        ['True Model: ',simdist,'(',strrep(strtrim(sprintf('%0.2f ',simprms)),' ',','),')']})
else
    title(['Data with Best Model: ',bstdst,'(',... 
        strrep(strtrim(sprintf('%0.2f ',bstparms)),' ',','),')'])    
end

disp(['Diary file saved:',dryname]),disp(repmat('@',1,75)),diary off

% get simulation and GA parameters
%prompt = {'Encoding Rate','Number Generations','Population Size',...
%    'Mutation Rate','Crossover Rate','Display Parameter (0,1,2)',...
%    'Information Criteria (AIC,SBC)'};
%answer = inputdlg(prompt,'GA Parameters',1,{'32','50','50','0.01','0.8','0','AIC'},'on');
%encrate = str2num(answer{1});
%options = zeros(1,14);
%options([2,3,4]) = [1e-3,1e-4,1e-6];
%options(14) = str2num(answer{2});   % number generations
%options(11) = str2num(answer{3});   % population size
%options(13) = str2num(answer{4});   % mutation rate
%options(12) = str2num(answer{5});   % crossover rate
%options(1) = str2num(answer{6});  % 0 = no display, 1 = text display, 2 = text and plot display
%ICuse = answer{7};