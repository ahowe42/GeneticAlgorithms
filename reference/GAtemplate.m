% usage: GAtemplate
% This is a general template used for creating and running the genetic
% algorithm for variable subsetting.  Find sections marked by ??? to
% identify where you need to insert / modify code for your specific use.
%
% Copyright Prof. Hamparsum Bozdogan & J. Andrew Howe
% All rights reserved, see LICENSE.TXT

clear, clc, close all
warning('off','MATLAB:dispatcher:InexactMatch');
tic, drvstt = clock; stt = sprintf('%4.0f%02.0f%02.0f_%02.0f%02.0f%02.0f',drvstt);

% find and store where I am located, prep output dir
mydir = dbstack; mydir = mydir(end);
mydir = which(mydir.file);
tmp = strfind(mydir,'\');
mydir = mydir([1:(tmp(end)-1)]);
if (exist([mydir,'\output\'],'dir') ~= 7)
    mkdir(mydir,'output');
end

% ???CODE FOR DATA PREPARATION SHOULD GO HERE???
data_path = [mydir,'\data\'];
data_file = 'BodyFatData.m';
data = dlmread([data_path,data_file]);
[n,p] = size(data);

% initialize model selection parameters
showtopsubs = 10;                % number of best subsets to show
popul_size = 30;                 % number of chromosomes in population
num_generns = 50;                % number of generations in GA
nochange_terminate = 30;         % number of generations with no improvement to go before termination
convgcrit = 1e-7;                % definition of "no improvement"
elitism = 1;                     % should best chromosome cross generations unchanged?
sel_type = 2;                    % 1 = sorted, 2 = roulette
prob_xover = 0.75;               % probability of crossover
xover_type = 1;                  % 1 = single, 2 = dual, 3 = uniform
prob_mutate = 0.10;              % probability of mutation - chromosomes & datapoints
prob_engineer = 0.00;            % probability of GA engineering
objec_func = 'MVGaussLogLike';   % objective function
maxmin = 1;                      % 1 = maximize, -1 = minimize
pltflg = 0;                      % 1 = update progress plot on-the-fly, 0 = don't
plt3d = 0;                       % 1 = create 3d surface plot at end, 0 = don't
randstate = 0;                   % 0 = randomize, else = random state

% set randomizer
if randstate == 0; randstate = sum(drvstt*1000000); end;
rand('state',randstate);

save_prefix = [mydir,'\output\GASub_',objec_func,'_',stt];
diary([save_prefix,'.out']);

% Display parameters
disp(upper(['GA BEGAN'])), disp(repmat('#',1,50)), disp(save_prefix)
disp(['Started on: ',stt])
disp(sprintf('Random State: %10.0f',randstate))
disp(sprintf('Maximum # Generations: %0.0f\nMininum # of Generations: %0.0f',num_generns,nochange_terminate))
disp(['Convergence Criteria: ',num2str(convgcrit)]);
if rem(popul_size,2) == 1
    popul_size = popul_size + 1;
    disp(sprintf('Population Size: %0.0f',popul_size))
    disp('!!Population Size Increased By 1 to be Even!!')
else
    disp(sprintf('Population Size: %0.0f',popul_size))
end
disp(sprintf('Crossover Rate: %0.2f\nMutation Rate: %0.2f',prob_xover,prob_mutate))
if xover_type == 1;
    disp('Crossover Method: SINGLE-POINT')
elseif xover_type == 2
    disp('Crossover Method: DUAL-POINT')
else
    disp('Crossover Method: UNIFORM')
end
if sel_type == 1; disp('Mating Method: SORTED'); else; disp('Mating Method: ROULETTE'); end;
if elitism == 1
    disp('Elitism is: ON');
    if prob_engineer > 0
        disp('!!With Elitism ON, the probability of GA engineering has been set to 0.00!!')
        prob_engineer == 0;
    else
        disp(sprintf('GA Engineering Rate: %0.2f',prob_engineer))
    end
    if plt3d == 1
        disp('!!With Elitism ON, the 3d Surface plot has been turned OFF!!')
        plt3d = 0;
    end
else
    disp('Elitism is: OFF');    
    disp(sprintf('GA Engineering Rate: %0.2f',prob_engineer))
end
if maxmin == 1; disp('Objective: MAXIMIZE'); else; disp('Objective: MINIMIZE'); end;
disp(['Objective Function: ',objec_func])
disp(['Data Modeled: ',data_path,data_file])
disp(repmat('#',1,50)),disp(' ')

% initialize population - start out with about half 1s
tmp = popul_size - min(p,5);
population = zeros(tmp,p);
population(unidrnd(tmp*p,1,ceil(tmp*p/2))) = 1;
all_zero = find((sum(population == zeros(tmp,p),2) == p).*[1:tmp]');
population(all_zero,unidrnd(p,length(all_zero),1)) = 1;
for cnt = 1:min(p,5)    % insert some chromosomes with just the first few subsets
    population = [population;[ones(1,cnt),zeros(1,p - cnt)]];
end                 % chromosomes loop
population = logical(population);
if maxmin == 1
    plttit = ['GA Progress: Maximize ',objec_func];
else
    plttit = ['GA Progress: Minimize ',objec_func];    
end

% initialize more "things"
genscores = zeros(num_generns,2); termcount = 0; best_score = maxmin*-1*Inf;
genbest = zeros(num_generns,p); pltxyy = []; allscores = [];
prevgenbestchrom = zeros(1,p); prevgenbestscore = maxmin*Inf;
allchroms = []; allscores = []; % JAH added allscores 20090803
best_chrom = []; % JAH 20090804

% Begin Genetic Algorithm
fhga = figure;
for gencnt = 1:num_generns

    % COMPUTE OBJECTIVE FUNCTION VALUES
    pop_fitness = ones(popul_size,1)*Inf;
    for popcnt = 1:popul_size
        %???modify the feval call as required???
        % JAH check if this chromosome already evaluated 20090803
        if gencnt > 1
            preveval = find(sum(allchroms == repmat(population(popcnt,:),size(allchroms,1),1),2) == p);
            if isempty(preveval)
                pop_fitness(popcnt) = feval(objec_func,data(:,population(popcnt,:)));
            else
                pop_fitness(popcnt) = allscores(preveval(1));
            end
        else
            pop_fitness(popcnt) = feval(objec_func,data(:,population(popcnt,:)));
        end
    end             % chromosomes loop
    % If maxmin is (+), this will not change the scores, so the true max
    % will be taken.  If maxmin is (-), the signs will all be changed,
    % and since max(X) = min(-X), taking the maximum will really be
    % taking the minimum.    
    [optval,optind]= max(maxmin*pop_fitness);
    % same some stuff before moving along
    genscores(gencnt,:) = [maxmin*optval,mean(pop_fitness(not(isnan(pop_fitness))))];
    genbest(gencnt,:) = population(optind,:);
    if plt3d == 1; allscores = [allscores;pop_fitness']; end;
    
    % UPDATE DATA FOR PLOT
    pltxyy = [pltxyy;[gencnt,genscores(gencnt,1),genscores(gencnt,2)]];
    if (pltflg == 1)
        figure(fhga)
        [AX,H1,H2] = plotyy(pltxyy(:,1),pltxyy(:,2),pltxyy(:,1),pltxyy(:,3)); xlabel('Generation');
        title(plttit,'interpreter','none');
        set(get(AX(1),'Ylabel'),'String','Optimum Value (o)','color','b');
        set(H1,'color','b','marker','o'); set(AX(2),'ycolor','b');
        set(get(AX(2),'Ylabel'),'String','Average Value (*)','color','r');
        set(AX(2),'ycolor','r'); set(H2,'color','r','marker','*');
        drawnow
    end

    if (gencnt == num_generns); break; end;       % don't bother with the next generation

    % EARLY TERMINATION ALLOWED?
    if maxmin*genscores(gencnt,1) > maxmin*best_score
        best_score = genscores(gencnt,1);
        best_chrom = genbest(gencnt,:);
        termcount = 1;
    elseif (maxmin*genscores(gencnt,1) < maxmin*best_score) && (elitism == 0)
        % if elitism is off, we can still do early termination with this
        termcount = termcount + 1;
    elseif maxmin*genscores(gencnt,1) - maxmin*best_score < convgcrit    % "no" improvement
        termcount = termcount + 1;
    end
    if termcount >= nochange_terminate
        disp(['Early Termination On Generation ',num2str(gencnt),' of ',num2str(num_generns)]);
        genscores = genscores([1:gencnt],:);    % get rid of unused generation spaces
        genbest = genbest([1:gencnt],:);
        break;
    end    

    % SELECTION OF NEXT GENERATION
    parents = GAselect(pop_fitness, -maxmin, sel_type);
    % CROSSOVER OPERATION ON NEW POPULATION
    new_pop = GAcrossover(population, parents, prob_xover, xover_type);
    % CHROMOSOME MUTATION
    new_pop = GAmutation(new_pop, prob_mutate);
    popul_size = size(new_pop,1);   % readjust in case it was odd and GAselect added 1
    % GA ENGINEERING
    if prob_engineer > 0
        % I check for prob_engineer > 0 because I can turn this off by
        % setting it to 0. Below we use genscores and gencnt, because
        % best_score and best_chrom won't hold the current best if the
        % current best solution is worse than the overall best, and elitism
        % is off.
        if (gencnt > 1) && (maxmin*prevgenbestscore > maxmin*genscores(gencnt,1))
            % only call GAengineering if the previous generation best is better
            new_pop = GAengineering(prevgenbestchrom,genbest(gencnt,:),new_pop,prob_engineer);
        end
        prevgenbestchrom = genbest(gencnt,:);
        prevgenbestscore = genscores(gencnt,1);
    end    
    
    % FIX ALL-ZERO CHROMOSOMES
    all_zero = find((sum(new_pop == zeros(popul_size,p),2) == p).*[1:popul_size]');
    new_pop(all_zero,unidrnd(p,length(all_zero),1)) = 1;

    % SAVE ALL UNIQUE CHROMOSOMES & SCORES JAH 20090803
    [tab,tmp] = unique(population,'rows');
    allchroms = [allchroms;tab];
    allscores = [allscores;pop_fitness(tmp)];
    
    % CONVEY BEST INDIVIDUAL INTO NEW POPULATION
    if elitism == 1
        population = [new_pop;best_chrom];
        popul_size = size(population,1);
    else
        population = new_pop;
    end
    population = (population == 1);
    if rem(gencnt,2) == 1
        disp(sprintf('Generation %0.0f of %0.0f: Best Score = %0.4f, Early Termination = %0.0f',gencnt,num_generns,best_score,termcount));
    end
end                 % generations loop
disp(sprintf('Generation %0.0f of %0.0f: Best Score = %0.4f, Early Termination = %0.0f',gencnt,num_generns,best_score,termcount));
% End Genetic Algorithm

figure(fhga)
[AX,H1,H2] = plotyy(pltxyy(:,1),pltxyy(:,2),pltxyy(:,1),pltxyy(:,3)); xlabel('Generation');
title(plttit,'interpreter','none');
set(get(AX(1),'Ylabel'),'String','Optimum Value (o)','color','b');
set(H1,'color','b','marker','o'); set(AX(2),'ycolor','b');
set(get(AX(2),'Ylabel'),'String','Average Value (*)','color','r');
set(AX(2),'ycolor','r'); set(H2,'color','r','marker','*');
drawnow, hgsave([save_prefix,'_GA.fig']);   % save the figure

if plt3d == 1
    fhIC = figure;
    surf(allscores),title(['3-dimensional Surface Plot of ',objec_func,' scores.'],'interpreter','none');
    ylabel('Generation'),xlabel('Population')
    drawnow, hgsave([save_prefix,'_3d.fig']);   % save the figure
end

% SUMMARY: GA_BEST = [scores,frequencies,solutions]
% get the unique sorted solutions and scores from all generations
gen_results = [genscores(:,1),zeros(gencnt,1),genbest]; % combine optimum score and chromosome
unique_gens = unique(gen_results,'rows');               % get just the unique results
numuni = size(unique_gens,1);                           % number unique results
unique_gens(:,1) = maxmin*unique_gens(:,1);             % set up for proper optimization
GA_BEST = sortrows(unique_gens,-1);                     % sort in descending order
GA_BEST(:,1) = maxmin*GA_BEST(:,1);                     % get back the raw scores
top_best = min(showtopsubs,numuni); chromloc = [3:(p + 2)];

% compute frequencies & prepare row headers
vars = [1:p]; rwhds = cell(top_best,1);
for pcnt = 1:numuni
    % compute frequency
    GA_BEST(pcnt,2) = sum(sum(repmat(GA_BEST(pcnt,chromloc),gencnt,1) == gen_results(:,chromloc),2) == p);
    % prepare row header
    if pcnt <= top_best
        vrs = sprintf('%d,',find(vars.*logicl(GA_BEST(pcnt,chromloc))));
        rwhds{pcnt,1} = ['{',vrs([1:(end-1)]),'}'];
    end
end                 % results loop

% display the best chromosomes and scores
tab = table2str({'Size',objec_func,'Frequency'},[sum(GA_BEST([1:top_best],[3:end]),2),...
    GA_BEST([1:top_best],[1,2])],{'%0.0f','%0.3f','%0.0f'},0,rwhds);
lin = repmat('=',1,60);
disp(' '), disp(lin), disp('GA Complete')
disp(sprintf('\tTotal Solutions Evaluated - %0.0f\n\tUnique Solutions Evaluated - %0.0f\n\tTotal Solutions Possible - %0.0f',size(allchroms,1),size(unique(allchroms,'rows'),1),2^p-1))
disp(tab), disp(lin)

% get rid of unneeded variables
clear all_zero AX H1 H2 best_chrom best_score fhga gencnt tab termcount
clear lin opt* new_pop numuni pcnt wghts vars population unique_gens vrs
clear pltxyy pop_fitness popcnt rwhds fhIC chromloc parents tmp prev* gen_results

tottim = etime(clock,drvstt);
disp(sprintf('GA Completed in \n\t%1.4f Seconds\n\t%1.4f Minutes\n\t%1.4f Hours',tottim./[1,60,3600]));
diary off, save([save_prefix,'.mat']);
warning('on','MATLAB:dispatcher:InexactMatch');
