function [best_chrom,best_score] = GArealoptim(parms,bits,LB,UB,titext,objec_func,P1,P2,P3,P4,P5,P6,P7,P8,P9,P10);
% [best solution, best score] = GArealoptim(parameters, number bits, lower bounds,...
% upper bounds, extra title line, objective function, up to 10 parameters P1-P10)
%  This implements the genetic algorithm for optimization of real valued
%  functions with p parameters.
%
%  Where
%  parameters --- (1x15) vector of GA parameters:
%     (1) population size
%     (2) number of generations
%     (3) number of generations with insufficient improvement before early termination
%     (4) convergence criteria
%     (5) elitisim: 1 = on, 2 = off
%     (6) generation seeding type: 1 = sorted, 2 = roulette
%     (7) probability of crossover
%     (8) crossover type: 1 = single, 2 = dual, 3 = uniform
%     (9) probability of mutation
%     (10) probability of GA engineering
%     (11) Optimization goal: 1 = maximize, -1 = minimize
%     (12) progress plot flag: -1 = no plot; 1 = update on-the-fly, 0 = at end
%     (13) 3d surface plot flag: -1 = no surface plot, 1 = create surface plot at end
%     (14) screen output flag: -1 = nothing, 0 = summary only, 1 = all
%     (15) randomization: 0 = randomize, else = random state
%  number bits --- (1xp) vector with number of bits used to encode real values
%  lower bounds --- (1xp) vector with lower bound of range for real values
%  upper bounds --- (1xp) vector with upper bound of range for real values
%  extra title --- String, extra to put on plot titles (just [] if not desired)
%  objective function --- String indicating name of function to optimize
%  parameters --- up to 10 additional parameters that will be passed to the
%     objective function can go here (first must be the value from the GA)
%  best solution --- Real value that optimized objective function
%  best score --- Scalar optimized value of objective function
%
%  See Also GAcrossover, GAselect, GAengineering, GAmutation, GA10to2, GA2to10
%
%  JAH 20071203
%  Copyright Prof. Hamparsum Bozdogan & J. Andrew Howe
%  All rights reserved, see LICENSE.TXT

Ll = length(LB); Ul = length(UB); Bl = length(bits);

if (nargin < 6) || (nargin > 16) || (Ll ~= Ul) || (Ll ~= Bl) || (Ul ~= Bl) || (length(parms) ~= 15)
    % wrong number arguments, sizes don't match, wrong number parms
    fprintf('GArealoptim: INVALID USAGE-Please read the following instructions!\n'), help GArealoptim, return
end

warning('off','MATLAB:dispatcher:InexactMatch');
drvstt = clock;
stt = sprintf('%4.0f%02.0f%02.0f_%02.0f%02.0f%02.0f',drvstt);

% initialize model selection parameters
popul_size = parms(1);          num_generns = parms(2);
nochange_terminate = parms(3);  convgcrit = parms(4);
elitism = parms(5);             sel_type = parms(6);
prob_xover = parms(7);          xover_type = parms(8);
prob_mutate = parms(9);         prob_engineer = parms(10);
maxmin = parms(11);             pltflg = parms(12);
plt3d = parms(13);              outtoscr = parms(14);
randstate = parms(15);

% get objective function string
evalstr = [objec_func,'(popreal'];
for pcnt = 7:nargin
  evalstr = [evalstr,',P',num2str(pcnt-6)];
end
evalstr = [evalstr,')'];

% set randomizer
if randstate == 0; randstate = sum(drvstt*1000000); end;
rand('state',randstate);

% Display parameters
if outtoscr ~= -1
    disp(upper(['GA BEGAN'])), disp(repmat('#',1,50))
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
        if prob_engineer > 0
            disp('!!With Elitism ON, the probability of GA engineering has been set to 0.00!!')
            prob_engineer == 0;
        else
            disp(sprintf('GA Engineering Rate: %0.2f',prob_engineer))
        end    
        disp('Elitism is: ON');
        if plt3d == 1
            disp('!!With Elitism ON, the 3d Surface plot has been turned OFF!!')
            plt3d = -1;
        end
    else
        disp(sprintf('GA Engineering Rate: %0.2f',prob_engineer))
        disp('Elitism is: OFF');
    end
    if maxmin == 1; disp('Objective: MAXIMIZE'); else; disp('Objective: MINIMIZE'); end;
    disp(['Objective Function: ',objec_func])
    disp(repmat('#',1,50)),disp(' ')
end

% initialize population - start out with about half 1s
p = sum(bits);
population = zeros(popul_size,p);
population(unidrnd(popul_size*p,1,ceil(popul_size*p/2))) = 1;
all_zero = find((sum(population == zeros(popul_size,p),2) == p).*[1:popul_size]');
population(all_zero,unidrnd(p,length(all_zero),1)) = 1;
population = logical(population);

% initialize more "things"
genscores = zeros(num_generns,2); termcount = 0; best_score = maxmin*-1*Inf;
genbest = zeros(num_generns,p); pltxyy = []; allscores = [];
prevgenbestchrom = zeros(1,p); prevgenbestscore = maxmin*Inf;
if (pltflg ~= -1) || (plt3d == 1)
    if maxmin == 1
        plttit = ['GA Progress: Maximize ',objec_func];
    else
        plttit = ['GA Progress: Minimize ',objec_func];
    end
    plt3tit = ['3-dimensional Surface Plot of ',objec_func,' scores.'];
    if not(isempty(titext))
        if (size(titext,2) > 5) || (size(titext,1) > 1)
            plttit = {plttit;titext}; plt3tit = {plt3tit;titext};
        else
            plttit = [plttit,':',titext]; plt3tit = [plt3tit,':',titext];
        end
    end
end

% Begin Genetic Algorithm
if pltflg ~= -1; fhga = figure; end;
for gencnt = 1:num_generns
    % COMPUTE OBJECTIVE FUNCTION VALUES
    pop_fitness = ones(popul_size,1)*Inf;
    for popcnt = 1:popul_size
        popreal = GA2to10(population(popcnt,:),bits,LB,UB);
        pop_fitness(popcnt) = eval(evalstr);
        if not(isreal(pop_fitness(popcnt)))
            [st,i] = dbstack; eval(['dbstop in ''GArealoptim.m'' at ',num2str(st(i).line+1)]);
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
        set(get(AX(1),'Ylabel'),'String','Maximum Value (o)','color','b');
        set(H1,'color','b','marker','o'); set(AX(2),'ycolor','b');
        set(get(AX(2),'Ylabel'),'String','Average Value (*)','color','r');
        set(AX(2),'ycolor','r'); set(H2,'color','r','marker','*');
        drawnow
    end

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
    
    if (gencnt == num_generns); break; end;       % don't bother with the next generation    
    
    if termcount >= nochange_terminate
        if outtoscr ~= -1
            disp(['Early Termination On Generation ',num2str(gencnt),' of ',num2str(num_generns)]);
        end
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
    
    % CONVEY BEST INDIVIDUAL INTO NEW POPULATION
    if elitism == 1
        population = [new_pop;best_chrom];
        popul_size = size(population,1);
    else
        population = new_pop;
    end
    population = (population == 1);
    if (rem(gencnt,2) == 1) & (outtoscr == 1)
        disp(sprintf('Generation %0.0f of %0.0f: Best Score = %0.4f, Early Termination = %0.0f',gencnt,num_generns,best_score,termcount));
    end
end                 % generations loop
if (outtoscr == 1)
    disp(sprintf('Generation %0.0f of %0.0f: Best Score = %0.4f, Early Termination = %0.0f',gencnt,num_generns,best_score,termcount));
end
% End Genetic Algorithm

if pltflg ~= -1
    figure(fhga)
    [AX,H1,H2] = plotyy(pltxyy(:,1),pltxyy(:,2),pltxyy(:,1),pltxyy(:,3)); xlabel('Generation');
    title(plttit,'interpreter','none');
    set(get(AX(1),'Ylabel'),'String','Maximum Value (o)','color','b');
    set(H1,'color','b','marker','o'); set(AX(2),'ycolor','b');
    set(get(AX(2),'Ylabel'),'String','Average Value (*)','color','r');
    set(AX(2),'ycolor','r'); set(H2,'color','r','marker','*');
    drawnow
end

if plt3d == 1
    fhIC = figure;
    surf(allscores),title(plt3tit,'interpreter','none');
    ylabel('Generation'),xlabel('Population')
    drawnow
end

% convert best_chrom to real from binary
best_chrom = GA2to10(best_chrom, bits, LB, UB);

% if no output of summary, there's no reason to do the summary
if outtoscr == -1
    warning('on','MATLAB:dispatcher:InexactMatch');
    return;
end

% SUMMARY: GA_BEST = [scores,frequencies,solutions]
showtopbest = 10;
% get the unique sorted solutions and scores from all generations
gen_results = [genscores(:,1),zeros(gencnt,1),genbest]; % combine optimum score and chromosome
unique_gens = unique(gen_results,'rows');               % get just the unique results
numuni = size(unique_gens,1);                           % number unique results
unique_gens(:,1) = maxmin*unique_gens(:,1);
GA_BEST = sortrows(unique_gens,-1);                     
GA_BEST(:,1) = maxmin*GA_BEST(:,1);                     
top_best = min(showtopbest,numuni); chromloc = [3:(p + 2)];

% compute frequencies & prepare row headers
rwhds = cell(top_best,1);
for pcnt = 1:top_best
    % compute frequency
    GA_BEST(pcnt,2) = sum(GA_BEST(pcnt,1) == genscores(:,1));
    % prepare row header
    vrs = sprintf('%0.3f,',GA2to10(GA_BEST(pcnt,chromloc),bits,LB,UB));
    rwhds{pcnt,1} = ['[',vrs([1:(end-1)]),']'];
end                 % results loop

% display the best chromosomes and scores
tab = table2str({objec_func,'Frequency'},GA_BEST([1:top_best],[1,2]),{'%0.3f','%0.0f'},0,rwhds);
lin = repmat('=',1,60); disp(' '), disp(lin), disp('GA Complete')
disp(tab), disp(lin)

ga_toc = etime(clock,drvstt);
disp(sprintf('GA Completed in \n\t%1.4f Seconds\n\t%1.4f Minutes\n\t%1.4f Hours',ga_toc./[1,60,3600]));
warning('on','MATLAB:dispatcher:InexactMatch');