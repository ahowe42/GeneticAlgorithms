function [MLE,maxloglike] = SimAnneal_MLE(prevest,lim,data,dist,saparms)
% [MLE, maximized likelihood] = SimAnneal_MLE(initial estimates, ...
% parameter limits, data, distribution, parameters)
%  This implements an adaptive simulated annealing algorithm to find the
%  maximum likelihood estimate(s) of the parameter(s) of certain
%  distributional models.
%
%  Where
%  initial estimates  --- (1 x p) vector of initial estimates for MLEs.
%  parameter limits --- (1 x p) vector describing limits on parameters: -1 = none,
%     0 = nonnegative, 1 = strictly positive.
%  data --- (n x 1) vector of data
%  distribution --- Character code of distributions in ComputeLogLike
%  parameters --- (1 x 5) vector holding:
%     1) beginning temperature (very problem dependent)
%     2) max number of iterations
%     3) randomization code: -1 = none, 0 = auto, else = state
%     4) screen output: 0 = no, 1 = yes
%     5) progress plot: 0 = no, 1 = yes
%  MLE --- (1 x p) vector of found maximum likelihood estimators.
%  maximized likelihood --- Scalar value of likelihood evaluated at MLE.
%
%  Example: [MLE,mll] = SimAnneal_MLE([1,1],[0,-1],SimDistData('NRM',1000,[5,2]),'NRM',[50000,100,0]);
%
%  JAH 20071205
%  Copyright Prof. Hamparsum Bozdogan & J. Andrew Howe
%  All rights reserved, see LICENSE.TXT

if (nargin ~= 5)
    % wrong number of arguments
    fprintf('SimAnneal_MLE: INVALID USAGE-Please read the following instructions!\n'), help SimAnneal_MLE, return
end

% extract SA parameters
tempmax = saparms(1);   % beginning temperature
evalmax = saparms(2);   % max number of iterations
rndstat = saparms(3);   % randomize: -1 = none, 0 = auto, else = state
scrnout = saparms(4);   % screen output: 0 = none, 1 = yes
prtplot = saparms(5);   % do plot: 0 = no, 1 = yes

% randomize
if rndstat == 0;
    rndstat = sum(clock*1000000);
    rand('state',rndstat); randn('state',rndstat);
elseif rndstat == -1;
    % no randomization
else
    rand('state',rndstat); randn('state',rndstat);
end

numparms = length(prevest);
prevenrg = -ComputeLogLike(prevest,data,dist);
statesocc = prevest; enrgsocc = prevenrg;
evalcnt = 1;
if prtplot == 1
    figure;
end
while (evalcnt <= evalmax)
    % compute current temperature
    currtemp = tempmax*exp(-(evalcnt)^(1/numparms));    % using adaptive SA with c = 1
    % get a "neighbor" to the current state, and evaluate energy
    neib = genrndmvnorm(1,numparms,prevest,eye(numparms)*2);             % gaussian moves
    % ensure param limits obeyed
    for pcnt = 1:numparms
        if lim(pcnt) ~= -1; neib(pcnt) = abs(neib(pcnt)); end;
        if (lim(pcnt) == 1) && (neib(pcnt) == 0)
            neib(pcnt) = prevest(pcnt);
        end
    end             % parameters loop
    currest = neib;
    currenrg = -ComputeLogLike(currest,data,dist);
    if scrnout == 1
        disp(sprintf('Iteration %0.0f - Prev: %0.3f, Curr: %0.3f',evalcnt,prevenrg,currenrg))   
        %[prevest;currest]
    end
    % should we move there?
    if currenrg < prevenrg
        % improvement
        prevest = currest; prevenrg = currenrg;
        statesocc = [statesocc;prevest];
        enrgsocc = [enrgsocc;prevenrg];
    elseif exp(-(currenrg - prevenrg)/currtemp) > rand
        % no improvement, but let's go there anyway
        if scrnout == 1
            disp(['  Made a Bad move! - prob was: ',sprintf('%0.5f (%0.0f)',exp(-(currenrg - prevenrg)/currtemp),evalcnt)])
        end
        prevest = currest; prevenrg = currenrg;
        statesocc = [statesocc;prevest];
        enrgsocc = [enrgsocc;prevenrg];
    else
        statesocc = [statesocc;prevest];
        enrgsocc = [enrgsocc;prevenrg];
    end
    evalcnt = evalcnt + 1;
    if prtplot == 1
        plot([1:evalcnt],enrgsocc,'b*-'),drawnow
    end
end                 % SA loop
if prtplot == 1
    title(['Progress of ASA for ',dist,' likelihood']),xlabel('Iteration')
    ylabel('l(\theta)')
end
MLE = prevest; maxloglike = -prevenrg;