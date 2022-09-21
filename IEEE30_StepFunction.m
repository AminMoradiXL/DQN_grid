function [NextObs,Reward,IsDone,LoggedSignals] = IEEE30_StepFunction(Action,LoggedSignals)

% This function applies the given action to the environment and evaluates
% the system dynamics for one simulation step.

a = Action;
obj=5;
d=[15 16];

% Unpack the state vector from the logged signals.
state = LoggedSignals.State;

flag=nnz(~state)>=obj;

[next_state, instant_out, genloss]=attack_eff_IEEE30(state, a, d);


% Perform Euler integration.
LoggedSignals.State = next_state;

% Transform state to observation.
NextObs = LoggedSignals.State;

% Check terminal condition.

Down=nnz(~next_state);

IsDone = Down>=obj;

IsFinal=IsDone*(~flag);

% Get reward

Reward=RewardFunction(instant_out,obj,IsFinal);

end