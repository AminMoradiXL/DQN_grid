function [NextObs,Reward,IsDone,LoggedSignals] = IEEE30_StepFunction_genloss(Action,LoggedSignals)

% This function applies the given action to the environment and evaluates
% the system dynamics for one simulation step.

a = Action;
d=[16 28 15];

% Unpack the state vector from the logged signals.
state = LoggedSignals.State;

[next_state, ~, genloss]=attack_eff_IEEE30(state, a, d);


LoggedSignals.State = next_state;

% Transform state to observation.
NextObs = LoggedSignals.State;

% Check terminal condition.

Down=nnz(~next_state);

IsDone = Down==41;

% Get reward.

Reward=genloss;
% Reward=genloss;

end