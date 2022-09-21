function [NextObs,Reward,IsDone,LoggedSignals] = WW6_StepFunction_genloss(Action,LoggedSignals)

% This function applies the given action to the environment and evaluates
% the system dynamics for one simulation step.

a = Action;
d=[5 2];

% Unpack the state vector from the logged signals.
state = LoggedSignals.State;

[next_state, ~, genloss]=attack_eff_WW6(state, a, d);


LoggedSignals.State = next_state;

% Transform state to observation.
NextObs = LoggedSignals.State;

% Check terminal condition.

Down=nnz(~next_state);

IsDone = Down==11;

% Get reward.

Reward=genloss;
% Reward=genloss;

end