function [InitialObservation, LoggedSignal] = WW6_ResetFunction()
% Reset function to place transmission line status into the initial state
% (all working)

% Return initial environment state variables as logged signals.

% LoggedSignal.State = ones(1,11);

state=[1 1 1 1 1 1 1 1 1 1 1;
    0 1 1 1 1 1 1 1 1 1 1;
    1 0 1 1 1 1 1 1 1 1 1;
    1 1 0 1 1 1 1 1 1 1 1;
    1 1 1 0 1 1 1 1 1 1 1;
    1 1 1 1 0 1 1 1 1 1 1;
    1 1 1 1 1 0 1 1 1 1 1;
    1 1 1 1 1 1 0 1 1 1 1;
    1 1 1 1 1 1 1 0 1 1 1;
    1 1 1 1 1 1 1 1 0 1 1;
    1 1 1 1 1 1 1 1 1 0 1;
    1 1 1 1 1 1 1 1 1 1 0];

LoggedSignal.State = state(randi([1 11]),:);

InitialObservation = LoggedSignal.State;

end