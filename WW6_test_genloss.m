clc
clear all

load('xp3.mat','agent');

addpath ('dcsimsep-master')
setup

generatePolicyFunction(agent)

%% Actual
d=[11 7];
state=[1 1 1 1 1 1 1 1 1 1 1];
Reward=0;
i=1;
IsDone=0;

while IsDone==0 && i<=5
    if i==1
        s=state;
    else
        s=next_state;
    end

    a(i)=evaluatePolicy(s);
    [next_state, instant_out, genloss]=attack_eff_WW6(s, a(i), d);

    Down=nnz(~next_state);
    IsDone = Down==11;
    r=genloss;
    i=i+1;
    Reward=Reward+r;
end

