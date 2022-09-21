clc
clear all

load('xp.mat','agent');

addpath ('dcsimsep-master')
setup

generatePolicyFunction(agent)

%% Actual
test_number=100;
d=[1 2];
obj=4;
obj_step=1/obj;

state=[1 1 1 1 1 1 1 1 1 1 1];
Reward=0;
i=1;
IsDone=0;
while IsDone==0 && i<5
    if i==1
        s=state;
    else
        s=next_state;
    end

    a(i)=evaluatePolicy(s);
    [next_state, instant_out, genloss]=attack_eff_WW6(s, a(i), d);

    Down=nnz(~next_state);
    IsDone = Down>=obj;
%     IsDone = genloss==210;

    r=RewardFunction(instant_out,obj,IsDone);
%     r=genloss;
    i=i+1;
    Reward=Reward+r;
 end

