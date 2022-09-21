clc
clear all

load('zp2.mat','agent');

addpath ('dcsimsep-master')
setup

generatePolicyFunction(agent)

%% Actual
d=[15 16];
obj=5;
obj_step=1/obj;

state=ones(1, 41);
% state=zeros(1, 41);

Reward=0;
i=1;
IsDone=0;
while IsDone==0 && i<=4
    if i==1
        s=state;
    else
        s=next_state;
    end

    a(i)=evaluatePolicy(s);
    [next_state, instant_out, genloss]=attack_eff_IEEE30(s, a(i), d);

    Down=nnz(~next_state);
    IsDone = Down>=obj;
    r=RewardFunction(instant_out,obj,IsDone);

    i=i+1;
    Reward=Reward+r;
end

