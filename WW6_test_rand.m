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

Reward=zeros(12,1);
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
temp=state;

for test=1:12
    i=1;
    IsDone=0;
    while IsDone==0 && i<5
        if i==1
            s=state(test,:);
        else
            s=next_state;
        end

        a(test,i)=evaluatePolicy(s);
        [next_state, instant_out, genloss]=attack_eff_WW6(s, a(test,i), d);

        Down=nnz(~next_state);
        IsDone = Down>=obj;
        r=RewardFunction(instant_out,obj,IsDone);

        i=i+1;
        Reward(test)=Reward(test)+r;
    end
end

X=[state,Reward,a];