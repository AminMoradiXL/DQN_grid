clc
clear

addpath ('dcsimsep-master')
setup

state=ones(1,11);
% state=[0 0 0 1 1 1 1 1 1 0 1];
a=5;
d=[1 2];
[next_state, instant_out, genloss]=attack_eff_WW6(state, a, d)

