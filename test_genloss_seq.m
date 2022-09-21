clc
clear 

addpath ('dcsimsep-master')
setup

state=ones(1,41);
d=[1 2 3];
TotalGenloss=0;
sequence=[16 36 6 4 5 29 19 38 39 33];
for i=1:length(sequence)
    [next_state, instant_out, genloss]=attack_eff_IEEE30(state, sequence(i), d);
    TotalGenloss=genloss+TotalGenloss;
    state=next_state;
end

TotalGenloss
TotalGenloss/length(sequence)