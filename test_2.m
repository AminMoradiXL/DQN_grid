clc 
clear

addpath ('dcsimsep-master')
setup

attack_sequence=[15 16];
s(1,:)=ones(1,41);
for i=1:length(attack_sequence)
%     [ns,ins,genloss(i)]=attack_eff_WW6(s(i,:),attack_sequence(i),[1 2]);
    [ns,ins,genloss(i)]=attack_eff_IEEE30(s(i,:),attack_sequence(i),[1 2 3]);
    s(i+1,:)=ns;
end

genloss
s