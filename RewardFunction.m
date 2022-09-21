function Reward=RewardFunction(instant_out,obj,IsDone)
if instant_out>=obj
    Reward = 10;
elseif IsDone
    Reward = 1;
else
    Reward=2*(1/obj)*instant_out;
end
end