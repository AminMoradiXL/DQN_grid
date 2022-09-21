function [next_state, instant_out, genloss]=attack_eff_WW6(state, a, d)

attacked=find(state==0);
attacked=[attacked a];
[~, failed_lines, failed_times]=WW6(attacked);

no_effect=(state(a)==0);
safe=0;
for i=1:length(d)
    if a==d(i)
        safe=1;
        break
    end
end
if safe==0
    failed_lines(length(failed_lines)+1)=a;
end
state_eff=zeros(1,length(state));
for i=1:length(failed_lines)
    state_eff(failed_lines(i))=-1;
end

if no_effect
    next_state=state;
else
    next_state=state+state_eff;
    next_state=max(next_state,zeros(1,length(state)));
end

x=state-next_state;
instant_out=~no_effect*sum(x);

if isempty(failed_times)
    ts=1;
else
    ts=failed_times(length(failed_times))*1.5;
end
s=0;
attacked=find(state==0);
attacked=[attacked a];

if isempty(failed_times)
    s=ts*WW6(attacked);
else
    for i=1:length(failed_times)+1
        if i==1
            s=s+WW6(attacked)*failed_times(i);
        elseif i==length(failed_times)+1
            s=s+WW6(attacked)*(ts-failed_times(i-1));
        else
            s=s+WW6(attacked)*(failed_times(i)-failed_times(i-1));
        end
        if i~=length(failed_times)+1
            attacked=[attacked failed_lines(i)];
        end
    end
end
genloss=~no_effect*s/ts;

end