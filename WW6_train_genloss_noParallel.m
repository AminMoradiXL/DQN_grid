clc
clear 
close all

addpath ('dcsimsep-master')
setup

ObservationInfo = rlNumericSpec([1 11]);
ObservationInfo.Name = 'Line State';
ObservationInfo.Description = 'line1, line2, line3, line4, line5, line6, line7, line8, line9, line10, line11';
ObservationInfo.LowerLimit=0;
ObservationInfo.UpperLimit=1;

ActionInfo = rlFiniteSetSpec([1 2 3 4 5 6 7 8 9 10 11]);
ActionInfo.Name = 'Attacker Action';
ActionInfo.Description = ['attack-line1, attack-line2, attack-line3, attack-line4, ' ...
    'attack-line5, attack-line6, attack-line7, attack-line8, attack-line9, attack-line10, attack-line11'];

env = rlFunctionEnv(ObservationInfo, ActionInfo,'WW6_StepFunction_genloss','WW6_ResetFunction');

obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

% obsInfo.Dimension % 1 11
% actInfo.Dimension % 1 1

%% Hard
dnn = [
    featureInputLayer(obsInfo.Dimension(2),'Normalization','none','Name','state')
    fullyConnectedLayer(100,'Name','CriticStateFC1')
    reluLayer('Name','CriticRelu1')
    fullyConnectedLayer(100, 'Name','CriticStateFC2')
    reluLayer('Name','CriticRelu2')
    fullyConnectedLayer(100, 'Name','CriticStateFC3')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(length(actInfo.Elements),'Name','output')];
% 
% figure
% plot(layerGraph(dnn))

criticOpts = rlRepresentationOptions('LearnRate',0.001,'GradientThreshold',Inf);
criticOpts.Optimizer="adam"; % adam, sgdm , rmsprop
critic = rlQValueRepresentation(dnn,obsInfo,actInfo,'Observation',{'state'},criticOpts);
agentOpts = rlDQNAgentOptions(...
    'UseDoubleDQN',true, ...    
    'TargetSmoothFactor',1e-1, ...
    'TargetUpdateFrequency',4, ...   
    'ExperienceBufferLength',100000, ...
    'DiscountFactor',0.7, ...
    'MiniBatchSize',256 ...
    );

agentOpts.EpsilonGreedyExploration.Epsilon=1;
agentOpts.EpsilonGreedyExploration.EpsilonDecay=0.005;
agentOpts.EpsilonGreedyExploration.EpsilonMin=0.01;
% agentOpts.EpsilonGreedyExploration.EpsilonDecay=0.0005;

agent = rlDQNAgent(critic,agentOpts);
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',1000, ...
    'MaxStepsPerEpisode',5, ...
    'Verbose',false, ...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',900);

trainOpts.ScoreAveragingWindowLength=5;

trainingStats = train(agent,env,trainOpts);

