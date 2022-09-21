clc
clear 
close all

addpath ('dcsimsep-master')
setup

ObservationInfo = rlNumericSpec([1 41]);
ObservationInfo.Name = 'Line State';
ObservationInfo.Description = ['line1, line2, line3, line4, line5, line6, line7, line8, line9, line10, ' ...
    'line11, line12, line13, line14, line15, line16, line17, line18, line19, line20, ' ...
    'line21, line22, line23, line24, line25, line26, line27, line28, line29, line30, ' ...
    'line31, line32, line33, line34, line35, line36, line37, line38, line39, line40, line41' ];
ObservationInfo.LowerLimit=0;
ObservationInfo.UpperLimit=1;

ActionInfo = rlFiniteSetSpec([1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 ...
    19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41]);
ActionInfo.Name = 'Attacker Action';
ActionInfo.Description = ['attack-line1, attack-line2, attack-line3, attack-line4, attack-line5, attack-line6, attack-line7, attack-line8, attack-line9, attack-line10, ' ...
    'attack-line11, attack-line12, attack-line13, attack-line14, attack-line15, attack-line16, attack-line17, attack-line18, attack-line19, attack-line20, ' ...
    'attack-line21, attack-line22, attack-line23, attack-line24, attack-line25, attack-line26, attack-line27, attack-line28, attack-line29, attack-line30, ' ...
    'attack-line31, attack-line32, attack-line33, attack-line34, attack-line35, attack-line36, attack-line37, attack-line38, attack-line39, attack-line40, attack-line41' ];

env = rlFunctionEnv(ObservationInfo, ActionInfo,'IEEE30_StepFunction_genloss','IEEE30_ResetFunction');

obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

%% Critic 
criticNetwork = [
    featureInputLayer(obsInfo.Dimension(2),'Normalization','none','Name','state')
    fullyConnectedLayer(200,'Name','CriticStateFC1')
    reluLayer('Name','CriticRelu1')
    fullyConnectedLayer(200, 'Name','CriticStateFC2')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(1,'Name','output')];

% figure
% plot(layerGraph(criticNetwork))

criticOpts = rlRepresentationOptions('LearnRate',0.001,'GradientThreshold',1);
critic = rlValueRepresentation(criticNetwork,obsInfo,'Observation',{'state'},criticOpts);

%% Actor
actorNetwork = [
    featureInputLayer(obsInfo.Dimension(2),'Normalization','none','Name','state')
    fullyConnectedLayer(200,'Name','ActorStateFC1')
    reluLayer('Name','ActorRelu1')
    fullyConnectedLayer(200, 'Name','ActorStateFC2')
    reluLayer('Name','ActorCommonRelu')
    fullyConnectedLayer(obsInfo.Dimension(2),'Name','fc')
    softmaxLayer('Name','actionProb')];

% figure
% plot(layerGraph(actorNetwork))

actorOpts = rlRepresentationOptions('LearnRate',0.001,'GradientThreshold',1);

actor = rlStochasticActorRepresentation(actorNetwork,obsInfo,actInfo,...
    'Observation',{'state'},actorOpts);

%% Agent

agentOpts = rlACAgentOptions(...
    'NumStepsToLookAhead',32, ...
    'DiscountFactor',0.7);
agent = rlACAgent(actor,critic,agentOpts);

%% Training
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',10000,...
    'MaxStepsPerEpisode',5,...
    'Verbose',false,...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',2000,...
    'ScoreAveragingWindowLength',20); 

trainingStats = train(agent,env,trainOpts);

