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

actionInfo = getActionInfo(env);
observationInfo = getObservationInfo(env);
numObs = observationInfo.Dimension(2);
numAct = numel(actionInfo.Elements);

rng(0)

%% Critic Layer
criticLayerSizes = [400 300];
actorLayerSizes = [400 300];

criticNetwork = [
        featureInputLayer(numObs,'Normalization','none','Name','observation')
        fullyConnectedLayer(criticLayerSizes(1),'Name','CriticFC1', ...
            'Weights',sqrt(2/numObs)*(rand(criticLayerSizes(1),numObs)-0.5), ...
            'Bias',1e-3*ones(criticLayerSizes(1),1))
        reluLayer('Name','CriticRelu1')
        fullyConnectedLayer(criticLayerSizes(2),'Name','CriticFC2', ...
            'Weights',sqrt(2/criticLayerSizes(1))*(rand(criticLayerSizes(2),criticLayerSizes(1))-0.5), ...
            'Bias',1e-3*ones(criticLayerSizes(2),1))
        reluLayer('Name','CriticRelu2')
        fullyConnectedLayer(1,'Name','CriticOutput', ...
            'Weights',sqrt(2/criticLayerSizes(2))*(rand(1,criticLayerSizes(2))-0.5), ...
            'Bias',1e-3)];
criticOpts = rlRepresentationOptions('LearnRate',1e-4);
critic = rlValueRepresentation(criticNetwork,observationInfo,'Observation',{'observation'},criticOpts);

%% Actor Network
actorNetwork = [featureInputLayer(numObs,'Normalization','none','Name','observation')
        fullyConnectedLayer(actorLayerSizes(1),'Name','ActorFC1', ...
            'Weights',sqrt(2/numObs)*(rand(actorLayerSizes(1),numObs)-0.5), ...
            'Bias',1e-3*ones(actorLayerSizes(1),1))
        reluLayer('Name','ActorRelu1')
        fullyConnectedLayer(actorLayerSizes(2),'Name','ActorFC2', ...
            'Weights',sqrt(2/actorLayerSizes(1))*(rand(actorLayerSizes(2),actorLayerSizes(1))-0.5), ...
            'Bias',1e-3*ones(actorLayerSizes(2),1))
        reluLayer('Name', 'ActorRelu2')
        fullyConnectedLayer(numAct,'Name','Action', ...
            'Weights',sqrt(2/actorLayerSizes(2))*(rand(numAct,actorLayerSizes(2))-0.5), ...
            'Bias',1e-3*ones(numAct,1))
        softmaxLayer('Name','actionProb')];
actorOpts = rlRepresentationOptions('LearnRate',1e-3);
actor = rlStochasticActorRepresentation(actorNetwork,observationInfo,actionInfo,...
    'Observation',{'observation'},actorOpts);

%% Agent
Ts = 0.1;
agentOpts = rlPPOAgentOptions(...
                'ExperienceHorizon',600,... 
                'ClipFactor',0.02,...
                'EntropyLossWeight',0.01,...
                'MiniBatchSize',256,...
                'NumEpoch',3,...
                'AdvantageEstimateMethod','gae',...
                'GAEFactor',0.95,...
                'SampleTime',Ts,...
                'DiscountFactor',0.7);
agent = rlPPOAgent(actor,critic,agentOpts);

%% Training
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',10000,...
    'MaxStepsPerEpisode',5,...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',500,...
    'ScoreAveragingWindowLength',25,...
    'SaveAgentCriteria',"EpisodeReward",...
    'SaveAgentValue',700);
trainingStats = train(agent,env,trainOpts);
