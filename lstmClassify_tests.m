clear
close all
clc

global trMat1
global trMat2
global cnts1
global cnts2

% Load data and network template
load('2020_09_21_smallDataset_allClasses_RiemannMean.mat');
trajSmall.fixClasses('HC',{'PPMS','SPMS'});

% Compute Euclidean projections of FC matrices
projs=cell(length(trajSmall.FC),1);
for currSubj=1:length(projs)
    projs{currSubj}=zeros(trajSmall.nROIs*(trajSmall.nROIs+1)/2,size(trajSmall.FC{currSubj},3));
    for currSample=1:size(trajSmall.FC{currSubj},3)
        projs{currSubj}(:,currSample)=real(trajSmall.RS.project(squeeze(trajSmall.FC{currSubj}(:,:,currSample))));
    end
end

% % Align projections with principal axis
% [~,score]=pca(cat(2,projs{:})');
% projs=mat2cell(score',size(score,2),repmat(size(projs{1},2),length(projs),1))';

% Fix data
dataLbls=categorical(trajSmall.lbls);

% Split data into folds
nReps=5;
nFolds=5;
foldID=ceil((1:length(dataLbls))/(length(dataLbls))*nFolds);
foldID=foldID(randperm(length(foldID)));

lossCorrection=0;

% Allocate variables
lblsEst=zeros(size(dataLbls));
scores=zeros(length(dataLbls),2);
trndNet=cell(nFolds,1);
valBAcc=zeros(nFolds,nReps);
testBAcc=zeros(nFolds,nReps);
lblsEstLog=cell(nFolds,1);
repBAcc=zeros(nReps,1);
clustLog=cell(nFolds,nReps);
classLog=cell(nFolds,nReps);
targetData=cell(length(dataLbls),1);
for currSubj=1:length(dataLbls)
    targetData{currSubj}=[projs{currSubj};repmat(double(dataLbls(currSubj)),1,size(projs{currSubj},2))];
end

% Perform n-fold training
for currRep=1:nReps
    for currFold=1:nFolds
        % Determine index of fold data
        testIdx=find(foldID==currFold);
        trainIdx=find(foldID~=currFold);
        valIdx=trainIdx(1:3:end);
        trainIdx=setdiff(trainIdx,valIdx);
        
        % Define training options for clustering network
        batchSize=1;
        clusterLayers=load('20_12_22_clusterLayers.mat','clusterLayers');
        clusterLayers=clusterLayers.clusterLayers;
        
        clusterLayers(strcmpi({clusterLayers.Name},'regressionoutput'))=clusterLayer;
        
        % Clear global variables
        trMat1=[];
        trMat2=[];
        cnts1=[];
        cnts2=[];
        
        options = trainingOptions('adam', ...
            'ExecutionEnvironment','gpu', ...
            'MaxEpochs',1500, ...
            'MiniBatchSize',batchSize, ...
            'GradientThreshold',1, ...
            'Verbose',false, ...
            'Plots','training-progress', ... %training-progress
            'Shuffle','every-epoch', ... % every-epoch
            'InitialLearnRate',5e-3, ...
            'ValidationData',{projs(valIdx),targetData(valIdx)}, ...
            'ValidationPatience',3, ...
            'ValidationFrequency',15, ...
            'LearnRateSchedule','piecewise', ...
            'LearnRateDropPeriod',1, ...
            'LearnRateDropFactor',0.99,...
            'OutputFcn',@(info)outTrainingCurves(info));
        
        % Plot curves only for first fold
        if currRep>1
            options.Plots='none';
        end
        
        % Train clustering network
        clusterNet=trainNetwork(projs(trainIdx),targetData(trainIdx),clusterLayers,options);
        clustLog{currFold,currRep}=evalin('base','trainLog');

        % Use clustering network as starting point for following
        % step
        layers=clusterNet.Layers;
        layers(4)=fullyConnectedLayer(10);
        layers(4).Weights=clusterNet.Layers(4).Weights(1:10,:);
        layers(4).Bias=clusterNet.Layers(4).Bias(1:10);
        layers(4).Name=clusterNet.Layers(4).Name;
        
        % Load template of classification network and modify it
        untrndNet=load('20_12_10_clusterClassifyNet.mat');
        netLayers=untrndNet.untrndNet;
        untrndNet=[layers(1:5);softmaxLayer('Name','softmax_1');netLayers(7:10)];
        
        % Compute class weights and update last network layer appropriately
        W=1./histcounts(dataLbls(trainIdx),'Normalization','probability');
        untrndNet(end)=customClassLayer(W,lossCorrection);
        
        batchSize=150;
        valPat=25;
        options = trainingOptions('adam', ...
            'ExecutionEnvironment','gpu', ...
            'MaxEpochs',500, ...
            'MiniBatchSize',batchSize, ...
            'GradientThreshold',1, ...
            'Verbose',false, ...
            'Plots','training-progress', ... %training-progress
            'Shuffle','every-epoch', ... % every-epoch
            'InitialLearnRate',5e-3, ...
            'ValidationData',{projs(valIdx),dataLbls(valIdx)}, ...
            'ValidationFrequency',1,... 
            'ValidationPatience',valPat,...
            'LearnRateSchedule','piecewise', ...
            'LearnRateDropPeriod',10, ...
            'LearnRateDropFactor',0.995, ...
            'L2Regularization', 1e-2,...
            'CheckpointPath',[pwd,'\tempNets'],...
            'OutputFcn',@(info)outTrainingCurves(info));
        
        % Plot curves only for first fold
        if currRep>1
            options.Plots='none';
        end
        
        % Train network
        tempNet=trainNetwork(projs(trainIdx),dataLbls(trainIdx),untrndNet,options);
        classLog{currFold,currRep}=evalin('base','trainLog');
        
        % Recover best net, then clear folder
        D=dir([pwd,'\tempNets']);
        D(1:2)=[];
        [~,netOrdr]=sort({D(:).date});
        bestNet=load([pwd,'\tempNets\',D(netOrdr(end-valPat)).name]);
        trndNet{currFold}=bestNet.net;
        delete([pwd,'\tempNets\*.*']);
        
        % Estimate results
        try
            [lblsEst(testIdx),scores(testIdx,:)]=trndNet{currFold}.classify(projs(testIdx));
        catch
            keyboard;
        end
        
        % Display intermediate results
        valEst=trndNet{currFold}.classify(projs(valIdx));
        currValBAcc=computeBAcc(dataLbls(valIdx),valEst);
        currTestBAcc=computeBAcc(dataLbls(testIdx),lblsEst(testIdx));
        lblsEstLog{currRep}=lblsEst;
        fprintf('Fold %d/%d, rep %d/%d: val BAcc %0.2f, test BAcc %0.2f\n',currFold,nFolds,currRep,nReps,currValBAcc,currTestBAcc);
        valBAcc(currFold,currRep)=currValBAcc;
        testBAcc(currFold,currRep)=currTestBAcc;
    end
    repBAcc(currRep)=computeBAcc(dataLbls,lblsEst);
end