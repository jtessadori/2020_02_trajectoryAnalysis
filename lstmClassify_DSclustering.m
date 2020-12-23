clear
close all
clc

% Load data and network template
load('2020_09_21_smallDataset_allClasses_RiemannMean.mat');
trajSmall.fixClasses('HC',{'PPMS','SPMS'});
load('20_11_12_classificationNetShort.mat','layers_1');

% Fix data
dataLbls=categorical(trajSmall.lbls);

% Split data into folds
nReps=5;
nFolds=5;
foldID=ceil((1:length(dataLbls))/(length(dataLbls))*nFolds);
foldID=foldID(randperm(length(foldID)));

% Allocate variables
lblsEst=zeros(size(dataLbls));
scores=zeros(length(dataLbls),2);
trndNet=cell(nFolds,1);
valBAcc=zeros(nFolds,nReps);
testBAcc=zeros(nFolds,nReps);
relData=num2cell(trajSmall.subTrajLbls,2);
lblsEstLog=cell(nFolds,1);
repBAcc=zeros(nReps,1);
classLog=cell(nFolds,nReps);

% Perform n-fold training
for currRep=1:nReps
    for currFold=1:nFolds
        testIdx=find(foldID==currFold);
        trainIdx=find(foldID~=currFold);
        valIdx=trainIdx(1:3:end);
        trainIdx=setdiff(trainIdx,valIdx);
        
        % Compute class weights and update last network layer appropriately
        W=1./histcounts(dataLbls(trainIdx),'Normalization','probability');
        layers_1(end)=customClassLayer(W,2);
        
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
            'ValidationData',{relData(valIdx),dataLbls(valIdx)}, ...
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
        tempNet=trainNetwork(relData(trainIdx),dataLbls(trainIdx),layers_1,options);
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
            [lblsEst(testIdx),scores(testIdx,:)]=trndNet{currFold}.classify(relData(testIdx));
        catch
            keyboard;
        end
        
        % Display intermediate results
        valEst=trndNet{currFold}.classify(relData(valIdx));
        currValBAcc=computeBAcc(dataLbls(valIdx),valEst);
        currTestBAcc=computeBAcc(dataLbls(testIdx),lblsEst(testIdx));
        lblsEstLog{currRep}=lblsEst;
        fprintf('Fold %d/%d, rep %d/%d: val BAcc %0.2f, test BAcc %0.2f\n',currFold,nFolds,currRep,nReps,currValBAcc,currTestBAcc);
        valBAcc(currFold,currRep)=currValBAcc;
        testBAcc(currFold,currRep)=currTestBAcc;
    end
    repBAcc(currRep)=computeBAcc(dataLbls,lblsEst);
end