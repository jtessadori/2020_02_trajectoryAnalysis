clear
close all
clc

% Load data and network template
load('2020_09_21_smallDataset_allClasses_RiemannMean.mat');
% trajSmall.fixMultipleClasses({'HC',{'PPMS','SPMS'},'RRMS'});
trajSmall.fixClasses('HC',{'PPMS','SPMS'});

% Compute Euclidean projections of FC matrices
projs=cell(length(trajSmall.FC),1);
for currSubj=1:length(projs)
    projs{currSubj}=zeros(trajSmall.nROIs*(trajSmall.nROIs+1)/2,size(trajSmall.FC{currSubj},3));
    for currSample=1:size(trajSmall.FC{currSubj},3)
        projs{currSubj}(:,currSample)=real(trajSmall.RS.project(squeeze(trajSmall.FC{currSubj}(:,:,currSample))));
    end
%     projs{currSubj}=projs{currSubj}-repmat(projs{currSubj}(:,1),[1,size(projs{currSubj},2)]);
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
    targetData{currSubj}=[projs{currSubj};zeros(3,size(projs{currSubj},2))];
end

% Perform n-fold training
for currRep=1:nReps
    for currFold=1:nFolds
        % Determine index of fold data
        testIdx=find(foldID==currFold);
        trainIdx=find(foldID~=currFold);
        valIdx=trainIdx(1:3:end);
        trainIdx=setdiff(trainIdx,valIdx);

        % Load template of classification network and modify it
        untrndNet=load('21_01_10b_clusterClassifyNet.mat');
        untrndNet=untrndNet.untrndNet;
        
%         untrndNet(6)=lstmLayer(length(unique(trajSmall.lbls)),'OutputMode','last');
        
        % Compute class weights and update last network layer appropriately
        W=1./histcounts(dataLbls(trainIdx),'Normalization','probability');
        untrndNet(end)=customClassLayer(W,lossCorrection);
        
        batchSize=150;
        options = trainingOptions('adam', ...
            'ExecutionEnvironment','gpu', ...
            'MaxEpochs',2500, ...
            'MiniBatchSize',batchSize, ...
            'GradientThreshold',1, ...
            'Verbose',false, ...
            'Plots','training-progress', ... %training-progress
            'Shuffle','every-epoch', ... % every-epoch
            'InitialLearnRate',5e-3, ...
            'ValidationData',{projs(valIdx),dataLbls(valIdx)}, ...
            'ValidationFrequency',10,... 
            'ValidationPatience',15,...
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
        bestNet=load([pwd,'\tempNets\',D(netOrdr(end-options.ValidationPatience*options.ValidationFrequency+1)).name]);
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