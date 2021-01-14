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
end

% Fix data
dataLbls=categorical(trajSmall.lbls);

% Split data into folds
nReps=5;
nFolds=5;
foldID=ceil((1:length(dataLbls))/(length(dataLbls))*nFolds);
foldID=foldID(randperm(length(foldID)));

% Allocate variables
lblsEst=zeros(size(dataLbls));
scores=zeros(length(dataLbls),length(unique(trajSmall.lbls)));
trndNet=cell(nFolds,1);
valBAcc=zeros(nFolds,nReps);
testBAcc=zeros(nFolds,nReps);
lblsEstLog=cell(nFolds,1);
repBAcc=zeros(nReps,1);
clustLog=cell(nFolds,nReps);
classLog=cell(nFolds,nReps);

% untrndNet = [
%     sequenceInputLayer(3916,"Name","sequence","Normalization","zscore")
%     fullyConnectedLayer(3,"Name","fc_dim")
%     tanhLayer("Name","tanh_1")
%     fullyConnectedLayer(10,"Name","fc_cluster")
%     tanhLayer("Name","tanh_2")
%     bilstmLayer(20,"Name","bilstm","OutputMode","last")
%     fullyConnectedLayer(2,"Name","fc_out")
%     softmaxLayer("Name","softmax")
%     classificationLayer("Name","classoutput")];

untrndNet = layerGraph();

tempLayers = sequenceInputLayer(3916,"Name","sequence","Normalization","zscore");
untrndNet = addLayers(untrndNet,tempLayers);

tempLayers = [
    fullyConnectedLayer(3,"Name","fc_dim_1")
    tanhLayer("Name","tanh_1")
    fullyConnectedLayer(10,"Name","fc_cluster_1")];
untrndNet = addLayers(untrndNet,tempLayers);

tempLayers = [
    fullyConnectedLayer(3,"Name","fc_dim_2")
    tanhLayer("Name","tanh_3")
    fullyConnectedLayer(10,"Name","fc_cluster_2")];
untrndNet = addLayers(untrndNet,tempLayers);

tempLayers = [
    fullyConnectedLayer(3,"Name","fc_dim_5")
    tanhLayer("Name","tanh_6")
    fullyConnectedLayer(10,"Name","fc_cluster_5")];
untrndNet = addLayers(untrndNet,tempLayers);

tempLayers = [
    fullyConnectedLayer(3,"Name","fc_dim_3")
    tanhLayer("Name","tanh_4")
    fullyConnectedLayer(10,"Name","fc_cluster_3")];
untrndNet = addLayers(untrndNet,tempLayers);

tempLayers = [
    fullyConnectedLayer(3,"Name","fc_dim_4")
    tanhLayer("Name","tanh_5")
    fullyConnectedLayer(10,"Name","fc_cluster_4")];
untrndNet = addLayers(untrndNet,tempLayers);

tempLayers = [
    concatenationLayer(1,5,"Name","concat")
    tanhLayer("Name","tanh_2")
    bilstmLayer(20,"Name","bilstm","OutputMode","last")
    fullyConnectedLayer(length(unique(trajSmall.lbls)),"Name","fc_out")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","outLayer")];
untrndNet = addLayers(untrndNet,tempLayers);

% clean up helper variable
clear tempLayers;

untrndNet = connectLayers(untrndNet,"sequence","fc_dim_1");
untrndNet = connectLayers(untrndNet,"sequence","fc_dim_2");
untrndNet = connectLayers(untrndNet,"sequence","fc_dim_5");
untrndNet = connectLayers(untrndNet,"sequence","fc_dim_3");
untrndNet = connectLayers(untrndNet,"sequence","fc_dim_4");
untrndNet = connectLayers(untrndNet,"fc_cluster_5","concat/in1");
untrndNet = connectLayers(untrndNet,"fc_cluster_3","concat/in5");
untrndNet = connectLayers(untrndNet,"fc_cluster_2","concat/in4");
untrndNet = connectLayers(untrndNet,"fc_cluster_4","concat/in2");
untrndNet = connectLayers(untrndNet,"fc_cluster_1","concat/in3");

% Perform n-fold training
for currRep=1:nReps
    for currFold=1:nFolds
        % Determine index of fold data
        testIdx=find(foldID==currFold);
        trainIdx=find(foldID~=currFold);
        valIdx=trainIdx(1:3:end);
        trainIdx=setdiff(trainIdx,valIdx);
        
        % Compute class weights and update last network layer appropriately
        W=1./histcounts(dataLbls(trainIdx),'Normalization','probability');
        untrndLayers=untrndNet.Layers;
        untrndLayers(end)=trajClassLayer(W);
        untrndLayers(end).Name='outLayer';
        untrndNet=createLgraphUsingConnections(untrndLayers,untrndNet.Connections);

%         untrndNet(end)=trajClassLayer(W);
%         untrndNet(end).Name='outLayer';

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