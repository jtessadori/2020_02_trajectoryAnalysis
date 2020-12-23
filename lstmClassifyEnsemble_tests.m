clear
close all
clc

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

% Fix data
dataLbls=categorical(trajSmall.lbls);

% Split data into folds
nReps=25;
nFolds=5;
nEns=5;
foldID=ceil((1:length(dataLbls))/(length(dataLbls))*nFolds);
foldID=foldID(randperm(length(foldID)));

lossCorrection=3;

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

% Perform n-fold training
for currRep=1:nReps
    for currFold=1:nFolds
        % Determine index of fold data        
        testIdx=find(foldID==currFold);
        trainIdx=find(foldID~=currFold);
        valIdx=trainIdx(1:3:end);
        trainIdx=setdiff(trainIdx,valIdx);
        
        layers=cell(nEns,1);
        for currEns=1:nEns
            % Define nested training indeces
            subFoldID=ceil((1:length(trainIdx))/(length(trainIdx))*nEns);
            subFoldID=subFoldID(randperm(length(subFoldID)));
            subTrainIdx=find(subFoldID==currEns);
            subValIdx=find(subFoldID~=currEns);

            % Define training options for clustering network
            batchSize=5;
            clusterLayers=load('20_12_22_clusterLayers.mat','clusterLayers');
            clusterLayers=clusterLayers.clusterLayers;
            
            outLayerIdx=strcmpi({clusterLayers.Name},'regressionoutput');
            clusterLayers(outLayerIdx)=clusterLayer;
            clusterLayers(outLayerIdx).Name='regressionoutput';
            
            % Rename layers so to distinguish between different ensembles
            for currLayer=1:length(clusterLayers)
                clusterLayers(currLayer).Name=sprintf('%s_%s',clusterLayers(currLayer).Name,num2str(currEns));
            end
            
            options = trainingOptions('adam', ...
                'ExecutionEnvironment','gpu', ...
                'MaxEpochs',1500, ...
                'MiniBatchSize',batchSize, ...
                'GradientThreshold',1, ...
                'Verbose',false, ...
                'Plots','none', ... %training-progress
                'Shuffle','every-epoch', ... % every-epoch
                'InitialLearnRate',5e-3, ...
                'ValidationData',{projs(trainIdx(subValIdx)),projs(trainIdx(subValIdx))}, ...
                'ValidationPatience',5, ...
                'ValidationFrequency',1, ...
                'LearnRateSchedule','piecewise', ...
                'LearnRateDropPeriod',1, ...
                'LearnRateDropFactor',0.99,...
                'OutputFcn',@(info)outTrainingCurves(info));
            
            % Plot curves only for first fold
            if currRep>1
                options.Plots='none';
            end
            
            % Train clustering network
            clusterNet=trainNetwork(projs(trainIdx(subTrainIdx)),projs(trainIdx(subTrainIdx)),clusterLayers,options);
            clustLog{currFold,currRep}=evalin('base','trainLog');
            
            % Use clustering network as starting point for following
            % step
            layers{currEns}=clusterNet.Layers;
            layers{currEns}(4)=fullyConnectedLayer(10);
            layers{currEns}(4).Weights=clusterNet.Layers(4).Weights(1:10,:);
            layers{currEns}(4).Bias=clusterNet.Layers(4).Bias(1:10);
            layers{currEns}(4).Name=clusterNet.Layers(4).Name;
        end
        
        % Load template of classification network
        untrndNet=load('20_12_10_clusterClassifyNet.mat');
        netLayers=untrndNet.untrndNet;
        
        % Assemble ensemble network
        untrndNet=[layers{1}(1:4);softmaxLayer('Name','softmax_1');concatenationLayer(1,nEns,'Name','concLayer');netLayers(7:10)];
        W=1./histcounts(dataLbls(trainIdx),'Normalization','probability');
        untrndNet(end)=customClassLayer(W,lossCorrection);
        untrndNet(end).Name='customClassLayer';
        untrndNet=layerGraph(untrndNet);
        for currEns=2:nEns
            untrndNet=untrndNet.addLayers(layers{currEns}(2));
            untrndNet=untrndNet.connectLayers('sequence_1',layers{currEns}(2).Name);
            untrndNet=untrndNet.addLayers(layers{currEns}(3));
            untrndNet=untrndNet.connectLayers(layers{currEns}(2).Name,layers{currEns}(3).Name);
            untrndNet=untrndNet.addLayers(layers{currEns}(4));
            untrndNet=untrndNet.connectLayers(layers{currEns}(3).Name,layers{currEns}(4).Name);
            untrndNet=untrndNet.addLayers(softmaxLayer('Name',sprintf('softmax_%d',currEns)));
            untrndNet=untrndNet.connectLayers(layers{currEns}(4).Name,sprintf('softmax_%d',currEns));
            untrndNet=untrndNet.connectLayers(sprintf('softmax_%d',currEns),sprintf('concLayer/in%d',currEns));
        end
        
        
        batchSize=150;
        valPat=50;
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