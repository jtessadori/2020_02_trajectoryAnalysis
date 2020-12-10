classdef traj < handle
    % February 2020, Jacopo Tessadori
    % The concept here is to compute some meaningful distance between
    % matrices for dynamic connectivity and apply some trajectory
    % classification in the resulting space
    
    properties
        dataPath;
        FC;
        lbls;
        nROIs;
        nSubjs;
        nDims;
        seqLength;
        RS;
        D;
        subTrajLbls;
        feats;
        HMMlog;
        slicesPerMat=30;
        stepSize=2;
        maxDims=3;
        simScale=500;
        age;
    end
    
    methods
        function this=traj(path)
            % Constructor for traj class.
            % Only argument (required) is absolut path of data file.
            this.dataPath=path;
        end
        
        function subjDataLog=loadRawData(this)
            % Load xls file
            subjInfoFile=dir(sprintf('%s\\*.xls*',this.dataPath));
            subjInfoTable=readtable(sprintf('%s\\%s',this.dataPath,subjInfoFile.name),'Range','A1:B160');
            subjAgeTable=readtable(sprintf('%s\\%s',this.dataPath,subjInfoFile.name),'Range','I1:I160');
            
            % Recover relevant portion of PAT_ID
            patID=cellfun(@(x)x([3:4,6:8]),subjInfoTable.PAT_ID,'UniformOutput',false);
            
            % Generate appropriate filenames to load
            fileNames=cellfun(@(x)sprintf('sub-%s_ROI_mean_val_my.txt',x),patID,'UniformOutput',false);
            
            % Load all file contents
            subjDataLog=cell(length(fileNames),1);
            for currFile=1:length(fileNames)
                subjData=dlmread(sprintf('%s\\%s',this.dataPath,fileNames{currFile}));
                
                % Remove "bad" ROIs (anything past 90 and 75 & 76)
                subjData=subjData(:,[1:74,77:90]);
                
                % Log subjData, if requested
                if nargout>0
                    subjDataLog{currFile}=subjData;
                end
                
                % Compute covariance matrices
                nSteps=(size(subjData,1)-this.slicesPerMat)/this.stepSize+1;
                if ~isfinite(nSteps)
                    nSteps=1;
                end
                subjCorr=zeros(size(subjData,2),size(subjData,2),nSteps);
                for currSlice=1:nSteps
                    subjCorr(:,:,currSlice)=corrcoef(subjData((currSlice-1)*this.stepSize+1:(currSlice-1)*this.stepSize+this.slicesPerMat,:));
                end
                this.FC{currFile}=subjCorr;
            end
            
            % Recover lbls info
            this.lbls=subjInfoTable.Group;
            
            % Recover age info
            this.age=subjAgeTable.Age;
        end
        
        function loadDataHC_RR(this)
            % Recovers dynamical functional connectivity matrices as well
            % as lbls from the file specified as dataPath. Task is
            % classification between HC and RR subjects
            inData=load(this.dataPath);
            
            % Old datasets may have data split in different ways
            if isfield(inData,'CM')
                if length(inData.CM)==28
                    % This is a very specific test dataset, define lbls
                    % here
                    this.lbls=zeros(1,28);
                    this.lbls(15:end)=1;
                    this.FC=inData.CM;
                end
            else
                % Define labels and remove unlabeled data
                this.lbls=cat(1,zeros(length(inData.idHC),1),ones(length(inData.idRR),1));
                inData.cm_all_subj_corr=inData.cm_all_subj_corr(union(inData.idHC,inData.idRR));
                this.FC=inData.cm_all_subj_corr;
            end
        end
        
        function loadDataHC_MS(this)
            % Recovers dynamical functional connectivity matrices as well
            % as lbls from the file specified as dataPath. Task is
            % classification between HC and union of SPMS and PPMS subjects
            inData=load(this.dataPath);
            
            % Old datasets may have data split in different ways
            if isfield(inData,'CM')
                if length(inData.CM)==28
                    % This is a very specific test dataset, define lbls
                    % here
                    this.lbls=zeros(1,28);
                    this.lbls(15:end)=1;
                    this.FC=inData.CM;
                end
            else
                % Define labels and remove unlabeled data
                try
                    HCidx=find(inData.HC_All_lab==0);
                    MSidx=find(ismember(inData.HC_All_lab,[2,3]));
                    this.lbls=cat(1,zeros(length(HCidx),1),ones(length(MSidx),1));
                    inData.cm_all_subj_corr=inData.cm_all_subj_corr(union(HCidx,MSidx));
                    this.FC=inData.cm_all_subj_corr;
                catch
                    warning('Data path file does not contain lbl information. Trying to recover it from xls file');
                    num=xlsread(inData.xlsFile);
                    relLbls=num(:,8);
                    relLbls(isnan(relLbls))=0;
                    this.FC=inData.FCs(relLbls~=1); % Excluding RR and bening
                    relLbls(relLbls==1)=[];
                    this.lbls=ismember(relLbls,[2,3]);
                end
            end
        end
        
        function loadDataImprovement(this)
            % Recovers dynamical functional connectivity matrices as well
            % as lbls from the file specified as dataPath. Task is
            % classification between stable/improving subjects and
            % worsening ones
            inData=load(this.dataPath);
            
            this.lbls=inData.lbls;
            this.FC=inData.FCs;
        end
        
        function computeRiemannMean(this)
            % Warn user that this will take several minutes
            fprintf('Computing Riemannian mean. This will take between several minutes and several hours\n');
%             fprintf('Computing Riemannian mean takes too long. It has been substituted here with the log-Euclidean mean\n');
            
            % Compute Riemann average of points
            FCs=permute(cat(3,this.FC{:}),[3,1,2]);
            FCs=FCs+repmat(permute(eye(this.nROIs)*1e-5,[3,1,2]),size(FCs,1),1,1); % Add a small regularization term
            this.RS=riemannSpace(FCs,'cov');
        end
        
        function plotData(this)
            % Project all matrices in tangent space
            FCs=permute(cat(3,this.FC{:}),[3,1,2]);
            FCs=FCs+repmat(permute(eye(this.nROIs)*1e-6,[3,1,2]),size(FCs,1),1,1); % Add a small regularization term
            projs=zeros((this.nROIs*(this.nROIs+1))/2,size(FCs,1));
            for currFC=1:size(FCs,1)
                projs(:,currFC)=real(this.RS.project(squeeze(FCs(currFC,:,:))));
            end
            
            % Compute PCA of original projections and plot results
            [~,score]=pca(projs');
            score3D=reshape(score(:,1:3),this.seqLength,this.nSubjs,3);
            figure;
            hold on
            clrs='rk';
            for currSubj=1:this.nSubjs
                plot3(score3D(:,currSubj,1),score3D(:,currSubj,2),score3D(:,currSubj,3),clrs(this.lbls(currSubj)+1));
            end
            view(3)
            
            % Plot effect of compression on curve with maximum extension on
            % dim 1
            [~,maxExtIdx]=max(max(score3D(:,:,1))-min(score3D(:,:,1)));
            figure;
            hold on;
            plot(score3D(:,maxExtIdx,1))
            plot(atan(score3D(:,maxExtIdx,1)/(3*mad(score(:,1))))*(3*mad(score(:,1))));
            xlabel('Time slice')
            ylabel('Feature amplitude [a.u.]')
            legend({'Original Data','Compressed Data'},'Location','best')
            h=findall(gcf,'Type','Line');
            set(h,'LineWidth',1.5)
            set(gca,'LineWidth',1.5,'TickDir','out')
            axis tight
            
            % Compute PCA results after compression
            figure;
            hold on
            clrs='rk';
            for currSubj=1:this.nSubjs
                plot3(this.feats(:,currSubj,1),this.feats(:,currSubj,2),this.feats(:,currSubj,3),clrs(this.lbls(currSubj)+1));
            end
            view(3)
            
            % Plot grouped data
            groups=reshape(this.subTrajLbls',[],1);
            feats2D=reshape(this.feats,[],size(this.feats,3));
            figure;
            hold on;
            clrs='kycmgr';
            for currGroup=1:max(groups)
                plot3(feats2D(groups==currGroup,1),feats2D(groups==currGroup,2),feats2D(groups==currGroup,3),['.',clrs(mod(currGroup,length(clrs))+1)]);
            end
            view(3)
            
            % Plot histograms of groups in each class
            figure;
            lbls1D=reshape(repmat(this.lbls,1,this.seqLength)',[],1);
            [C1,edges]=histcounts(groups(lbls1D==0,:),'BinMethod','integer','Normalization','pdf');
            C2=histcounts(groups(lbls1D==1,:),edges,'Normalization','pdf');
            [C1,groupOrdr]=sort(C1,'descend');
            lin1=plot(C1);hold on;lin2=plot(C2(groupOrdr));
            xlabel('Group Idx');ylabel('Fractions of points per group')
            h=findall(gcf,'Type','Line');
            set(h,'LineWidth',1.5)
            set(gca,'LineWidth',1.5,'TickDir','out')
            axis tight
            
            % Add very rough confidence intervals
            C1count=histcounts(groups(lbls1D==0,:),edges,'Normalization','count');
            C1count=sort(C1count,'descend');
            p1=patch([1:length(C1),length(C1):-1:1],[(C1count+3*sqrt(C1count))./sum(C1count),(C1count(length(C1):-1:1)-3*sqrt(C1count(length(C1):-1:1)))./sum(C1count)],[0 0 1]);
            set(p1,'EdgeAlpha',0,'FaceAlpha',.3,'FaceColor',get(lin1,'Color'));
            C2count=histcounts(groups(lbls1D==1,:),edges,'Normalization','count');
            p2=patch([1:length(C2),length(C2):-1:1],[(C2count(groupOrdr)+3*sqrt(C2count(groupOrdr)))./sum(C2count),(C2count(groupOrdr(length(C1):-1:1))-3*sqrt(C2count(groupOrdr(length(C1):-1:1))))./sum(C2count)],[0 0 1]);
            set(p2,'EdgeAlpha',0,'FaceAlpha',.3,'FaceColor',get(lin2,'Color'));
            legend({'Class 1','Class 2'},'Location','best');
            
            
            % Plot image of groups in time
            figure;
            imagesc(this.subTrajLbls)
            xlabel('Time slice');
            ylabel('Subject');
            
%             % Plot classification results
%             title('Boxplot of balanced classificaiton accuracy')
%             boxplot(this.BAccReps,'labels',' ');
%             h=findall(gcf,'Type','Line');
%             set(h,'LineWidth',1.5)
%             set(gca,'LineWidth',1.5,'TickDir','out')
%             ylabel('BAcc');
        end
        
        function computeD(this)
            % Project all matrices in tangent space
            FCs=permute(cat(3,this.FC{:}),[3,1,2]);
            FCs=FCs+repmat(permute(eye(this.nROIs)*1e-6,[3,1,2]),size(FCs,1),1,1); % Add a small regularization term         
            projs=zeros((this.nROIs*(this.nROIs+1))/2,size(FCs,1));
            for currFC=1:size(FCs,1)
                projs(:,currFC)=this.RS.project(squeeze(FCs(currFC,:,:)));
            end

            % Compute distance matrix on reduced dimension set
            % Demean projections, compress outliers and compute PCA
            dProjs=real(projs-repmat(mean(projs,2),1,size(projs,2)));
            cProjs=atan(dProjs./repmat(3*mad(dProjs,1,2),1,size(projs,2))).*repmat(mad(dProjs,1,2),1,size(projs,2))*3;
            [~,score]=pca(cProjs');
            
            % Compute distance matrix on smaller number of dimensions
            score=score(:,1:this.maxDims);
            distFun=@(x1,x2)sqrt(sum((x1-x2).^2));
            this.D=zeros(size(score,1));
            for currX1=1:size(score,1)
                for currX2=currX1+1:size(score,1)
                    this.D(currX1,currX2)=distFun(score(currX1,:),score(currX2,:));
                    this.D(currX2,currX1)=this.D(currX1,currX2);
                end
                if mod(currX1,100)==0
                    fprintf('%d/%d\n',currX1,size(score,1));
                end
            end
            this.feats=reshape(score,this.seqLength,this.nSubjs,this.maxDims);
        end
        
        function clusterPoints(this)
            % Cluster points
            sigma=this.simScale*std(reshape(this.D,[],1));
            A=exp(-this.D./sigma);
            A=A.*not(eye(size(A)));
            C=dominantset(A);
            if any(C==0)
                C=C+1;
            end
            this.subTrajLbls=reshape(C,[],this.nSubjs)';
        end
        
        function fixClasses(this,class1,class2)
            % This function fixes data so that only class1 and class2
            % remain, both in labels and in actual data).
            % WARNING: this causes loss of data related to other classes
            tempLbls=nan(size(this.lbls));
            for currSubj=1:length(tempLbls)
                if ismember(this.lbls(currSubj),class1)
                    tempLbls(currSubj)=0;
                elseif ismember(this.lbls(currSubj),class2)
                    tempLbls(currSubj)=1;
                end
            end
            toBeRemoved=isnan(tempLbls);
            this.FC(toBeRemoved)=[];
            if size(this.feats,2)==length(this.lbls)
                this.feats(:,toBeRemoved,:)=[];
            end
            this.subTrajLbls(toBeRemoved,:)=[];
            this.age(toBeRemoved)=[];
            this.lbls=tempLbls;
            this.lbls(toBeRemoved)=[];
        end
                
        function [BAcc,logP]=HMMclassification(this)
            % Compute transition matrix, prior and distribution stats for
            % each class
            nStates=max(this.subTrajLbls,[],'all');
            nFolds=5;
            CV=cvpartition(length(this.lbls),'KFold',nFolds);
            logP=zeros(size(this.subTrajLbls,1),2);
            for currFold=1:nFolds
                trainData=this.feats(:,CV.training(currFold),:);
                trainGroups=this.subTrajLbls(CV.training(currFold),:);
                trainLbls=this.lbls(CV.training(currFold));
                testData=this.feats(:,CV.test(currFold),:);
                
                HMM=struct('TR',{},'GM',{},'prior',{},'states',{});
                for currClass=1:2
                    % Recover training cluster lbls only for current class
                    relData=reshape(permute(trainData(:,trainLbls==currClass-1,:),[2,1,3]),[],size(trainData,3));
                    relGroups=reshape(trainGroups(trainLbls==currClass-1,:),[],1);
                    
                    % Compute emissione probs
                    for currGroup=1:nStates
                        currData=relData(relGroups==currGroup,:);
                        if size(currData,1)>this.nDims
                            HMM(currClass).GM{currGroup}=fitgmdist(currData,1);
                        elseif size(currData,1)>1
                            HMM(currClass).GM{currGroup}=gmdistribution(mean(currData,1),std(currData,[],1));
                        elseif size(currData,1)==1
                            HMM(currClass).GM{currGroup}=gmdistribution(mean(currData,1),std(relData,[],1));
                        else
                            HMM(currClass).GM{currGroup}=gmdistribution(mean(relData,1),std(relData,[],1));
                        end
                    end
                    HMM(currClass).states=1:nStates;
                    
                    % Compute transmission probs and priors
                    relGroups=trainGroups(trainLbls==currClass-1,:);
                    HMM(currClass).TR=ones(nStates)/nStates;
                    HMM(currClass).prior=ones(nStates,1)/nStates;
                    for currSubj=1:size(relGroups,1)
                        HMM(currClass).prior(relGroups(currSubj,1))=HMM(currClass).prior(relGroups(currSubj,1))+1;
                        for currT=1:size(relGroups,2)-1
                            HMM(currClass).TR(relGroups(currSubj,currT),relGroups(currSubj,currT+1))=HMM(currClass).TR(relGroups(currSubj,currT),relGroups(currSubj,currT+1))+1;
                        end
                    end
                    HMM(currClass).TR=HMM(currClass).TR./repmat(sum(HMM(currClass).TR),nStates,1);
                    HMM(currClass).prior=HMM(currClass).prior/sum(HMM(currClass).prior);
                    
                    % Compute end-state log likelihoods and correct for
                    % unbalanced datasets
                    testSubjIdx=find(CV.test(currFold));
                    for currSubj2=1:size(testData,2)
                        relData=squeeze(testData(:,currSubj2,:));
                        currSubjIdx=testSubjIdx(currSubj2);
                        
                        [likelyPath,classLogP]=traj.viterbi(relData,HMM(currClass));
                        logP(currSubjIdx,currClass)=classLogP(likelyPath(end),end)+log(sum(this.lbls==currClass-1)/this.nSubjs);
                    end
                    this.HMMlog{currFold}=HMM;
                end
                fprintf('%d/%d\n',currFold,nFolds);
            end
            
            % Perform classification
            [~,lblsEst]=max(logP,[],2);
            lblsEst=lblsEst-1;
            BAccFun=@(Yreal,Yest)((sum((Yreal==0).*(Yest==0))/sum(Yreal==0))+(sum((Yreal==1).*(Yest==1))/sum(Yreal==1)))/2;
            BAcc=BAccFun(this.lbls,lblsEst);
            fprintf('BAcc: %0.2f\n\n',BAcc);
        end
        
        function [BAcc,coeffs]=stabilityClassification(this)
            % Generate unsupervised HMM, then classify subjects based on
            % diagonals of transition matrices (called here stability
            % values)
            nStates=max(this.subTrajLbls,[],'all');
            
            % Compute transmission probs for each subject
            HMM=struct('TR',{},'GM',{},'prior',{},'states',{});
%             stabFeats=zeros(this.nSubjs,nStates);
            stabFeats=zeros(this.nSubjs,nStates^2);
            for currSubj=1:this.nSubjs
                relGroups=this.subTrajLbls(currSubj,:);
                HMM(currSubj).TR=ones(nStates)/nStates;
                for currT=1:size(relGroups,2)-1
                    HMM(currSubj).TR(relGroups(currT),relGroups(currT+1))=HMM(currSubj).TR(relGroups(currT),relGroups(currT+1))+1;
                end
                HMM(currSubj).TR=HMM(currSubj).TR./repmat(sum(HMM(currSubj).TR),nStates,1);
%                 stabFeats(currSubj,:)=diag(HMM(currSubj).TR);
                stabFeats(currSubj,:)=reshape(HMM(currSubj).TR,1,[]);
            end
            
            % Perform crossvalidation
            pdf=histcounts(this.lbls(~isnan(this.lbls)),length(unique(this.lbls(~isnan(this.lbls)))),'Normalization','probability');
            costMat=[0,1/pdf(1);1/pdf(2),0];
%             classifierFun=@(x,y)fitcsvm(x,y,'Cost',costMat,'KernelFunction','polynomial','PolynomialOrder',2,'KernelScale','auto','BoxConstraint',1,'Standardize',true,'CrossVal','on');
            classifierFun=@(x,y)fitcsvm(x,y,'Cost',costMat,'KernelScale','auto','BoxConstraint',1,'Standardize',true,'CrossVal','on');
            svm=classifierFun(stabFeats,this.lbls);
            coeffs=[];
            for currFold=1:length(svm.Trained)
                coeffs=cat(2,coeffs,svm.Trained{currFold}.Beta);
            end
            BAcc=computeBAcc(this.lbls,svm.kfoldPredict);
        end
                
        function n=get.nROIs(this)
            n=size(this.FC{1},1);
        end
        
        function n=get.nSubjs(this)
            n=length(this.lbls);
        end
        
        function sl=get.seqLength(this)
            sl=size(this.FC{1},3);
        end
        
        function nD=get.nDims(this)
            nD=size(this.feats,3);
        end
        
        function lstmClassify(this)
            % Compute Euclidean projections of FC matrices
            projs=cell(length(this.FC),1);
            for currSubj=1:length(projs)
                projs{currSubj}=zeros(this.nROIs*(this.nROIs+1)/2,size(this.FC{currSubj},3));
                for currSample=1:size(this.FC{currSubj},3)
                    projs{currSubj}(:,currSample)=real(this.RS.project(squeeze(this.FC{currSubj}(:,:,currSample))));
                end
            end
            
            % Fix data
            dataLbls=categorical(this.lbls);
            
            % Split data into folds
            nReps=1;
            nFolds=5;
            foldID=ceil((1:length(dataLbls))/(length(dataLbls))*nFolds);
            foldID=foldID(randperm(length(foldID))); 
            
            % Allocate variables
            lblsEst=zeros(size(dataLbls));
            scores=zeros(length(dataLbls),2);
            trndNet=cell(nFolds,1);
            valBAcc=zeros(nFolds,nReps);
            testBAcc=zeros(nFolds,nReps);
            
            % Perform n-fold training
            for currFold=1:nFolds
                testIdx=find(foldID==currFold);
                trainIdx=find(foldID~=currFold);
                valIdx=trainIdx(1:3:end);
                trainIdx=setdiff(trainIdx,valIdx);
                
                for currRep=1:nReps
                    
                    % Define training options for clustering network
                    batchSize=5;
                    clusterLayers=load('clusterLayers.mat','clusterLayers');
                    clusterLayers=clusterLayers.clusterLayers;
                    
                    clusterLayers(strcmpi({clusterLayers.Name},'regressionoutput'))=clusterLayer;
                    
%                     layers=clusterLayers.Layers;
%                     customLayerN=find(strcmpi({layers.Name},'regressionoutput'));
%                     layers(customLayerN)=clusterLayer;
%                     layers(customLayerN).Name='regressionoutput';
%                     clusterLayers=createLgraphUsingConnections(layers,clusterLayers.Connections);

                    options = trainingOptions('adam', ...
                        'ExecutionEnvironment','gpu', ...
                        'MaxEpochs',1500, ...
                        'MiniBatchSize',batchSize, ...
                        'GradientThreshold',1, ...
                        'Verbose',false, ...
                        'Plots','training-progress', ... %training-progress
                        'Shuffle','every-epoch', ... % every-epoch
                        'InitialLearnRate',5e-4, ...
                        'ValidationData',{projs(valIdx),projs(valIdx)}, ...
                        'ValidationPatience',2, ...
                        'ValidationFrequency',1, ...
                        'LearnRateSchedule','piecewise', ...
                        'LearnRateDropPeriod',1, ...
                        'LearnRateDropFactor',0.99);
                    % Plot curves only for first fold
                    if currFold>1
                        options.Plots='none';
                    end
                    
                    % Train clustering network
                    clusterNet=trainNetwork(projs(trainIdx),projs(trainIdx),clusterLayers,options);
                    
                    % Use clustering network as starting point for following
                    % step
                    layers=clusterNet.Layers;
                    layers(4)=fullyConnectedLayer(10);
                    layers(4).Weights=clusterNet.Layers(4).Weights(1:10,:);
                    layers(4).Bias=clusterNet.Layers(4).Bias(1:10);
                    layers(4).Name=clusterNet.Layers(4).Name;
                    
                    % Define training options
                    untrndNet=load('20_12_10_clusterClassifyNet.mat.mat');
                    netLayers=untrndNet.layers_1;
%                     untrndNet=[layers(1:5);netLayers(7:10)];
                    untrndNet=[layers(1:5);softmaxLayer('Name','softmax_1');netLayers(7:10)];
                    
%                     untrndNetLayers=[layers(1:5);netLayers(7:10)];
%                     untrndNetConnections=clusterNet.Connections;
%                     untrndNetConnections(end,:)=[];
%                     for currLayer=6:length(untrndNetLayers)-1
%                         untrndNetConnections(end+1,:)={untrndNetLayers(currLayer).Name,untrndNetLayers(currLayer+1).Name}; %#ok<AGROW>
%                     end
%                     untrndNet=createLgraphUsingConnections(untrndNetLayers,untrndNetConnections);
                    
                    batchSize=150;
                    options = trainingOptions('adam', ...
                        'ExecutionEnvironment','gpu', ...
                        'MaxEpochs',1500, ...
                        'MiniBatchSize',batchSize, ...
                        'GradientThreshold',1, ...
                        'Verbose',false, ...
                        'Plots','training-progress', ... %training-progress
                        'Shuffle','every-epoch', ... % every-epoch
                        'InitialLearnRate',5e-4, ...
                        'ValidationData',{projs(valIdx),dataLbls(valIdx)}, ...
                        'ValidationFrequency',5,...
                        'ValidationPatience',2,...
                        'LearnRateSchedule','piecewise', ...
                        'LearnRateDropPeriod',50, ...
                        'LearnRateDropFactor',0.8, ...
                        'L2Regularization', 1e-2);
                    
                    % Plot curves only for first fold
                    if currFold>1
                        options.Plots='none';
                    end
                    
                    % Train network
                    trndNet{currFold}=trainNetwork(projs(trainIdx),dataLbls(trainIdx),untrndNet,options);

%                     % Define training options
%                     untrndNet=load('clusterClassifyNet.mat');
%                     untrndNet=untrndNet.layers_1;
%                     batchSize=150;
%                     options = trainingOptions('adam', ...
%                         'ExecutionEnvironment','gpu', ...
%                         'MaxEpochs',500, ...
%                         'MiniBatchSize',batchSize, ...
%                         'GradientThreshold',1, ...
%                         'Verbose',false, ...
%                         'Plots','training-progress', ... %training-progress
%                         'Shuffle','every-epoch', ... % every-epoch
%                         'InitialLearnRate',5e-4, ...
%                         'ValidationData',{projs(valIdx),dataLbls(valIdx)}, ...
%                         'ValidationFrequency',5,...
%                         'ValidationPatience',2,...
%                         'LearnRateSchedule','piecewise', ...
%                         'LearnRateDropPeriod',5, ...
%                         'LearnRateDropFactor',0.8, ...
%                         'L2Regularization', 1e-2);
%                     
%                     % Train network
%                     trndNet{currFold}=trainNetwork(projs(trainIdx),dataLbls(trainIdx),untrndNet,options);
                    
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
                    fprintf('Fold %d/%d, rep %d/%d: val BAcc %0.2f, test BAcc %0.2f\n',currFold,nFolds,currRep,nReps,currValBAcc,currTestBAcc);
                    valBAcc(currFold,currRep)=currValBAcc;
                    testBAcc(currFold,currRep)=currTestBAcc;
                end
            end
            keyboard;
        end
    end
    
    methods (Static)
        function [x,logP]=viterbi(obs,HMM)
            nStates=length(HMM.prior);
            nObs=length(obs);
            T1=zeros(nStates,nObs);
            T2=T1;
            for currState=1:nStates
                T1(currState,1)=log(HMM.prior(currState)*HMM.GM{currState}.pdf(obs(1,:)));
                T2(currState,1)=0;
            end
            for currT=2:nObs
                for currState=1:nStates
%                     currOut=T1(:,currT-1)+log(HMM.TR(currState,:)')+log(HMM.GM{currState}.pdf(obs(currT,:)));
                    currOut=T1(:,currT-1)+log(HMM.TR(:,currState))+log(HMM.GM{currState}.pdf(obs(currT,:)));
                    [T1(currState,currT),T2(currState,currT)]=max(currOut);
                end
            end
            [~,z(nObs)]=max(T1(:,end));
            x(nObs)=HMM.states(z(end));
            for currT=nObs:-1:2
                z(currT-1)=T2(z(currT),currT);
                x(currT-1)=HMM.states(z(currT-1));
            end
            logP=T1;
        end
    end
end