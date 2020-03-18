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
        seqLength;
        nNN=150;
        RS;
        projs;
        D;
    end
    
    methods
        function this=traj(path)
            % Constructor for traj class.
            % Only argument (required) is absolut path of data file.
            this.dataPath=path;
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
        
        function loadDataImprovement(this)
            % Recovers dynamical functional connectivity matrices as well
            % as lbls from the file specified as dataPath. Task is
            % classification between stable/improving subjects and
            % worsening ones
            inData=load(this.dataPath);
            
            this.lbls=inData.lbls;
            this.FC=inData.FCs;
        end
        
        function computeD(this)
            % Warn user that this will take several minutes
            fprintf('Computing distance matrix in tangent space. This will take between several minutes and several hours\n');
            
            % Compute Riemann average of points
            FCs=permute(cat(3,this.FC{:}),[3,1,2]);
            FCs=FCs+repmat(permute(eye(this.nROIs)*1e-6,[3,1,2]),size(FCs,1),1,1); % Add a small regularization term
            this.RS=riemannSpace(FCs,'cov');
            
            % Project all matrices in tangent space
            this.projs=zeros((this.nROIs*(this.nROIs+1))/2,size(FCs,1));
            for currFC=1:size(FCs,1)
                this.projs(:,currFC)=this.RS.project(squeeze(FCs(currFC,:,:)));
            end
            
            % Compute distance matrix
            this.D=zeros(size(this.projs,2));
            distFun=@(x1,x2)sqrt(sum((x1-x2).^2));
            for currFC1=1:size(this.projs,2)
                for currFC2=currFC1+1:size(this.projs,2)
                    this.D(currFC1,currFC2)=distFun(this.projs(:,currFC1),this.projs(:,currFC2));
                    this.D(currFC2,currFC1)=this.D(currFC1,currFC2);
                end
                fprintf('%d/%d\n',currFC1,size(this.projs,2));
            end
        end
        
        function BAcc=classifyTrajs(this)
            % Compare each point with its n nearest neighbors (nNN) and
            % give it a score as average of the labels of the NNs.
            pointScore=zeros(this.nSubjs,this.seqLength);
            subjLbls=floor((0:size(this.D,1)-1)/this.seqLength)+1;
            lblsLong=reshape(repmat(this.lbls,1,this.seqLength)',[],1);
            for currSubj=1:this.nSubjs
                for currT=1:this.seqLength
                    relData=this.D((currSubj-1)*this.seqLength+currT,:);
                    relData(subjLbls==currSubj)=Inf;
                    [~,NNidx]=sort(relData,'ascend');
                    NNidx=NNidx(1:this.nNN);
                    pointScore(currSubj,currT)=nanmean(lblsLong(NNidx));

%                     coeffs=pinv(this.projs(:,NNidx)')*lblsLong(NNidx);
%                     pointScore(currSubj,currT)=this.projs(:,(currSubj-1)*this.seqLength+currT)'*coeffs;
%                     mdl=fitglm(this.projs(:,NNidx)',lblsLong(NNidx),'Distribution','binomial','Link','logit');
%                     pointScore(currSubj,currT)=mdl.predict(this.projs(:,(currSubj-1)*this.seqLength+currT)');
                end
                fprintf('%d/%d\n',currSubj,length(this.lbls));
            end
            
            % Determine where to put a threshold to attribute classes. This
            % is potentially problematic, as the two classe might be
            % strongly imbalanced
            lblsEst=zeros(size(this.lbls));
            BAccFun=@(Yreal,Yest)((sum((Yreal==0).*(Yest==0))/sum(Yreal==0))+(sum((Yreal==1).*(Yest==1))/sum(Yreal==1)))/2;
            for currSubj=1:this.nSubjs
                [~,~,T]=perfcurve(this.lbls([1:currSubj-1,currSubj+1:end]),mean(pointScore([1:currSubj-1,currSubj+1:end],:),2),1);
                BAccs=zeros(size(T));
                for currT=1:length(T)
                    BAccs(currT)=BAccFun(this.lbls([1:currSubj-1,currSubj+1:end]),mean(pointScore([1:currSubj-1,currSubj+1:end],:),2)>T(currT));
                end
                Tbest=unique(T(BAccs==max(BAccs)));
                lblsEst(currSubj)=nanmean(pointScore(currSubj,:),2)>Tbest;
            end
            BAcc=BAccFun(this.lbls,lblsEst);
        end
        
        function BAcc=HMMclassification(this)
%             % Demean projections, compress outliers and compute PCA
%             dProjs=this.projs-repmat(mean(this.projs,2),1,size(this.projs,2));
%             cProjs=atan(dProjs./repmat(3*mad(dProjs,1,2),1,size(this.projs,2))).*repmat(mad(dProjs,1,2),1,size(this.projs,2))*3;
%             [~,score]=pca(cProjs');
%             
%             % Compute distance matrix on smaller number of dimensions
            nDims=3;
%             distFun=@(x1,x2)sqrt(sum((x1-x2).^2));
%             DlowD=zeros(size(score,1));
%             for currX1=1:size(score,1)
%                 for currX2=currX1+1:size(score,1)
%                     DlowD(currX1,currX2)=distFun(score(currX1,1:nDims),score(currX2,1:nDims));
%                     DlowD(currX2,currX1)=DlowD(currX1,currX2);
%                 end
%                 if mod(currX1,100)==0
%                     fprintf('%d/%d\n',currX1,size(score,1));
%                 end
%             end
%             
%             % Cluster points
%             sigma=3*std(reshape(DlowD,[],1));
%             A=exp(-DlowD./sigma);
%             A=A.*not(eye(size(A)));
%             C=dominantset(A);
            score=evalin('base','score');
            C=evalin('base','C');
            if any(C==0)
                C=C+1;
            end
            subTrajLbls=reshape(C,[],this.nSubjs)';
            
            % WARNING: the following block overwrites actual data with
            % phantom data
            trainData=subTrajLbls;
            trainLbls=this.lbls;
            HMM=struct('TR',{rand(2^nDims),rand(2^nDims)},'E',{rand(2^nDims,max(C)+1),rand(2^nDims,max(C))});
            for currClass=1:2
                % Recover training cluster lbls only for current class
                relData=trainData(trainLbls==currClass-1,:);
                % Compute distribution stats for each subtraj
                [HMM(currClass).TR,HMM(currClass).E]=hmmtrain(relData,rand(2^nDims),rand(2^nDims,max(C)));
                % Define prior distribution and possible states
                HMM(currClass).states=1:max(C);
                HMM(currClass).prior=ones(2^nDims,1)/2^nDims;
            end
            for currSubj=1:this.nSubjs
                [phSeq(currSubj,:),phStates(currSubj,:)]=hmmgenerate(this.seqLength*100,HMM(this.lbls(currSubj)+1).TR,HMM(this.lbls(currSubj)+1).E);
            end
            subTrajLbls=phSeq;
            
            % Compute transition matrix, prior and distribution stats for
            % each class
            nFolds=5;
            CV=cvpartition(length(this.lbls),'KFold',nFolds);
            logP=zeros(size(subTrajLbls,1),2);
            for currFold=1:nFolds
                trainData=subTrajLbls(CV.training(currFold),:);
                trainLbls=this.lbls(CV.training(currFold));
                testData=subTrajLbls(CV.test(currFold),:);
                
                HMM=struct('TR',{rand(2^nDims),rand(2^nDims)},'E',{rand(2^nDims,max(C)+1),rand(2^nDims,max(C))});
                for currClass=1:2
                    % Recover training cluster lbls only for current class
                    relData=trainData(trainLbls==currClass-1,:);
                    
                    % Compute distribution stats for each subtraj
                    [HMM(currClass).TR,HMM(currClass).E]=hmmtrain(relData,rand(2^nDims),rand(2^nDims,max(C)));
                    
                    % Define prior distribution and possible states
                    HMM(currClass).states=1:max(C);
                    HMM(currClass).prior=ones(2^nDims,1)/2^nDims;
                    
                    % Compute end-state log likelihoods and correct for
                    % unbalanced datasets
                    testSubjIdx=find(CV.test(currFold));
                    for currSubj2=1:size(testData,1)
                        relData=testData(currSubj2,:);
                        currSubjIdx=testSubjIdx(currSubj2);
%                         [likelyPath(currSubjIdx,:),endLogP]=traj.viterbi(relData,HMM(currClass));
%                         logP(currSubjIdx,currClass)=max(endLogP)+log(sum(this.lbls==currClass-1)/this.nSubjs);
                        endLogP=log(traj.forward(relData,HMM(currClass)));
                        logP(currSubjIdx,currClass)=max(endLogP)+log(sum(this.lbls==currClass-1)/this.nSubjs);
                    end
                end
                fprintf('%d/%d\n',currFold,nFolds);
            end
            
            % Perform classification
            [~,lblsEst]=max(logP,[],2);
            lblsEst=lblsEst-1;
            BAccFun=@(Yreal,Yest)((sum((Yreal==0).*(Yest==0))/sum(Yreal==0))+(sum((Yreal==1).*(Yest==1))/sum(Yreal==1)))/2;
            BAcc=BAccFun(this.lbls,lblsEst);
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
    end
    
    methods (Static)
        function X=dist2coords(D)
            D2=D.^2;
            
            M=zeros(size(D));
            for currSlice1=1:size(D,1)
                for currSlice2=1:size(D,2)
                    M(currSlice1,currSlice2)=(D2(1,currSlice2)+D2(currSlice1,1)-D2(currSlice1,currSlice2))/2;
                end
            end
            [u,s]=svd(M);
            X=u*sqrt(s);
        end
        
        function prepareDataFile(fileName,xlsFile,dataFolder,slicesPerMat,stepSize)
            % Recover patient data from xls file
            [~,~,RAW]=xlsread(xlsFile);
            f=RAW(1,:);
            f(cellfun(@(x)~ischar(x),f))=[];
            s=cell2struct(RAW(2:end,1:length(f)),f,2);
            
            % Use recovered data to load patients data
            lbls=zeros(size(s));
            FCs=cell(length(s),1);
            D=dir(dataFolder);
            for currSubj=1:length(s)
                for currD=3:length(D)
                    D2=dir([dataFolder,'\',D(currD).name]);
                    T=struct2table(D2);
                    subjIdx=find(cellfun(@(x)strcmp(x,sprintf('%04d',s(currSubj).IDsoggetto)),T.name));
                    if ~isempty(subjIdx)
                        subjData=dlmread(sprintf('%s\\%s\\%s\\resting_conn\\AAL116_gm_ROISignals.txt',dataFolder,D(currD).name,D2(subjIdx).name));
                        
                        % Occasionally, one region may be missing from a
                        % subject. Put white noise there
                        if sum(sum(isnan(subjData)))
                            subjData(isnan(subjData))=randn(size(subjData(isnan(subjData))));
                            fprintf('ROI missing data, substituting with white noise\n');
                        end
                        
                        % Compute covariance matrices
                        nSteps=(size(subjData,1)-slicesPerMat)/stepSize+1;
                        if ~isfinite(nSteps)
                            nSteps=1;
                        end
                        subjCorr=zeros(size(subjData,2),size(subjData,2),nSteps);
                        for currSlice=1:nSteps
                            subjCorr(:,:,currSlice)=cov(subjData((currSlice-1)*stepSize+1:(currSlice-1)*stepSize+slicesPerMat,:));
                        end
                        break;
                    end
                end
                FCs{currSubj}=subjCorr;
                
                % Recover lables
                if isempty(s(currSubj).Dis_Prog_2)
                    lbls(currSubj)=nan;
                else
                    lbls(currSubj)=s(currSubj).Dis_Prog_2;
                end
                fprintf('%d/%d\n',currSubj,length(s));
            end
            fprintf('Saving results...\n');
            save(fileName,'FCs','-v7.3');
            save(fileName,'lbls','-append');
            save(fileName,'xlsFile','-append');
            save(fileName,'dataFolder','-append');
            save(fileName,'slicesPerMat','-append');
            save(fileName,'stepSize','-append');
            save(fileName,'xlsFile','-append');
        end
        
        function [x,logP]=viterbi(obs,HMM)
            nStates=length(HMM.prior);
            nObs=length(obs);
            T1=zeros(nStates,nObs);
            T2=T1;
            for currState=1:nStates
                T1(currState,1)=log(HMM.prior(currState)*HMM.E(currState,obs(1)));
                T2(currState,1)=0;
            end
            for currState=1:nStates
                for currT=2:nObs
                    currOut=T1(:,currT-1)+log(HMM.TR(currState,:)')+log(HMM.E(currState,obs(currT)));
                    [T1(currState,currT),T2(currState,currT)]=max(currOut);
                end
            end
            [~,z(nObs)]=max(T1(:,end));
            x(nObs)=HMM.states(z(end));
            for currT=nObs:-1:2
                z(currT-1)=T2(z(currT),currT);
                x(currT-1)=HMM.states(z(currT-1));
            end
            logP=T1(:,end);
        end
        
        function alpha=forward(obs,HMM)
            % Forward
            nStates=length(HMM.prior);
            nObs=size(obs,1);
            alpha=zeros(nStates,nObs);
            for currState=1:nStates
                alpha(currState,1)=HMM.prior(currState)*HMM.E(currState,obs(1));
                for currT=2:nObs
                    alpha(currState,currT)=HMM.E(currState,obs(currT))*sum(HMM.transMat(currState,:).*alpha(:,currT-1)');
                end
            end
        end
    end
end