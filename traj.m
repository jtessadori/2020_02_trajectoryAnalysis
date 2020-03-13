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
            
            % Compute distance matrix on smaller number of dimensions
            nDims=6;
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
            subTrajLbls=reshape(C,[],this.nSubjs)';
            
%             %% WARNING: generating phantom with the next block of code
%             C1=histcounts(subTrajLbls(this.lbls==0,:),'BinMethod','integer');
%             C2=histcounts(subTrajLbls(this.lbls==1,:),'BinMethod','integer');
%             score(ismember(C,find(C1>C2)),nDims/2+1:nDims)=score(ismember(C,find(C1>C2)),1:nDims/2);
%             score(ismember(C,find(C1<=C2)),nDims/2+1:nDims)=-score(ismember(C,find(C1<=C2)),1:nDims/2);
%             score(:,nDims/2+1:nDims)=score(:,nDims/2+1:nDims)+randn(size(score(:,nDims/2+1:nDims)))*.01.*repmat(std(score(:,nDims/2+1:nDims)),size(score,1),1);
            
            % Recover first dimensions and compute GM model for each
            % subTraj
            GM=cell(max(C),1);
            lblVals=1:max(C);
            for currSubtraj=1:length(lblVals)
                relData=score(C==lblVals(currSubtraj),1:nDims);
                if ~isempty(relData)
                    GM{lblVals(currSubtraj)}=fitgmdist(relData,1);
                end
            end
            
            % Compute transition matrix, prior and distribution stats for
            % each class
            nFolds=this.nSubjs;
            CV=cvpartition(length(this.lbls),'KFold',nFolds);
            feats=mat2cell(score(:,1:nDims),ones(this.nSubjs,1)*this.seqLength,nDims);
            logP=zeros(size(feats,1),2);
            for currFold=1:nFolds
                trainData=feats(CV.training(currFold));
                trainClusterLbls=subTrajLbls(CV.training(currFold),:);
                trainLbls=this.lbls(CV.training(currFold));
                testData=feats(CV.test(currFold));
                
                for currClass=1:2
                    % Recover training cluster lbls only for current class
                    currClusterLbls=trainClusterLbls(trainLbls==currClass-1,:);
                    relData=cat(3,trainData{trainLbls==currClass-1});
                    relData=permute(relData,[3,1,2]);
                    lbls1D=reshape(currClusterLbls,[],1);
                    data1D=reshape(relData,[],size(relData,3));
                    
                    % Compute distribution stats for each subtraj
                    for currSubtraj=1:length(lblVals)
                        if sum(lbls1D==currSubtraj)>nDims
                            HMM(currClass).GM{currSubtraj}=fitgmdist(data1D(lbls1D==currSubtraj,:),1); %#ok<AGROW>
                        elseif sum(lbls1D==currSubtraj)>1
                            % Assume a diagonal covariance matrix if number
                            % of samples for current label is too low
                            HMM(currClass).GM{currSubtraj}=gmdistribution(mean(data1D(lbls1D==currSubtraj,:),1),diag(std(data1D(lbls1D==currSubtraj,:),[],1))); %#ok<AGROW>
                        end
                    end
                    HMM(currClass).states=1:max(lblVals);
                    
                    % Compute prior probability of each state
                    HMM(currClass).prior=histcounts(currClusterLbls(:,1),.5:max(lblVals)+.5);
                    HMM(currClass).prior=HMM(currClass).prior+min(HMM(currClass).prior(HMM(currClass).prior>0));
                    HMM(currClass).prior=HMM(currClass).prior/sum(HMM(currClass).prior);
                    
                    % Compute transition matrix
                    HMM(currClass).transMat=zeros(length(lblVals));
                    for currSubj2=1:size(currClusterLbls,1)
                        for currT=2:size(currClusterLbls,2)
                            currState=find(lblVals==trainClusterLbls(currSubj2,currT));
                            prevState=find(lblVals==trainClusterLbls(currSubj2,currT-1));
                            HMM(currClass).transMat(currState,prevState)=HMM(currClass).transMat(currState,prevState)+1;
                        end
                    end
                    HMM(currClass).transMat=HMM(currClass).transMat./repmat(sum(HMM(currClass).transMat),length(lblVals),1);
                    HMM(currClass).transMat(isnan(HMM(currClass).transMat))=0;
                    
                    % Remove from transition matrix states with no
                    % associated GM
                    for currLbl=1:length(HMM(currClass).GM)
                        if isempty(HMM(currClass).GM{currLbl})
                            HMM(currClass).transMat(currLbl,:)=0;
                            HMM(currClass).transMat(:,currLbl)=0;
                        end
                    end
                    
                    % Compute end-state log likelihoods
                    for currSubj2=1:length(testData)
                        relData=testData(currSubj2);
                        relData=squeeze(cat(2,relData{:}));
                        [~,endLogP]=traj.viterbi(relData,HMM(currClass));
                        logP(find(CV.test(currFold),currSubj2,'first'),currClass)=max(endLogP);
                    end
                end
                fprintf('%d/%d\n',currFold,nFolds);
            end
            
            % Perform classification
            [~,lblsEst]=max(logP,[],2);
            lblsEst=lblsEst-1;
            BAccFun=@(Yreal,Yest)((sum((Yreal==0).*(Yest==0))/sum(Yreal==0))+(sum((Yreal==1).*(Yest==1))/sum(Yreal==1)))/2;
            BAcc=BAccFun(this.lbls,lblsEst);
            
            
%             % Compute curvature on first two PCA dimensions
%             [~,score]=pca(this.projs);
%             score=real(score);
%             curvature=zeros(this.nSubjs,this.seqLength);
%             for currSubj=1:this.nSubjs
%                 traj2D=score((currSubj-1)*this.seqLength+1:currSubj*this.seqLength,1:2);
%                 x=traj2D(:,1);
%                 y=traj2D(:,2);
%                 x1=[x(2)-x(1);diff(x)];
%                 x2=[x1(2)-x1(1);diff(x1)];
%                 y1=[y(2)-y(1);diff(y)];
%                 y2=[y1(2)-y1(1);diff(y1)];
%                 curvature(currSubj,:)=(x1.*y2-y1.*x2)./(x1.^2+y1.^2).^(1.5);
%             end
%             
%             % Trajectory segmentation
%             nSamples=3;
%             trajBreak=cell(size(curvature,1),1);
%             trajBreakMat=zeros(size(curvature));
%             for currSubj=1:size(curvature,1)
%                 [~,trajBreak{currSubj}]=findpeaks(abs(curvature(currSubj,:)),'MinPeakDistance',nSamples);
%                 trajBreakMat(currSubj,trajBreak{currSubj})=1;
%                 trajBreakMat(currSubj,1)=1;
%             end
%             subTrajLbls=reshape(cumsum(reshape(trajBreakMat',[],1)),size(curvature,2),size(curvature,1))';
%             
%             % Recover first dimensions and compute GM model for each
%             % subTraj
%             nDims=3;
%             feats=score(1:nDims,:);
%             feats=reshape(feats,size(feats,1),this.nSubjs,[]);
%             feats=permute(feats,[2,3,1]);
%             feats=num2cell(feats,3);
%             GM=cell(max(max(subTrajLbls)),1);
%             for currSubtraj=1:max(max(subTrajLbls))
%                 relData=feats(subTrajLbls==currSubtraj);
%                 relData=squeeze(cat(2,relData{:}));
%                 if size(relData,1)>nDims
%                     GM{currSubtraj}=fitgmdist(relData,1);
%                 else
%                     GM{currSubtraj}=GM{currSubtraj-1}; % This will cause subtrajectories too small to be evaluated to be lumped with the previous one
%                 end
%             end
%             
%             % "Collapse" subtrajs: cluster params
%             subtrajFeats=cellfun(@(x)[x.mu,diag(x.Sigma)'],GM,'UniformOutput',false);
%             subtrajFeats=cat(1,subtrajFeats{:});
%             Z=linkage(subtrajFeats,'weighted','seuclidean');
%             C=cluster(Z,'maxclust',40);
%             newSubTrajLbls=subTrajLbls;
%             for currSubtraj=1:max(max(subTrajLbls))
%                 newSubTrajLbls(subTrajLbls==currSubtraj)=C(currSubtraj);
%             end
%             subTrajLbls=newSubTrajLbls;
%             clear newSubTrajLbls
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
        
        function alpha=forward(obs,HMM)
            % Forward
            nStates=length(HMM.prior);
            nObs=size(obs,1);
            alpha=zeros(nStates,nObs);
            for currState=1:nStates
                alpha(currState,1)=HMM.prior(currState)*normpdf(obs(1),HMM.mu(currState),HMM.sigma(currState));
                for currT=2:nObs
                    alpha(currState,currT)=normpdf(obs(currT),HMM.mu(currState),HMM.sigma(currState))*sum(HMM.transMat(currState,:).*alpha(:,currT-1)');
                end
            end
        end
        
        function beta=backward(obs,HMM)
            % Backward
            nStates=length(HMM.prior);
            nObs=size(obs,1);
            beta=zeros(nStates,nObs);
            for currState=1:nStates
                beta(currState,end)=1/nStates*normpdf(obs(end),HMM.mu(currState),HMM.sigma(currState));
                for currT=nObs-1:-1:1
                    beta(currState,currT)=normpdf(obs(currT),HMM.mu(currState),HMM.sigma(currState))*sum(HMM.transMat(:,currState)'.*beta(:,currT+1)');
                end
            end
        end
        
        function [x,logP]=viterbi(obs,HMM)
            missingStates=find(cellfun(@(x)isempty(x),HMM.GM));
            nStates=length(HMM.prior);
            nObs=size(obs,1);
            T1=zeros(nStates,nObs);
            T2=T1;
            for currState=1:nStates
                if ~isempty(HMM.GM{currState})
                    T1(currState,1)=log(HMM.prior(currState)*HMM.GM{currState}.pdf(obs(1,:)));
                    T2(currState,1)=0;
                else
                    T1(currState,1)=-Inf;
                end
            end
            for currState=1:nStates
                if ~isempty(HMM.GM{currState})
                    for currT=2:nObs
                        currOut=T1(:,currT-1)+log(HMM.transMat(currState,:)')+log(HMM.GM{currState}.pdf(obs(currT,:)));
                        [T1(currState,currT),T2(currState,currT)]=max(currOut);
                    end
                else
                    T1(currState,2:end)=-Inf;
                end
            end
            [~,z(nObs)]=max(T1(:,end));
            x(nObs)=HMM.states(z(end));
            for currT=nObs:-1:2
                z(currT-1)=T2(z(currT),currT);
                if ismember(z(currT-1),missingStates)
                    z(currT-1)=z(currT);
                end
                try
                x(currT-1)=HMM.states(z(currT-1));
                catch
                    keybord;
                end
            end
            logP=T1(:,end);
        end
        
        function HMM=BW(obs,HMM)
            % Compute forward and backward probabilities
            alpha=traj.forward(obs,HMM);
            beta=traj.backward(obs,HMM);
            gamma=(alpha.*beta)./repmat(sum(alpha.*beta),size(alpha,1),1);
            
            % Compute epsilon
            nStates=length(HMM.prior);
            nObs=size(obs,1);
            epsilon=zeros(nStates,nStates,nObs-1);
            for currState1=1:nStates
                for currState2=1:nStates
                    for currT=1:nObs-1
                        epsilon(currState1,currState2,currT)=alpha(currState1,currT)*HMM.transMat(currState1,currState2)*beta(currState2,currT+1)*normpdf(obs(currT+1),HMM.mu(currState2),HMM.sigma(currState2));
                        epsilon(currState1,currState2,currT)=epsilon(currState1,currState2,currT)./sum(alpha(:,currT).*beta(:,currT));
                    end
                end
            end
            HMM.prior=gamma(:,1);
            for currState1=1:nStates
                for currState2=1:nStates
                    HMM.transMat(currState1,currState2)=sum(epsilon(currState1,currState2,1:end-1))/sum(gamma(currState1,1:end-1));
                end
            end
        end
    end
end