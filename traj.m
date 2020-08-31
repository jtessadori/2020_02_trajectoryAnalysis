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
        nNN=150;
        RS;
        D;
        subTrajLbls;
        feats;
        BAccReps;
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
        
        function computeD(this,maxDims)
            % Project all matrices in tangent space
            FCs=permute(cat(3,this.FC{:}),[3,1,2]);
            FCs=FCs+repmat(permute(eye(this.nROIs)*1e-6,[3,1,2]),size(FCs,1),1,1); % Add a small regularization term         
            projs=zeros((this.nROIs*(this.nROIs+1))/2,size(FCs,1));
            for currFC=1:size(FCs,1)
                projs(:,currFC)=this.RS.project(squeeze(FCs(currFC,:,:)));
            end
            
%             % Recover step by step changes and compute rotation matrix
%             projs3D=reshape(projs,[],this.seqLength,this.nSubjs);
%             X1=zeros(size(projs,1),(this.seqLength-1)*this.nSubjs);
%             X2=X1;
%             for currSubj=1:this.nSubjs
%                 for currT=1:this.seqLength-1
%                     X1(:,(currSubj-1)*(this.seqLength-1)+currT)=projs3D(:,currT,currSubj);
%                     X2(:,(currSubj-1)*(this.seqLength-1)+currT)=projs3D(:,currT+1,currSubj);
%                 end
%             end
%             B=(X1-X2);
%             dProjs=real(B-repmat(mean(B,2),1,size(B,2)));
%             cProjs=atan(dProjs./repmat(3*mad(dProjs,1,2),1,size(dProjs,2))).*repmat(mad(dProjs,1,2),1,size(dProjs,2))*3;
%             coeffs=pca((cProjs)');
%             
%             % Remove means from original data, rotate and add rotated means
%             meanData=mean(projs,2);
%             score=(projs-repmat(meanData,1,size(projs,2)))'*coeffs+meanData'*coeffs;

            % Compute distance matrix on reduced dimension set
            % Demean projections, compress outliers and compute PCA
            dProjs=real(projs-repmat(mean(projs,2),1,size(projs,2)));
            cProjs=atan(dProjs./repmat(3*mad(dProjs,1,2),1,size(projs,2))).*repmat(mad(dProjs,1,2),1,size(projs,2))*3;
%             cProjs=dProjs;
            [~,score]=pca(cProjs');
            
            % Compute distance matrix on smaller number of dimensions
            if nargin<2
                maxDims=3;
            end
            score=score(:,1:maxDims);
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
            this.feats=reshape(score,this.seqLength,this.nSubjs,maxDims);
        end
        
        function clusterPoints(this,simScale)
            % Cluster points
            if nargin<2
                simScale=3;
            end
            sigma=simScale*std(reshape(this.D,[],1));
            A=exp(-this.D./sigma);
            A=A.*not(eye(size(A)));
            C=dominantset(A);
            %             score=evalin('base','score');
            %             C=evalin('base','C');
            if any(C==0)
                C=C+1;
            end
            this.subTrajLbls=reshape(C,[],this.nSubjs)';
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
            
% dims=2:2:20;
% scales=[2,3,4,15,40,100,300,500,1000];
% clear BAcc
% for currDim=1:length(dims)
% trajOld.computeD(dims(currDim));
% for currScale=1:length(scales)
% trajOld.clusterPoints(scales(currScale));
% [BAcc(currDim,currScale),logP{currDim,currScale}]=trajOld.HMMclassification;
% fprintf('Dims: %d/%d, scales: %d/%d\n',currDim,length(dims),currScale,length(scales));
% end
% end
            
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
                    
                    % Compute end-state log likelihoods and correct for
                    % unbalanced datasets
                    testSubjIdx=find(CV.test(currFold));
                    for currSubj2=1:size(testData,2)
                        relData=squeeze(testData(:,currSubj2,:));
                        currSubjIdx=testSubjIdx(currSubj2);
                        
                        [likelyPath,classLogP]=traj.viterbi(relData,HMM(currClass));
                        logP(currSubjIdx,currClass)=classLogP(likelyPath(end),end)+log(sum(this.lbls==currClass-1)/this.nSubjs);

%                         pStates=traj.FB(relData,HMM(currClass));
%                         pStates=pStates./repmat(sum(pStates,1),size(pStates,1),1);
%                         logP(currSubjIdx,currClass)=sum(pStates.^2,'all');
%                         logP(currSubjIdx,currClass)=sum(log(pStates),'all')+log(sum(this.lbls==currClass-1)/this.nSubjs);
                        
%                         pObs=zeros(nStates,this.seqLength);
%                         for currState=1:nStates
%                             pObs(currState,:)=HMM(currClass).GM{currState}.pdf(relData);
%                         end
%                         logP(currSubjIdx,currClass)=log(sum(pStates.*pObs,'all'))+log(sum(this.lbls==currClass-1)/this.nSubjs);

                    end
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
        
        function BAcc=trajErrClassification(this)
            % Compute distribution stats for each class
            nStates=max(this.subTrajLbls,[],'all');
            nFolds=5;
            CV=cvpartition(length(this.lbls),'KFold',nFolds);
            trajErr=zeros(this.nSubjs,2);
            for currFold=1:nFolds
                trainData=this.feats(:,CV.training(currFold),:);
                trainChange=cat(1,diff(trainData),nan(1,size(trainData,2),size(trainData,3)));
                trainGroups=this.subTrajLbls(CV.training(currFold),:);
                trainLbls=this.lbls(CV.training(currFold));
                testData=this.feats(:,CV.test(currFold),:);
                
                classModels=struct('GM',{},'meanChange',{});
                for currClass=1:2
                    % Recover training cluster lbls only for current class
                    relData=reshape(permute(trainData(:,trainLbls==currClass-1,:),[2,1,3]),[],size(trainData,3));
                    relChange=reshape(permute(trainChange(:,trainLbls==currClass-1,:),[2,1,3]),[],size(trainData,3));
                    relGroups=reshape(trainGroups(trainLbls==currClass-1,:),[],1);
                    
                    % Compute gaussian models and mean changes
                    for currGroup=1:nStates
                        currData=relData(relGroups==currGroup,:);
                        currChange=relChange(relGroups==currGroup,:);
                        if size(currData,1)>this.nDims
                            classModels(currClass).GM{currGroup}=fitgmdist(currData,1);
                        elseif size(currData,1)>1
                            classModels(currClass).GM{currGroup}=gmdistribution(mean(currData,1),std(currData,[],1));
                        elseif size(currData,1)==1
                            classModels(currClass).GM{currGroup}=gmdistribution(mean(currData,1),std(relData,[],1));
                        else
                            classModels(currClass).GM{currGroup}=gmdistribution(mean(relData,1),std(relData,[],1));
                        end
                        classModels(currClass).meanChange{currGroup}=nanmean(currChange,1);
                    end
                    meanChangeMat=cat(1,classModels(currClass).meanChange{:});
                    meanChangeMat(isnan(meanChangeMat))=0;                    
                    
                    % Compute traj errs for all test trajs
                    testSubjIdx=find(CV.test(currFold));
                    for currSubj2=1:size(testData,2)
                        relData=squeeze(testData(:,currSubj2,:));
                        currSubjIdx=testSubjIdx(currSubj2);
                        GMval=zeros(this.seqLength,nStates);
                        for currT=1:length(relData)
                            for currGroup=1:nStates
                                GMval(currT,currGroup)=classModels(currClass).GM{currGroup}.pdf(relData(currT,:));
                            end
                        end
                        GMvalNorm=GMval./repmat(sum(GMval,2),1,size(GMval,2));
                        estChange=GMvalNorm*meanChangeMat;
                        trajErr(currSubjIdx,currClass)=mean((acos(diag(estChange(1:end-1,:)*diff(relData)')./(sqrt(sum(estChange(1:end-1,:).^2,2)).*sqrt(sum(diff(relData).^2,2))))));
%                         trajErr(currSubjIdx,currClass)=sum(sqrt(sum((estChange(1:end-1,:)-diff(relData)).^2,2)));
                    end
                end
                fprintf('%d/%d\n',currFold,nFolds);
            end
            
            % Perform classification
            [~,lblsEst]=max(trajErr,[],2);
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
        
        function nD=get.nDims(this)
            nD=size(this.feats,3);
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
        end
        
        function [timeData,lbls]=recoverTimeData(xlsFile,dataFolder)
            % Recover patient data from xls file
            [~,~,RAW]=xlsread(xlsFile);
            f=RAW(1,:);
            f(cellfun(@(x)~ischar(x),f))=[];
            s=cell2struct(RAW(2:end,1:length(f)),f,2);
            
            % Use recovered data to load patients data
            lbls=zeros(size(s));
            timeData=cell(length(s),1);
            D=dir(dataFolder);
            for currSubj=1:length(s)
                for currD=3:length(D)
                    D2=dir([dataFolder,'\',D(currD).name]);
                    T=struct2table(D2);
                    subjIdx=find(cellfun(@(x)strcmp(x,sprintf('%04d',s(currSubj).IDsoggetto)),T.name));
                    if ~isempty(subjIdx)
                        subjData=dlmread(sprintf('%s\\%s\\%s\\resting_conn\\AAL116_gm_ROISignals.txt',dataFolder,D(currD).name,D2(subjIdx).name));
                        break;
                    end
                end
                timeData{currSubj}=subjData;
                
                % Recover lables
                if isempty(s(currSubj).Dis_Prog_2)
                    lbls(currSubj)=nan;
                else
                    lbls(currSubj)=s(currSubj).Dis_Prog_2;
                end
                fprintf('%d/%d\n',currSubj,length(s));
            end
        end
        
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
        
        function alpha=forward(obs,HMM)
            % Forward
            nStates=length(HMM.prior);
            nObs=size(obs,1);
            alpha=zeros(nStates,nObs);
            for currState=1:nStates
                alpha(currState,1)=HMM.prior(currState)*HMM.GM{currState}.pdf(obs(1,:));
            end
            for currT=2:nObs
                for currState=1:nStates
                    alpha(currState,currT)=HMM.GM{currState}.pdf(obs(currT,:))*sum(HMM.TR(currState,:).*alpha(:,currT-1)');
                end
            end
%             alpha=zeros(nStates,nObs);
%             for currState=1:nStates
%                 alpha(currState,1)=HMM.prior(currState)*HMM.GM{currState}.pdf(obs(1,:));
%             end
%             alpha(:,1)=alpha(:,1)./sum(alpha(:,1));
%             for currT=2:nObs
%                 for currState=1:nStates
%                     alpha(currState,currT)=HMM.GM{currState}.pdf(obs(currT,:))*sum(HMM.TR(currState,:).*alpha(:,currT-1)');
%                 end
%                 alpha(:,currT)=alpha(:,currT)./sum(alpha(:,currT));
%             end
        end
        
        function pStates=FB(obs,HMM)
            % Forward
            nStates=length(HMM.prior);
            nObs=size(obs,1);
            alpha=zeros(nStates,nObs);
            scaling=zeros(1,nObs);
            for currState=1:nStates
                alpha(currState,1)=HMM.prior(currState)*HMM.GM{currState}.pdf(obs(1,:));
            end
            for currT=2:nObs
                for currState=1:nStates
                    alpha(currState,currT)=HMM.GM{currState}.pdf(obs(currT,:))*sum(HMM.TR(currState,:).*alpha(:,currT-1)');
                end
                scaling(currT)=sum(alpha(:,currT));
                alpha(:,currT)=alpha(:,currT)./scaling(currT);
            end
            
            % Backward
            beta=ones(nStates,nObs);
            for currT=nObs-1:-1:1
                for currState=1:nStates
                    beta(currState,currT)=1/scaling(currT+1)*sum(HMM.TR(:,currState).*beta(:,currT+1).*HMM.GM{currState}.pdf(obs(currT+1,:)));
                end
            end
            
            % Conclude
            pStates=alpha.*beta;
        end
    end
end