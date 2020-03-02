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
            distFun=@(x1,x2)log(sqrt(sum((x1-x2).^2)));
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
        
        function CFDclassification(this)
            % First, compute centroid function distance
            cellProjs=mat2cell(this.projs,size(this.projs,1),ones(this.nSubjs,1)*this.seqLength);
            CFD=cellfun(@(x)normalize(sqrt(sum((x-repmat(mean(x,2),1,size(x,2))).^2))),cellProjs,'UniformOutput',0);
            CFD=real(cat(1,CFD{:}));
            
            % Compute curvature on first two PCA dimensions
            [~,score]=pca(this.projs);
            score=real(score);
            curvature=zeros(this.nSubjs,this.seqLength);
            for currSubj=1:this.nSubjs
                traj2D=score((currSubj-1)*this.seqLength+1:currSubj*this.seqLength,1:2);
                x=traj2D(:,1);
                y=traj2D(:,2);
                x1=[x(2)-x(1);diff(x)];
                x2=[x1(2)-x1(1);diff(x1)];
                y1=[y(2)-y(1);diff(y)];
                y2=[y1(2)-y1(1);diff(y1)];
                curvature(currSubj,:)=(x1.*y2-y1.*x2)./(x1.^2+y1.^2).^(1.5);
            end
            
            % Trajectory segmentation
            nSamples=10;
            trajBreak=cell(size(curvature,1),1);
            trajBreakMat=zeros(size(curvature));
            d=zeros(size(curvature));
            for currSubj=1:size(curvature,1)
                for currT=1:size(curvature,2)-2*nSamples
                    x=curvature(currSubj,currT:currT+2*nSamples-1);
                    mu1=mean(x(1:nSamples));
                    mu2=mean(x(nSamples+1:end));
                    mu3=mean(x);
                    sigma1=std(x(1:nSamples));
                    sigma2=std(x(nSamples+1:end));
                    sigma3=std(x);
                    L0=1/sqrt(2*pi)*sigma3*exp(-(x-mu3).^2/(2*sigma3^2));
                    L1=1/sqrt(2*pi)*sigma1*sigma2*exp(-((x-mu1).^2/(2*sigma1^2)+(x-mu2).^2/(2*sigma2^2)));
                    d(currSubj,currT+nSamples)=-log(L0/L1);
                end
                [~,trajBreak{currSubj}]=findpeaks(d(currSubj,:),'MinPeakDistance',10,'MinPeakProminence',.10*max(d(currSubj,:)));
                trajBreakMat(currSubj,trajBreak{currSubj})=1;
                trajBreakMat(currSubj,1)=1;
            end
            
            % For each subject and each class, compute HMM models (Matlab
            % (still) has not implemented support for continuous-emission
            % HMMs, so will discretize data and use discrete emissions)
            for currSubj=1:size(CFD,1)
                for currClass=1:2
                    % Only consider data for current class, excluding
                    % current subject if relevant
                    relDataIdx=this.lbls==(currClass-1);
                    relDataIdx(currSubj)=0;
                    
                    % Estimate starting point for HMM parameters
                    stateData=reshape(cumsum(reshape(trajBreakMat(relDataIdx,:)',[],1))',[],size(CFD,2));
                    emitData=CFD(relDataIdx,:);
                    emitData=permute(emitData,[3 2 1]);
                    stateData=permute(stateData,[3 2 1]);
                    mu1=zeros(1,max(max(stateData)));
                    sigma1=zeros(1,1,max(max(stateData)));
                    for currSubTraj=1:max(max(stateData))
                        mu1(currSubTraj)=mean(emitData(stateData==currSubTraj));
                        sigma1(currSubTraj)=std(emitData(stateData==currSubTraj));
                    end
                    transmat1=eye(max(max(stateData)));
                    transmat1(max(max(stateData))+1:max(max(stateData))+1:end)=1/max(max(stateData));
                    transmat1=transmat1./repmat(sum(transmat1),max(max(stateData)),1);
                    prior1=histcounts(stateData,'Normalization','pdf','BinMethod','integers');
                    mixmat1=mk_stochastic(rand(length(unique(stateData)),1));
                    
                        
%                     [mu1,sigma1]=mixgauss_init(length(unique(stateData)),emitData,'full');
%                     mu1=reshape(mu1,[1 length(unique(stateData)) 1]);
%                     mixmat1=mk_stochastic(rand(length(unique(stateData)),1));
%                     prior1=histcounts(stateData,'Normalization','pdf','BinMethod','integers');
%                     transmat1=mk_stochastic(rand(length(unique(stateData))));
                    
                    % Start EM with annealing (no idea if this is correct)
                    th=1e-2;
                    T=1e5;
                    counts=0;
                    while true
                        % Perform one step of EM
                        transmatOld=transmat1;
                        [~,prior1,transmat1,mu1,sigma1,mixmat1]=mhmm_em(emitData,prior1,transmat1,mu1,sigma1,mixmat1,'max_iter',1);
                        
                        % Evaluate change
                        currChange=max(max(abs(transmatOld-transmat1)));
                        
                        % Break loop if change is smaller than th for 10
                        % consecutive iterations
                        if currChange<th
                            counts=counts+1;
                            if counts==2
                                break
                            end
                        else
                            counts=0;
                        end
                                                
                        % Perform annealing and change temperature
                        prior1=ones(size(prior1))/length(prior1)*(1-(1/T))+prior1*(1/T);
                        transmat1=ones(size(transmat1))/size(transmat1,1)*(1-(1/T))+transmat1*(1/T);
                        T=.95*T;
                        
                        fprintf('T: %0.2f, currChange: %0.3f, counts: %d\n',T,currChange,counts);
                    end
                    HMM(currClass).prior=prior1; %#ok<AGROW>
                    HMM(currClass).transmat=transmat1; %#ok<AGROW>
                    HMM(currClass).mu=mu1; %#ok<AGROW>
                    HMM(currClass).sigma=sigma1; %#ok<AGROW>
                    HMM(currClass).mixmat=mixmat1; %#ok<AGROW>
                end
            end
            
%             % For each subject, compute HMM model
%             firstSubTrajLength=cellfun(@(x)x(2)-1,trajBreak);
%             prior=cell(size(CFD,1),1);
%             mu=cell(size(CFD,1),1);
%             sigma=cell(size(CFD,1),1);
%             T=cell(size(CFD,1),1);
%             for currSubj=1:size(CFD,1)
%                 % Compute prior probability of each state (very rough, it
%                 % is just given by the normalized lengths of first states)
%                 prior{currSubj}=firstSubTrajLength([1:currSubj-1,currSubj+1:end]);
%                 prior{currSubj}=prior{currSubj}/sum(prior{currSubj});
%                 
%                 subTrajLbls=cumsum(trajBreakMat(currSubj,:));
%                 T{currSubj}=zeros(sum(trajBreakMat(currSubj,:)));
%                 for currSubTraj=1:length(trajBreak{currSubj})+1
%                     % Compute mus and sigmas
%                     mu{currSubj}(currSubTraj)=mean(CFD(currSubj,subTrajLbls==currSubTraj));
%                     sigma{currSubj}(currSubTraj)=std(CFD(currSubj,subTrajLbls==currSubTraj));
%                     
%                     % Compute transition probabilities
%                     if currSubTraj~=length(trajBreak{currSubj})+1
%                         T{currSubj}(currSubTraj,currSubTraj)=sum(subTrajLbls==currSubTraj);
%                         T{currSubj}(currSubTraj,currSubTraj+1)=1;
%                         T{currSubj}(currSubTraj,:)=T{currSubj}(currSubTraj,:)/sum(T{currSubj}(currSubTraj,:));
%                     else
%                         T{currSubj}(end,end)=1;
%                     end
%                 end
%             end
            
            % For each subject and each class, compute HMM models (Matlab
            % (still) has not implemented support for continuous-emission
            % HMMs, so will discretize data and use discrete emissions)
            [~,~,discrCFD]=histcounts(reshape(CFD,[],1),'BinMethod','sqrt');
            discrCFD=reshape(discrCFD,size(CFD,1),[]);
            for currSubj=1:size(CFD,1)
                TR=cell(2,1);
                E=cell(2,1);
                for currClass=1:2
                    relDataIdx=this.lbls==(currClass-1);
                    relDataIdx(currSubj)=0;
                    
                    % Generate seq and emission data for current class
                    stateData=cumsum(reshape(trajBreakMat(relDataIdx,:)',1,[]));
                    emitData=reshape(discrCFD(relDataIdx,:)',1,[]);
                    
                    % Estimate transition and emission matrices
                    [TR{currClass},E{currClass}]=hmmestimate(emitData,stateData);
                    
                    % Modify TR and E to take into account the fact that
                    % first state is not necessarily 1
                    p=histcounts(stateData,'Normalization','pdf','BinMethod','integers');
                    TR{currClass}=[0 p;zeros(size(TR{currClass},1),1) TR{currClass}];
                    E{currClass}=[zeros(1,size(E{currClass},2)); E{currClass}];

                    [PSTATES{currClass},LOGPSEQ{currClass},FORWARD{currClass},BACKWARD{currClass},S{currClass}]=hmmdecode(discrCFD(currSubj,:),TR{currClass},E{currClass});
                end
            end
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
    end
end