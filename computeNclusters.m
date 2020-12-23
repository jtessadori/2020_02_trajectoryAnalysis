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

% Compute distance between projections
FCs=cat(2,projs{:});
D2=zeros(size(FCs,2));
for currPoint1=1:size(FCs,2)
    for currPoint2=currPoint1+1:size(FCs,2)
        D2(currPoint1,currPoint2)=sum((FCs(:,currPoint1)-FCs(:,currPoint2)).^2);
        D2(currPoint2,currPoint1)=D2(currPoint1,currPoint2);
    end
    fprintf('%d/%d\n',currPoint1,size(FCs,2));
end
D=sqrt(D2);

% Consider points closer than 5th percentile of D to be "close", everything
% else "far"
grph=graph(1./(D+1));
lap=laplacian(grph);
lap=lap+eye(size(lap));
E=eig(lap);
srtdE=sort(E,'ascend');

load clusterCompData.mat

plot(srtdE(1:2000))