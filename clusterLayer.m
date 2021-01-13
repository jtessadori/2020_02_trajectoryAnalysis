classdef clusterLayer < nnet.layer.RegressionLayer
        
    properties
        % (Optional) Layer properties.

        % Layer properties go here.
    end
 
    methods
        function loss = forwardLoss(layer, Y, T)
            % Return the loss between the predictions Y and the training
            % targets T.
            %
            % Inputs:
            %         layer - Output layer
            %         Y     – Predictions made by network
            %         T     – Training targets
            %
            % Output:
            %         loss  - Loss between Y and T

            % Layer forward loss function goes here.
            
            % Last entries of Y contain data representation after layer 1
            dataPoints=squeeze(Y(end-2:end,:,:));
            
            % Reshape T and Y
            T=reshape(T(1:end-3,:,:),[size(T,1)-3,1,size(T,2)*size(T,3)]);
            Y=reshape(Y(1:end-3,:,:),[size(Y,1)-3,1,size(Y,2)*size(Y,3)]);
            
            % Define number of max output clusters
            nClusters=10;
            
            % Compute fuzzy points
%             Y2=permute((Y(1:nClusters,:,:)+1)/2,[2,1,3]);
%             Y2=Y2./repmat(sum(Y2,2),1,size(Y2,2),1);
            softMaxFun=@(x)exp(x)./sum(exp(x));
            Y2=permute(softMaxFun(Y(1:nClusters,:,:)),[2,1,3]);
            fPoints=repmat(Y2,[size(T,1),1,1]).*repmat(T,[1,nClusters,1]);
%             semilogy(squeeze(extractdata(max(Y2,[],3))));
            if any(squeeze(sum(Y2,2))==0)
                keyboard;
            end
            
            % Compute cluster centroids
%             centroids=sum(fPoints,3)./repmat(sum(Y2,3),[size(dataPoints,1),1,1]);
            centroids=sum(fPoints,3)./repmat(sum(Y2,3),[size(T,1),1,1]);
            
            % Centroids should be as far as possible from each other,
            % points should be as close as possible to centroids
            clusterWeights=repmat(sum(Y2,3),[nClusters,1]);
%             interClustDist=mean(squeeze(sqrt(sum((repmat(centroids,[1,1,nClusters])-repmat(permute(centroids,[1,3,2]),[1,nClusters,1])).^2))).*clusterWeights.*clusterWeights','all');
%             intraClustDist=mean(squeeze(sqrt(sum((repmat(dataPoints,[1,1,nClusters])-repmat(permute(centroids,[1,3,2]),[1,size(dataPoints,2),1])).^2))).*permute(Y2,[3,2,1]),'all');

%             interClustDist=mean(squeeze(sum((repmat(centroids,[1,1,nClusters])-repmat(permute(centroids,[1,3,2]),[1,nClusters,1])).^2)).*clusterWeights.*clusterWeights','all');
%             intraClustDist=mean(squeeze(sum((repmat(dataPoints,[1,1,nClusters])-repmat(permute(centroids,[1,3,2]),[1,size(dataPoints,2),1])).^2)).*permute(Y2,[3,2,1]),'all');

            interClustDist=mean(squeeze(sum((repmat(centroids,[1,1,nClusters])-repmat(permute(centroids,[1,3,2]),[1,nClusters,1])).^2)).*clusterWeights.*clusterWeights','all');
            intraClustDist=mean(squeeze(sum((repmat(T,[1,nClusters,1])-repmat(centroids,[1,1,size(T,3)])).^2)).*permute(Y2,[2,3,1]),'all');

%             interClustDist=1;
%             intraClustDist=1;
            
            % Impose that representation must limit distance between
            % subsequent points. Downplay outliers.
%             trajSpeed=sum((dataPoints(:,2:end)-dataPoints(:,1:end-1)).^2);
%             trajSpeed=min(trajSpeed/std(trajSpeed),3)*std(trajSpeed);
%             trajSpeed=mean(trajSpeed)/mean(var(dataPoints,[],2));
%             trajSpeed=1;
            
            % Define loss
%             loss=(intraClustDist/interClustDist)*1e-6+trajSpeed/5;
            loss=intraClustDist/interClustDist;
%             fprintf('%0.4f, %0.4f, %0.4f\n',intraClustDist,interClustDist,trajSpeed);
            if isnan(extractdata(loss))
                keyboard;
            end
        end
    end
end