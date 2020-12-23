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
            
            % Reshape T and Y
            T=reshape(T,[size(T,1),1,size(T,2)*size(T,3)]);
            Y=reshape(Y,[size(Y,1),1,size(Y,2)*size(Y,3)]);
            
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
            centroids=sum(fPoints,3)./repmat(sum(Y2,3),[size(T,1),1,1]);
            
            % Centroids should be as far as possible from each other,
            % points should be as close as possible to centroids
            intraClustDist=mean(sum(((fPoints-repmat(centroids,[1,1,size(T,3)])).*repmat(Y2,[size(T,1),1,1])).^2,1),'all');
%             interClustDist=mean(sqrt(sum((repmat(centroids,[1,1,nClusters])-repmat(permute(centroids,[1,3,2]),[1,nClusters,1])).^2)),'all');
            interClustDist=mean(sum((centroids-repmat(mean(centroids,2),1,size(centroids,2))).^2));
            
            % Compute an L0 penalty norm
            L0=mean(max(Y2,[],3));
            
            % Define loss
%             loss=intraClustDist/interClustDist;%+L0;
%             fprintf('%0.2f - %0.2f - %0.2f\n',intraClustDist,interClustDist,L0);
%             fprintf('%0.2f\n',loss);
            loss=exp(log(intraClustDist)-log(interClustDist))+L0;
            if isnan(extractdata(loss))
                keyboard;
            end
        end
    end
end