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
            
            global trMat1
            global trMat2
            global cnts1
            global cnts2
            
            % Preserve original dimensions data
            seqLength=size(T,3);
            nSeqs=size(T,2);
            
            % Recover current class
            currLbl=T(end,1,1);
            
            % Reshape T and Y
            T=reshape(T(1:end-1,:,:),[size(T,1)-1,1,size(T,2)*size(T,3)]);
            Y=reshape(Y(1:end-1,:,:),[size(Y,1)-1,1,size(Y,2)*size(Y,3)]);
            
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
            
            % Compute distance between transition matrices between classes
            Y2=reshape(permute(Y2,[3,1,2]),nSeqs,seqLength,nClusters);
            currTrMat=squeeze(sum(repmat(permute(Y2(:,2:end,:),[1,2,4,3]),[1,1,nClusters,1]).*repmat(Y2(:,1:end-1,:),[1,1,1,nClusters]),2)./repmat(sum(Y2(:,1:end-1,:),2),[1,1,1,nClusters]));
            switch currLbl
                case 1
                    if isempty(trMat1)
                        trMat1=currTrMat;
                        cnts1=1;
                    else
                        trMat1=(trMat1*cnts1+currTrMat)/(cnts1+1);
                        cnts1=cnts1+1;
                    end
                case 2
                    if isempty(trMat2)
                        trMat2=currTrMat;
                        cnts2=1;
                    else
                        trMat2=(trMat2*cnts2+currTrMat)/(cnts2+1);
                        cnts2=cnts2+1;
                    end
            end
%             if isempty(trMat1)||isempty(trMat2)
                trMatDiff=1;
%             else
% %                 trMatDiff=gather(sqrt(sum((trMat1-trMat2).^2,'all'))/nClusters.^2)*1e4+1;
% %                 trMatDiff=gather(1-sum((trMat1-mean(trMat1,'all')).*(trMat2-mean(trMat2,'all')),'all')/sqrt(sum((trMat1-mean(trMat1,'all')).^2,'all').*sum((trMat2-mean(trMat2,'all')).^2,'all')));
%                 trMatDiff=gather(1-sum((trMat1-repmat(mean(trMat1),[nClusters,1])).*(trMat2-repmat(mean(trMat2),[nClusters,1])),'all')./sqrt(sum((trMat1-repmat(mean(trMat1),[nClusters,1])).^2,'all').*sum((trMat2-repmat(mean(trMat2),[nClusters,1])).^2,'all')));
%             end
            
%             % Compute an L0 penalty norm
%             L0=mean(max(Y2,[],3));
            
            % Define loss
%             loss=intraClustDist/interClustDist;%+L0;
%             fprintf('%0.2f - %0.2f - %0.2f\n',intraClustDist,interClustDist,L0);
%             fprintf('%0.2f\n',loss);
%             loss=(exp(log(intraClustDist)-log(interClustDist)))/trMatDiff;
            loss=intraClustDist/(interClustDist*trMatDiff);
%             fprintf('%0.4f, %0.4f, %0.4f\n',intraClustDist,interClustDist,trMatDiff);
            if isnan(extractdata(loss))
                keyboard;
            end
        end
    end
end