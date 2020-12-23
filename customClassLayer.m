classdef customClassLayer < nnet.layer.ClassificationLayer
        
    properties
        W; % Class weights
        cutoff; % Samples farther than this many SDs from the mean will have reduced impact on loss
    end
 
    methods
        function layer = customClassLayer(inW,outlrCutoff)
            layer.W=inW;
            if nargin<2||isempty(outlrCutoff)
                outlrCutoff=0;
            end
            layer.cutoff=outlrCutoff;
        end

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
            
            % Compute error, then remove contribution of largest outliers
            % from loss computation
            E=(Y-T).*repmat(layer.W',1,size(Y,2));
            if layer.cutoff~=0
                outlrsW=@(X)min(max(-abs(X)+layer.cutoff+1,0),1);
                sdE=std(E,[],'all');
                E=E.*outlrsW(E/sdE);
            end
            loss=sum(mean((E).^2,2),'all');
        end
    end
end