classdef trajClassLayer < nnet.layer.ClassificationLayer
        
    properties
        W; % Class weights
        cutoff; % Samples farther than this many SDs from the mean will have reduced impact on loss
    end
 
    methods
        function layer = trajClassLayer(inW)
            layer.W=inW;
%             layer.NumInputs=3;
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
            
            % Compute weighted error
            E=(Y-T).*repmat(layer.W',1,size(Y,2));
            loss=sum(mean((E).^2,2),'all');
        end
    end
end