classdef L2WeightLoss
    % weight decay using L2 parameter norm penalty
    methods       
        function loss = forward(obj, w_mat)
            loss = 0;
            for i = 1:size(w_mat,2)
                loss = loss + sum(w_mat(:,i).^2);
            end
        end
        function gradient = backward(obj, w_mat)
                gradient = w_mat;
        end
    end
end

