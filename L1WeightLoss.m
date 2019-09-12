classdef L1WeightLoss
    % weight decay using L2 parameter norm penalty
    methods       
        function loss = forward(obj, w_mat)
            loss = 0;
            for i = 1:size(w_mat,2)
                loss = loss + sum(abs(w_mat(:,i)));
            end
        end
        function gradient = backward(obj, w_mat)
                gradient = sign(w_mat);
        end
    end
end

