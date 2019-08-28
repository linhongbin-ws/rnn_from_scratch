classdef QuadraticLoss
    % quadratic loss of one sample
    methods       
        function loss = forward(obj, y, y_hat)
            loss = sum((y-y_hat).^2)./2;
        end
        function gradient = backward(obj, y, y_hat)
                gradient = y_hat - y;
        end
    end
end

