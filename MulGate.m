classdef MulGate
    % mulplication operatior for matrix W and vector x
    methods       
        function out = forward(obj, W, x)
            out = W * x;
        end
        function [dW, dx] = backward(obj, W, x, dz)
            dW = dz*x.';
            dx = W.'*dz;
        end
    end
end 

