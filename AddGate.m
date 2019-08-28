classdef AddGate
    % element-wise operator for vector addition
    methods       
        function out = forward(obj, x, y)
            out = x + y;
        end
        function [dx, dy] = backward(obj, x, y, dz)
           dx = dz.*ones(size(x));
           dy = dz.*ones(size(y));
        end
    end
end