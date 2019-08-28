classdef SigmoidMap
    methods       
        function out = forward(obj, x)
           out = 1.0 ./ (1.0 + exp(-x));
        end
        function bottom_delta = backward(obj, x, top_delta)
            output = obj.forward(x);
            bottom_delta = (1.0 - output) .* output .* top_delta;
        end
    end
end

