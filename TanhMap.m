classdef TanhMap
    methods       
        function out = forward(obj, x)
           out = tanh(x);
        end
        function bottom_delta = backward(obj, x, top_delta)
            f_o = obj.forward(x);
            bottom_delta = top_delta.*(1 - f_o.^2);
        end
    end
end

