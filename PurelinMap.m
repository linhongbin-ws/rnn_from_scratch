classdef PurelinMap
    % pure linear gate
    methods       
        function out = forward(obj, x)
            out = x;
        end
        function bottom_delta = backward(obj, x, top_delta)
            bottom_delta = top_delta;
        end
    end
end

