classdef ReLUMap
    methods       
        function out = forward(obj, x)
            if(x>0)
               out = x;
            else
               out = zeros(size(x));
            end
        end
        function bottom_delta = backward(obj, x, top_delta)
            if(x>0)
               bottom_delta = top_delta;
            else
               bottom_delta = zeros(size(x));
            end
        end
    end
end

