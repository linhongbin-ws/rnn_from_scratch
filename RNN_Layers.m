classdef RNN_Layers
    properties
        mulgate 
        addgate
        tanh_layer
    end
        
    methods
        function obj = RNN_Layers(hidden_neuron_num)
            obj.mulgate = MulGate();
            obj.addgate = AddGate();
            obj.tanh_layer = Tanh_Layer();
        end
        function out = feedforward(obj, x, pre_s, U, W, V)
            a = feedforward(mulgate, x, U);
        end
        function diff_bottom = backward(obj, x, diff_top)
            f_o = obj.feedforward(x);
            diff_bottom = 1 - f_o^2;
        end
    end
end

