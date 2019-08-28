classdef FFNNLayer
    properties
        mulgate 
        addgate
        Weight_layers = {}
        Bias_Layers = {}
        activation_fun_str_list = {}
        activation_fun_obj_list = {}
    end
    methods
        function obj = FFNNLayer(input_dim,...
                                output_dim,...
                                hidden_layer_size_arr,...
                                varargin)
            % parse arguments
            is_array =  @(x) isvector(x.');
            addRequired(p,'input_dim',@isscalar);
            addRequired(p,'output_dim',@isscalar);
            addRequired(p,'hidden_layer_size_arr',is_array);
            addOptional(p,'activation_fun_str_list',defal,@iscell); 
            parse(p,input_dim, output_dim, hidden_layer_size_arr, varargin{:});
            obj.activation_fun_str_list = p.Results.activation_fun_str_list;
            obj.input_dim = input_dim;
            obj.output_dim = output_dim;
            obj.hidden_layer_num = size(hidden_layer_size_arr,2)
            % default activation function is hyperbolic tangent function
            if size(obj.activation_fun_str_list,2) ~= obj.hidden_layer_num+1
                obj.activation_fun_str_list = {};
                for i =1:obj.hidden_layer_num
                    obj.activation_fun_str_list{end+1} = 'tanh';
                end
                
                % activation function of output layer is pure linear gate
                obj.activation_fun_str_list{end+1} = 'purelin';
            end
            
            
            % initial essensial elements
            obj.mulgate = MulGate();
            obj.addgate = AddGate();
            for i = 1:obj.hidden_layer_num+1
                obj.activation_fun_obj_list{end+1} = ...
                    obj.get_element_instance(obj.activation_fun_str_list{i});
            end
            
            % initial weight and bias layers by randomizing using normal
            %                                                 distribution
            for i = 1:obj.hidden_layer_num+1
                if(i == 1)
                    obj.Weight_layers{end+1} = randn(hidden_layer_size_arr(i),...
                                                     input_dim);
                    obj.Bias_Layers{end+1} = randn(hidden_layer_size_arr(i),...
                                                     1);
                elseif(i == obj.hidden_layer_num+1)
                    obj.Weight_layers{end+1} = randn(output_dim,...
                                                     hidden_layer_size_arr(i-1));
                    obj.Bias_Layers{end+1} = randn(output_dim,...
                                                     1);
                else
                    obj.Weight_layers{end+1} = randn(hidden_layer_size_arr(i),...
                                                     hidden_layer_size_arr(i-1));
                    obj.Bias_Layers{end+1} = randn(hidden_layer_size_arr(i),...
                                                     1);
                end
            end 
        end
        function out = forward(obj, x)
            a = x;
            a_list = {x};
            y_list = {};
            for i = 1:obj.hidden_layer_num+1
                y = obj.mulgate.forward(obj.Weight_layers{i}, a);
                y = obj.addgate.forward(y, obj.Bias_Layers{i});
                a = obj.activation_fun_obj_list{i}.forward(y);
                y_list{end+1} = y;
                a_list{end+1} = a;
            end
        end
        function bottom_delta = backward(obj, x, top_delta)
            f_o = obj.forward(x);
            bottom_delta = top_delta.*(1 - f_o.^2);
        end
    end
end

