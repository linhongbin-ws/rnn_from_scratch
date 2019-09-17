classdef FFNNLayer
    properties
        mulgate 
        addgate
        input_dim
        output_dim
        hidden_layer_size_arr
        hidden_layer_num
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
            p = inputParser;
            is_array =  @(x) isvector(x.');
            addRequired(p,'input_dim',@isscalar);
            addRequired(p,'output_dim',@isscalar);
            addRequired(p,'hidden_layer_size_arr',is_array);
            addOptional(p,'activation_fun_str_list',{},@iscell); 
            parse(p,input_dim, output_dim, hidden_layer_size_arr, varargin{:});
            obj.activation_fun_str_list = p.Results.activation_fun_str_list;
            obj.input_dim = input_dim;
            obj.output_dim = output_dim;
            obj.hidden_layer_num = size(hidden_layer_size_arr,2)
            obj.hidden_layer_size_arr = hidden_layer_size_arr;
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
                    get_element_instance(obj.activation_fun_str_list{i});
            end
            
            % initial weight layers by randomizing using normal distribution
            %         bias layers by zeros value
            for i = 1:obj.hidden_layer_num+1
                if(i == 1)
                    obj.Weight_layers{end+1} = randn(hidden_layer_size_arr(i),...
                                                     input_dim);
                    obj.Bias_Layers{end+1} = zeros(hidden_layer_size_arr(i),...
                                                     1);
                elseif(i == obj.hidden_layer_num+1)
                    obj.Weight_layers{end+1} = randn(output_dim,...
                                                     hidden_layer_size_arr(i-1));
                    obj.Bias_Layers{end+1} = zeros(output_dim,...
                                                     1);
                else
                    obj.Weight_layers{end+1} = randn(hidden_layer_size_arr(i),...
                                                     hidden_layer_size_arr(i-1));
                    obj.Bias_Layers{end+1} = zeros(hidden_layer_size_arr(i),...
                                                     1);
                end
            end 
        end
        function [h, a_list, h_list] = forward(obj, x)
            h = x;
            h_list = {x};
            a_list = {};
            for i = 1:obj.hidden_layer_num+1
                a = obj.mulgate.forward(obj.Weight_layers{i}, h);
                a = obj.addgate.forward(a, obj.Bias_Layers{i});
                h = obj.activation_fun_obj_list{i}.forward(a);
                a_list{end+1} = a;
                h_list{end+1} = h;
            end
           
        end
        % backward computation of neural network
        function  [Weight_layers_delta, Bias_Layers_delta] = backward(obj, x, t, cost_function_obj)
            [t_hat, a_list, h_list] = obj.forward(x);
            
            % calculate gradients of each weight and bias parameters
            gradient = cost_function_obj.backward(t, t_hat);
            Weight_layers_delta ={};
            Bias_Layers_delta = {};
            for i = obj.hidden_layer_num+1:-1:1
                gradient = obj.activation_fun_obj_list{i}.backward(a_list{i}, gradient);
                b_delta = gradient;
                [W_delta, gradient] = obj.mulgate.backward(obj.Weight_layers{i}, h_list{i}, gradient);
                Weight_layers_delta = [{W_delta}, Weight_layers_delta];
                Bias_Layers_delta = [{b_delta}, Bias_Layers_delta];
            end
        end
    end
end

