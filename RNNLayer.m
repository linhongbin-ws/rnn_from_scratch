classdef RNNLayer
    properties
        mulgate 
        addgate
        input_dim
        output_dim
        hidden_layer_size_arr
        hidden_layer_num
        Weight_layers = {}
        Recursive_Weight_layers = {}
        Bias_Layers = {}
        activation_fun_str_list = {}
        activation_fun_obj_list = {}
    end
    methods
        function obj = RNNLayer(input_dim,...
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
                % initialize recursive weights
                if(i~=obj.hidden_layer_num+1)
                    obj.Recursive_Weight_layers{i} = randn(hidden_layer_size_arr(i),...
                                                     hidden_layer_size_arr(i));
                end
            end 
        end
        function [h, a_list, h_list] = forward(obj, x, h_list_prev)
            h = x;
            h_list = {x};
            a_list = {};
            for i = 1:obj.hidden_layer_num+1
                a = obj.mulgate.forward(obj.Weight_layers{i}, h);
                if(i~=obj.hidden_layer_num+1)
                     r = obj.mulgate.forward(obj.Recursive_Weight_layers{i}, h_list_prev{i+1});
                     a = obj.addgate.forward(a,r);                     
                end
                a = obj.addgate.forward(a, obj.Bias_Layers{i});
                h = obj.activation_fun_obj_list{i}.forward(a);
                a_list{end+1} = a;
                h_list{end+1} = h;
            end
           
        end
        % backward computation of neural network
        function  [Weight_layers_delta,... 
                   Bias_Layers_delta,...
                   Recursive_Weight_layers_delta,...
                   recursive_gradient_list] = backward(obj,... 
                                               x,y, ...
                                               cost_function_obj,...
                                               latter_recursive_gradient_list,...
                                               h_list_prev)
            [y_hat, a_list, h_list] = obj.forward(x);
        
            % calculate gradients of each weight and bias parameters
            gradient = cost_function_obj.backward(y, y_hat);
            Weight_layers_delta ={};
            Bias_Layers_delta = {};
            Recursive_Weight_layers_delta = {};
            recursive_gradient_list = {};
            for i = obj.hidden_layer_num+1:-1:1
                gradient = obj.activation_fun_obj_list{i}.backward(a_list{i}, gradient);
                b_delta = gradient;
                if(i~=obj.hidden_layer_num+1)
                    R_delta = obj.mulgate.backward(obj.Recursive_Weight_layers{i}, h_list_prev{i+1}, gradient);
                    recursive_gradient = obj.Recursive_Weight_layers{i}.'*gradient;
                end
                [W_delta, gradient] = obj.mulgate.backward(obj.Weight_layers{i}, h_list{i}, gradient);
                if(i~=obj.hidden_layer_num+1)
                    gradient = gradient + latter_recursive_gradient_list{i};   
                end                
                Weight_layers_delta = [{W_delta}, Weight_layers_delta];
                Bias_Layers_delta = [{b_delta}, Bias_Layers_delta];
                h_gradient_list = [{gradient}, h_gradient_list];
                recursive_gradient_list = [{recursive_gradient}, recursive_gradient_list];
            end
        end
        
        function h_list = get_zero_h_list(obj)
            h_list = {zeros(obj.input_dim,1)};
            for i = 1:size(obj.hidden_layer_num)
                h_list = [h_list, {zeros(obj.hidden_layer_num(i),1)}];
            end
            h_list = [h_list, {zeros(obj.output_dim,1)}];
        end
    end
end

