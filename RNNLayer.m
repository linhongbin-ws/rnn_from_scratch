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
            obj.hidden_layer_num = size(hidden_layer_size_arr,2);
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
                    % dim(Recursive_Weight_layers) = dim(Weight_layers)-1
                    obj.Recursive_Weight_layers{i} = randn(hidden_layer_size_arr(i),...
                                                     hidden_layer_size_arr(i));
                end
            end 
        end
        
        function [y_hat_mat, a_list_list, h_list_list] = fptt(obj, X_mat)
            y_hat_mat = [];
            a_list_list = {};
            h_list_list = {};
            for i = 1:size(X_mat,2)
                if i == 1
                    h_list = obj.get_zero_h_list();
                    [y, a_list, h_list] = obj.forward(X_mat(:,i), h_list);
                else
                    [y, a_list, h_list] = obj.forward(X_mat(:,i), h_list);
                end
                y_hat_mat = [y_hat_mat, y];
                a_list_list = [a_list_list, {a_list}];
                h_list_list = [h_list_list, {h_list}];
            end
        end
        
        function [y, a_list, h_list] = forward(obj, x, h_list_prev)
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
            y = h;
        end
        % backward computation of neural network
        function  [Sum_Weight_layers_delta,... 
                   Sum_Bias_Layers_delta,...
                   Sum_Recursive_Weight_layers_delta] = bptt(obj,... 
                                                         x_mat,...
                                                         y_mat,...
                                                         cost_function_obj)
            Sum_Weight_layers_delta ={};
            Sum_Bias_Layers_delta = {};
            Sum_Recursive_Weight_layers_delta = {};
            % calculate the forward propagation through time
            [~, a_list_list, h_list_list] = fptt(obj, x_mat);
            
               
            for t = size(x_mat,2):-1:1
                x = x_mat(:,t);
                y = y_mat(:,t);
                h_list = h_list_list{t};
                a_list = a_list_list{t};
                
                if (t==1)
                    h_prev_list = obj.get_zero_h_list();
                else
                    h_prev_list = h_list_list{t-1};
                end
                
                if(t==size(x_mat,2))
                    gradient_list = {};
                    for i = 1:obj.hidden_layer_num
                        gradient_list = [gradient_list, zeros(obj.hidden_layer_size_arr(i), 1)];
                    end
                else
                    gradient_list = next_gradient_list;
                end
                
                [Weight_layers_delta,... 
                  Bias_Layers_delta,...
                  Recursive_Weight_layers_delta,...
                  next_gradient_list] = obj.backward(y,...
                                                     h_prev_list,...
                                                     h_list,...
                                                     a_list,...
                                                     cost_function_obj,...
                                                     gradient_list);
                if(t==size(x_mat,2))
                    Sum_Weight_layers_delta = Weight_layers_delta;
                    Sum_Bias_Layers_delta = Bias_Layers_delta;
                    Sum_Recursive_Weight_layers_delta = Recursive_Weight_layers_delta;
                else
                    for k = 1:size(Weight_layers_delta,2)
                        Sum_Weight_layers_delta{k} = Sum_Weight_layers_delta{k} + Weight_layers_delta{k};
                    end
                    for k = 1:size(Bias_Layers_delta,2)
                        Sum_Bias_Layers_delta{k} = Sum_Bias_Layers_delta{k} + Bias_Layers_delta{k};
                    end
                    for k = 1:size(Sum_Recursive_Weight_layers_delta,2)
                        Sum_Recursive_Weight_layers_delta{k} = Sum_Recursive_Weight_layers_delta{k} + Recursive_Weight_layers_delta{k};
                    end
                end
                
                %calculate mean delta
                for k = 1:size(Sum_Weight_layers_delta,2)
                    Sum_Weight_layers_delta{k} = Sum_Weight_layers_delta{k}./size(x_mat,2);
                end
                for k = 1:size(Sum_Bias_Layers_delta,2)
                    Sum_Bias_Layers_delta{k} = Sum_Bias_Layers_delta{k}./size(x_mat,2);
                end
                for k = 1:size(Sum_Recursive_Weight_layers_delta,2)
                    Sum_Recursive_Weight_layers_delta{k} = Sum_Recursive_Weight_layers_delta{k}./size(x_mat,2);
                end                
            end
        end
        
        function [Weight_layers_delta,... 
                  Bias_Layers_delta,...
                  Recursive_Weight_layers_delta,...
                  next_gradient_list] = backward(obj,... 
                                                 y,...
                                                 h_prev_list,...
                                                 h_list,...
                                                 a_list,...
                                                 cost_function_obj,...
                                                 gradient_list)
                % calculate gradients of each weight and bias parameters
                Weight_layers_delta ={};
                Bias_Layers_delta = {};
                Recursive_Weight_layers_delta = {};
                next_gradient_list = {};
                gradient = cost_function_obj.backward(y, h_list{end});
                for i = obj.hidden_layer_num+1:-1:1
                    gradient = obj.activation_fun_obj_list{i}.backward(a_list{i}, gradient);

                    % calculate gradient of Bias_layers
                    b_delta = gradient;
                    Bias_Layers_delta = [{b_delta}, Bias_Layers_delta];

                    % calculate gradient of Recursive_Weight_layers
                    if(i~=obj.hidden_layer_num+1)
                        [R_delta, h_delta] = obj.mulgate.backward(obj.Recursive_Weight_layers{i}, h_prev_list{i+1}, gradient);
                        Recursive_Weight_layers_delta = [{R_delta}, Recursive_Weight_layers_delta];
                        next_gradient_list = [{h_delta},next_gradient_list];
                    end

                    % calculate gradient of Weight_layers 
                    [W_delta, gradient] = obj.mulgate.backward(obj.Weight_layers{i}, h_list{i}, gradient);               
                    Weight_layers_delta = [{W_delta}, Weight_layers_delta];


                    % calculate gradient of h(l-1)  
                    if (i~=1)
                        gradient = gradient + gradient_list{i-1};
                    end
                end
        end
        
        function obj = update_layer_param(obj,...
                                    Weight_layers_delta,... 
                                    Bias_Layers_delta,...
                                    Recursive_Weight_layers_delta,...
                                    learning_rate)
            for i = 1:size(Weight_layers_delta,2)
                obj.Weight_layers{i} = obj.Weight_layers{i}...
                                        - learning_rate * Weight_layers_delta{i};
            end
                
            for i = 1:size(Bias_Layers_delta,2)
                obj.Bias_Layers{i} = obj.Bias_Layers{i} ...
                                    - learning_rate * Bias_Layers_delta{i};
            end
            
             for i = 1:size(Recursive_Weight_layers_delta,2)
                 obj.Recursive_Weight_layers{i} = obj.Recursive_Weight_layers{i} ...
                                                  - learning_rate * Recursive_Weight_layers_delta{i};
            end
        end
        
        function h_list = get_zero_h_list(obj)
            h_list = {zeros(obj.input_dim,1)};
            for i = 1:size(obj.hidden_layer_num)
                h_list = [h_list, {zeros(obj.hidden_layer_size_arr(i),1)}];
            end
            h_list = [h_list, {zeros(obj.output_dim,1)}];
        end
    end
end

