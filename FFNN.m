classdef FFNN
    properties
        mulgate 
        addgate
        tanhgate
        input_dim
        output_dim
        hidden_layer_size_arr
        hidden_layer_num
        Weight_layers = {}
        Bias_Layers = {}
        activation_fun_str_list = {}
        activation_fun_obj_list = {}
        a_list = {}
        y_list = {}
        cost_function
        learning_rate
        train_method
        is_train = false
        normalized_struct
        train_opt_struct
    end
        
    methods(Access=public)
        function obj = FFNN(input_dim,...
                            output_dim,...
                            hidden_layer_size_arr,...
                            varargin)
            % parse arguments
            p = inputParser;
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
        
        function y =  predict(obj, x)
            assert(size(x,1) == obj.input_dim, 'dimension of x dosent match');
            assert(obj.is_train==true, 'The network need to be trained before prediction');
            
            y = [];
            for i =1:size(x,2)
                input_vec = x(:,i);
                input_normalized_vec = (input_vec - obj.normalized_struct.input_mean_vec)...
                                                  ./obj.normalized_struct.input_std_vec;
                [output_normalized_vec,~,~] = obj.forward(input_normalized_vec);
                output_vec = output_normalized_vec.*obj.normalized_struct.output_std_vec...
                                                   +obj.normalized_struct.output_mean_vec;
                y = [y, output_vec];          
            end
        end
        
        function instance = get_element_instance(obj, str)
            switch lower(str)
               case 'tanh'
                  instance = TanhMap();
               case 'add'
                  instance = AddGate();
                case 'multiply'
                  instance = MulGate();
                case 'purelin'
                  instance = PurelinMap();
               case 'sigmoid'
                  instance = SigmoidMap();
                case 'quadratic'
                  instance = QuadraticLoss();                  
                  
               otherwise
                  error('cannot recogize %s in get_gate_instance function',str);
            end
        end
        
        % training function of Neural Network
        function obj = train(obj, X_train, Y_train, varargin)
            % X_train: feature matrix of training samples
            % Y_train: label matrix of training samples
            % method: SGD(default) --- stochastic gradient descent
            %         BGD --- Batch gradient descent
            %         MBGD --- Mini-batched gradient descent
            
            % CostFunction: Qudratic(default) --- Quadratic cost || aka MLS method
            p = inputParser;
            default_train_method = 'SGD'; 
            default_costfunction = 'quadratic';
            default_learningRate = 0.1;
            default_epoch_num = 40;
            default_normalized_method = 'zeroscore';
            addRequired(p,'X_train',@ismatrix);
            addRequired(p,'Y_train',@ismatrix);
            addParameter(p,'TrainMethod',default_train_method,@isstring);
            addParameter(p,'CostFunction',default_costfunction,@isstring);
            addParameter(p,'LearningRate',default_learningRate, @isnumeric);
            addParameter(p,'NormalizedMethod',default_normalized_method, @isstring);
            addParameter(p,'EpochNum',default_epoch_num, @isnumeric)
            parse(p, X_train, Y_train, varargin{:});          
            obj.train_method = p.Results.TrainMethod;
            cost_function_str = p.Results.CostFunction;
            obj.learning_rate = p.Results.LearningRate;
            obj.cost_function = obj.get_element_instance(cost_function_str);
            obj.is_train =true;
            obj.normalized_struct = [];
            obj.normalized_struct.normalized_method = p.Results.NormalizedMethod;
            obj.train_opt_struct.epoch_num = p.Results.EpochNum;

            
            assert(size(X_train,1) == obj.input_dim, 'dimension of X_train dosent match');
            assert(size(Y_train,1) == obj.output_dim, 'dimension of Y_train dosent match');
            assert(size(X_train,2) == size(Y_train,2), 'samples number of X_train and Y_train is not match');
            
            
            
            % normalizing input and output data
            switch lower(obj.normalized_struct.normalized_method)
                case 'zeroscore'
                    [obj.normalized_struct.input_mat,...
                     obj.normalized_struct.input_mean_vec,...
                     obj.normalized_struct.input_std_vec] = obj.z_score_normalize(X_train);
                    X_train = obj.normalized_struct.input_mat;
                    
                    [obj.normalized_struct.output_mat,...
                     obj.normalized_struct.output_mean_vec,...
                     obj.normalized_struct.output_std_vec] = obj.z_score_normalize(Y_train);
                    Y_train = obj.normalized_struct.output_mat;
                    
                otherwise
                    error(fprintf('method %s is not supported', obj.normalized_method))
            end            
            
            % choose type of training method
            switch upper(obj.train_method)
                case 'SGD'
                    obj = obj.SGD(X_train, Y_train);              
                case 'BGD'
                    obj = obj.BGD(X_train, Y_train);             
                case 'MBGD'
                    obj = obj.MBGD(X_train, Y_train);
                otherwise
                    error(fprintf('method %s is not supported', obj.train_method))
            end                    
        end 
    end
    
    methods(Access=protected)
        
        function obj = SGD(obj, X_train, Y_train)
            count = 0;
            sample_num = size(X_train,2);
            while(1)
                for i = 1:sample_num
                    % calculate gradients of parameters
                    [Weight_layers_delta, Bias_Layers_delta] = obj.backward(X_train(:,i),...
                                                                            Y_train(:,i));
                    % update parameters
                    for j = 1:size(Weight_layers_delta,2)
                        obj.Weight_layers{j} = obj.Weight_layers{j} - obj.learning_rate * Weight_layers_delta{j}/sample_num;
                        obj.Bias_Layers{j} = obj.Bias_Layers{j} - obj.learning_rate * Bias_Layers_delta{j}/sample_num;
                    end
                end
                count = count + 1;
                if count>=obj.train_opt_struct.epoch_num
                    break
                end
                
                loss = 0;
                for i=1:sample_num
                    t_hat = obj.forward(X_train(:,i));
                    loss = loss + obj.cost_function.forward(Y_train(:,i), t_hat);
                end
                loss = loss/sample_num;
                fprintf('log of loss is %.5f\n', loss);
            end
        end
        
        % forward computation of neural network
        function [a, y_list, a_list] = forward(obj, x)
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
        
        % backward computation of neural network
        function  [Weight_layers_delta, Bias_Layers_delta] = backward(obj, x, t)
            [t_hat, y_list, a_list] = obj.forward(x);
            
            % calculate gradients of each weight and bias parameters
            gradient = obj.cost_function.backward(t, t_hat);
            Weight_layers_delta ={};
            Bias_Layers_delta = {};
            for i = obj.hidden_layer_num+1:-1:1
                gradient = obj.activation_fun_obj_list{i}.backward(y_list{i}, gradient);
                b_delta = gradient;
                [W_delta, gradient] = obj.mulgate.backward(obj.Weight_layers{i}, a_list{i}, gradient);
                Weight_layers_delta = [{W_delta}, Weight_layers_delta];
                Bias_Layers_delta = [{b_delta}, Bias_Layers_delta];
            end
        end
        
        % Z-score normalization
        function [x_normalized, mean_vec, std_vec] = z_score_normalize(obj,x)
            mean_vec = mean(x,2);
            std_vec = std(x,0,2);
            x_normalized = [];
            for i = 1:size(x,2)
                x_normalized = [x_normalized,(x(:,i) - mean_vec) ./ std_vec];
            end
        end
        
        
    end
end

