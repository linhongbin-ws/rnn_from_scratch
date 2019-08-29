classdef FFNN
    properties
        cost_function
        learning_rate
        train_method
        is_train = false
        normalized_struct
        train_opt_struct
        net
    end
        
    methods(Access=public)
        function obj = FFNN(input_dim,...
                            output_dim,...
                            hidden_layer_size_arr,...
                            varargin)
               obj.net = FFNNLayer(input_dim,...
                            output_dim,...
                            hidden_layer_size_arr,...
                            varargin{:});
        end
        
        function y =  predict(obj, x)
            assert(size(x,1) == obj.net.input_dim, 'dimension of x dosent match');
            assert(obj.is_train==true, 'The network need to be trained before prediction');
            
            y = [];
            for i =1:size(x,2)
                input_vec = x(:,i);
                input_normalized_vec = (input_vec - obj.normalized_struct.input_mean_vec)...
                                                  ./obj.normalized_struct.input_std_vec;
                [output_normalized_vec,~,~] = obj.net.forward(input_normalized_vec);
                output_vec = output_normalized_vec.*obj.normalized_struct.output_std_vec...
                                                   +obj.normalized_struct.output_mean_vec;
                y = [y, output_vec];          
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
            obj.cost_function = get_element_instance(cost_function_str);
            obj.is_train =true;
            obj.normalized_struct = [];
            obj.normalized_struct.normalized_method = p.Results.NormalizedMethod;
            obj.train_opt_struct.epoch_num = p.Results.EpochNum;

            
            assert(size(X_train,1) == obj.net.input_dim, 'dimension of X_train dosent match');
            assert(size(Y_train,1) == obj.net.output_dim, 'dimension of Y_train dosent match');
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
                        obj.net.Weight_layers{j} = obj.net.Weight_layers{j} - obj.learning_rate * Weight_layers_delta{j}/sample_num;
                        obj.net.Bias_Layers{j} = obj.net.Bias_Layers{j} - obj.learning_rate * Bias_Layers_delta{j}/sample_num;
                    end
                end
                count = count + 1;
                if count>=obj.train_opt_struct.epoch_num
                    break
                end
                
                loss = 0;
                for i=1:sample_num
                    y_hat = obj.forward(X_train(:,i));
                    loss = loss + obj.cost_function.forward(Y_train(:,i), y_hat);
                end
                loss = loss/sample_num;
                fprintf('log of loss is %.5f\n', loss);
            end
        end
        
        % forward computation of neural network
        function [y_hat, a_list, h_list] = forward(obj, x)
            [y_hat, a_list, h_list] = obj.net.forward(x);        
        end
        
        % backward computation of neural network
        function  [Weight_layers_delta, Bias_Layers_delta] = backward(obj, x, y)
            [Weight_layers_delta, Bias_Layers_delta] = obj.net.backward(x, y, obj.cost_function);
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

