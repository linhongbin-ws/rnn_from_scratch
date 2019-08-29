classdef RNN
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
        function obj = RNN(input_dim,...
                            output_dim,...
                            hidden_layer_size_arr,...
                            varargin)
               obj.net = RNNLayer(input_dim,...
                            output_dim,...
                            hidden_layer_size_arr,...
                            varargin{:});
        end
        
        function y_list =  predict(obj, x_list)
            assert(iscell(x_list) && size(x_list,1), 'x_list should be cell array');
            for i= 1:size(x_list,2)
                x = x_list{i};
                assert(size(x,1) == obj.net.input_dim, fprintf('dimension of x dosent match in index %d of x_list', i));
                assert(obj.is_train==true, 'The network need to be trained before prediction');

                y_list = {};
                for i =1:size(x,2)
                    y = [];
                    input_vec = x(:,i);
                    input_normalized_vec = (input_vec - obj.normalized_struct.input_mean_vec)...
                                                      ./obj.normalized_struct.input_std_vec;
                    if(i == 1)  
                         [output_normalized_vec,~,h_list] = obj.net.forward(input_normalized_vec, obj.net.get_zero_h_list());
                    else
                         [output_normalized_vec,~,h_list] = obj.net.forward(input_normalized_vec, h_list);                        
                    end
                   
                    output_vec = output_normalized_vec.*obj.normalized_struct.output_std_vec...
                                                       +obj.normalized_struct.output_mean_vec;
                    y = [y, output_vec];    
                    y_list = [y_list, {y}];
                end
            end
        end
        
        
        % training function of Neural Network
        function obj = train(obj, X_train_list, Y_train_list, varargin)
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
            addRequired(p,'X_train_list',@iscell);
            addRequired(p,'Y_train_list',@iscell);
            addParameter(p,'TrainMethod',default_train_method,@isstring);
            addParameter(p,'CostFunction',default_costfunction,@isstring);
            addParameter(p,'LearningRate',default_learningRate, @isnumeric);
            addParameter(p,'NormalizedMethod',default_normalized_method, @isstring);
            addParameter(p,'EpochNum',default_epoch_num, @isnumeric)
            parse(p, X_train_list, Y_train_list, varargin{:});          
            obj.train_method = p.Results.TrainMethod;
            cost_function_str = p.Results.CostFunction;
            obj.learning_rate = p.Results.LearningRate;
            obj.cost_function = get_element_instance(cost_function_str);
            obj.is_train =true;
            obj.normalized_struct = [];
            obj.normalized_struct.normalized_method = p.Results.NormalizedMethod;
            obj.train_opt_struct.epoch_num = p.Results.EpochNum;

            
%             assert(size(X_train,1) == obj.net.input_dim, 'dimension of X_train dosent match');
%             assert(size(Y_train,1) == obj.net.output_dim, 'dimension of Y_train dosent match');
%             assert(size(X_train,2) == size(Y_train,2), 'samples number of X_train and Y_train is not match');
%             
%             
            
            % normalizing input and output data
            switch lower(obj.normalized_struct.normalized_method)
                case 'zeroscore'
                    [obj.normalized_struct.input_list,...
                     obj.normalized_struct.input_mean_vec,...
                     obj.normalized_struct.input_std_vec] = obj.z_score_normalize(X_train_list);
                    X_train_list = obj.normalized_struct.input_list;
                    
                    [obj.normalized_struct.output_list,...
                     obj.normalized_struct.output_mean_vec,...
                     obj.normalized_struct.output_std_vec] = obj.z_score_normalize(Y_train);
                    Y_train_list = obj.normalized_struct.output_list;
                    
                otherwise
                    error(fprintf('method %s is not supported', obj.normalized_method))
            end            
            
            % choose type of training method
            switch upper(obj.train_method)
                case 'SGD'
                    obj = obj.SGD(X_train_list, Y_train_list);              
                case 'BGD'
                    obj = obj.BGD(X_train_list, Y_train_list);             
                case 'MBGD'
                    obj = obj.MBGD(X_train_list, Y_train_list);
                otherwise
                    error(fprintf('method %s is not supported', obj.train_method))
            end                    
        end 
    end
    
    methods(Access=protected)
        
        function obj = SGD(obj, X_train_list, Y_train_list)
            count = 0;
            sample_num = size(X_train,2);
            while(1)
                for i = 1:sample_num
                    % calculate gradients of parameters
                    [Weight_layers_delta, Bias_Layers_delta] = obj.net.backward(X_train(:,i),...
                                                                                Y_train(:,i),...
                                                                                obj.cost_function);
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
                    y_hat = obj.net.forward(X_train(:,i));
                    loss = loss + obj.cost_function.forward(Y_train(:,i), y_hat);
                end
                loss = loss/sample_num;
                fprintf('log of loss is %.5f\n', loss);
            end
        end
        
        function [Weight_layers_delta, Bias_Layers_delta,] = SGD(obj, X_train_list, Y_train_list)
        
        
        % Z-score normalization
        function [x_normalized_list, mean_vec, std_vec] = z_score_normalize(obj,x_list)
            x_mat = [];
            for i=1:size(x_list,2)
                x_mat = [x_mat x_list{i}];
            end
            mean_vec = mean(x_mat,2);
            std_vec = std(x_mat,0,2);
            
            x_normalized_list = {};
            for j = 1:size(x_list,2)
                x_normalized = [];
                x = x_list{j};
                for i = 1:size(x,2)
                    x_normalized = [x_normalized,(x(:,i) - mean_vec) ./ std_vec];
                end
                x_normalized_list = [x_normalized_list, {x_normalized}];
            end
        end
    end
end

