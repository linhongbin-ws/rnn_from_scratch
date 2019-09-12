classdef FFNN
    properties
        cost_function
        learning_rate
        train_method
        update_method
        is_train = false
        normalized_struct
        train_opt_struct
        train_result_struct
        adapt_method_struct
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
            
            output_mat_norm = [];
            
            input_mat_norm = obj.normalized_struct.method_obj.forward(x,...
                                                                      obj.normalized_struct.input_mean_vec,...
                                                                      obj.normalized_struct.input_std_vec);
            for i =1:size(input_mat_norm,2)
                [output_normalized_vec,~,~] = obj.net.forward(input_mat_norm(:,i));
                output_mat_norm = [output_mat_norm, output_normalized_vec];          
            end
            
            y = obj.normalized_struct.method_obj.backward(output_mat_norm,...
                                                          obj.normalized_struct.output_mean_vec,...
                                                          obj.normalized_struct.output_std_vec);
        end
        
        
        % training function of Neural Network
        function obj = setTrainOpt(obj, X_train, Y_train, varargin)
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
            default_update_method = 'gradient_descend';
            default_adapt_method = [];
            default_fz_layer_index_arr = [];
            addRequired(p,'X_train',@ismatrix);
            addRequired(p,'Y_train',@ismatrix);
            addParameter(p,'TrainMethod',default_train_method,@ischar);
            addParameter(p,'CostFunction',default_costfunction,@ischar);
            addParameter(p,'LearningRate',default_learningRate, @isnumeric);
            addParameter(p,'NormalizedMethod',default_normalized_method, @ischar);
            addParameter(p,'EpochNum',default_epoch_num, @isnumeric)
            addParameter(p,'UpdateMethod',default_update_method, @ischar);
            addParameter(p,'FreezeLayer', default_fz_layer_index_arr, @ismatrix);
            addParameter(p,'AdaptMethod', default_adapt_method, @ischar);
            parse(p, X_train, Y_train, varargin{:});          
            obj.train_method = p.Results.TrainMethod;
            cost_function_str = p.Results.CostFunction;
            obj.learning_rate = p.Results.LearningRate;
            obj.update_method = p.Results.UpdateMethod;
            obj.cost_function = get_element_instance(cost_function_str);
            obj.normalized_struct = [];
            obj.normalized_struct.normalized_method = p.Results.NormalizedMethod;
            obj.train_opt_struct.epoch_num = p.Results.EpochNum;
            obj.train_opt_struct.freeze_layer_arr = p.Results.FreezeLayer;
            obj.adapt_method_struct.adapt_method = p.Results.AdaptMethod;
           
            % default setting
            obj.train_opt_struct.train_ratio = 0.8;
            
            assert(size(X_train,1) == obj.net.input_dim, 'dimension of X_train dosent match');
            assert(size(Y_train,1) == obj.net.output_dim, 'dimension of Y_train dosent match');
            assert(size(X_train,2) == size(Y_train,2), 'samples number of X_train and Y_train is not match');
            
            % configure adaptive method
            if (~isempty(obj.adapt_method_struct.adapt_method))
                obj.adapt_method_struct.isAdapt = true;
                switch lower(obj.adapt_method_struct.adapt_method)
                    case 'adam'
                        obj.adapt_method_struct.adapt_obj = Adam();    
                    otherwise
                        error(fprintf('method %s is not supported', obj.adapt_method_struct))
                end    
            else
                obj.adapt_method_struct.isAdapt = false;
            end
             obj.train_opt_struct.input_data = X_train;
             obj.train_opt_struct.output_data = Y_train;    
        end
        function obj = start_train(obj)
            % partition training and validation data
            random_idx = randperm(size(obj.train_opt_struct.input_data,2));
            X = obj.train_opt_struct.input_data(random_idx);
            Y = obj.train_opt_struct.output_data(random_idx);
            
            train_size = fix(size(obj.train_opt_struct.input_data,2)*obj.train_opt_struct.train_ratio);
            X_train = X(1:train_size);
            Y_train = Y(1:train_size);
            X_validate = X(train_size+1:end);
            Y_validate = Y(train_size+1:end);
            
            
            % normalizing input and output data
            switch lower(obj.normalized_struct.normalized_method)
                case 'zeroscore'
                    obj.normalized_struct.method_obj = ZScoreNorm();
                    [obj.normalized_struct.input_mat,...
                     obj.normalized_struct.input_mean_vec,...
                     obj.normalized_struct.input_std_vec] = obj.normalized_struct.method_obj.normalize(X_train);
                    
                    [obj.normalized_struct.output_mat,...
                     obj.normalized_struct.output_mean_vec,...
                     obj.normalized_struct.output_std_vec] = obj.normalized_struct.method_obj.normalize(Y_train);
                    
                otherwise
                    error(fprintf('method %s is not supported', obj.normalized_struct.normalized_method))
            end
            X_validate_norm = obj.normalized_struct.method_obj.forward(X_validate,...
                                                                       obj.normalized_struct.input_mean_vec,...
                                                                       obj.normalized_struct.input_std_vec);
            Y_validate_norm = obj.normalized_struct.method_obj.forward(Y_validate,...
                                                                       obj.normalized_struct.output_mean_vec,...
                                                                       obj.normalized_struct.output_std_vec);
            
            % choose type of training method
            switch upper(obj.train_method)
                % stochastic gradient descent
                case 'SGD'
                    obj = obj.SGD(obj.normalized_struct.input_mat,...
                                  obj.normalized_struct.output_mat,...
                                  X_validate_norm,...
                                  Y_validate_norm);              
                case 'BGD'
                    obj = obj.BGD(obj.normalized_struct.input_mat,...
                                  obj.normalized_struct.output_mat,...
                                  X_validate_norm,...
                                  Y_validate_norm,0.1);  
                % gradient descent    
                case 'GD'
                    obj = obj.GD(obj.normalized_struct.input_mat,...
                                  obj.normalized_struct.output_mat,...
                                  X_validate_norm,...
                                  Y_validate_norm);
                otherwise
                    error(fprintf('method %s is not supported', obj.train_method))
            end    
            obj.is_train =true;
        end 
    end
    
    methods(Access=protected)
        
        function obj = SGD(obj, X_train, Y_train, X_validate, Y_validate)
            count =  0;
            random_idx = randperm(size(X_train,2));
            X_train = X_train(random_idx);
            Y_train = Y_train(random_idx);
            while(1) 
                for i = 1:size(X_train,2)
                    obj = update_batch(obj, X_train(:,i), Y_train(:,i));
                end
                count = count + 1;
                if count>=obj.train_opt_struct.epoch_num
                    break
                end
                obj.evaluate_model(count, X_train, Y_train, X_validate, Y_validate);
            end
        end
  
        function obj = BGD(obj, X_train, Y_train, X_validate, Y_validate, partition_ratio)
            count =  0;
            random_idx = randperm(size(X_train,2));
            X_train = X_train(random_idx);
            Y_train = Y_train(random_idx);
            
            % find the nearest integer number on the right
            batch_size = fix(size(X_train,2)*partition_ratio)+1;
            batch_index_list = {};
            count = 0;
            for i = 1:fix(size(X_train,2)/batch_size)
                if (i == fix(size(X_train,2)/batch_size))
                    batch_index_list = [batch_index_list, {count+1:1:size(X_train,2)}];
                    count = count + size(batch_index_list{end},2);
                else
                    batch_index_list = [batch_index_list, {count+1:1:count+batch_size}];
                    count = count + batch_size;
                end
            end
            
            while(1) 
                for i = 1:size(batch_index_list,2)
                    obj = update_batch(obj, X_train(:,batch_index_list{i}), Y_train(:,batch_index_list{i}));
                end
                count = count + 1;
                if count>=obj.train_opt_struct.epoch_num
                    break
                end
                obj.evaluate_model(count, X_train, Y_train, X_validate, Y_validate);
            end
        end

      function obj = GD(obj, X_train, Y_train, X_validate, Y_validate)
            count =  0;
            while(1) 
                obj = update_batch(obj, X_train, Y_train);
                count = count + 1;
                if count>=obj.train_opt_struct.epoch_num
                    break
                end
                obj.evaluate_model(count, X_train, Y_train, X_validate, Y_validate);
            end
      end
        
      function evaluate_model(obj, epoch_num, X_train, Y_train, X_validate, Y_validate)
                % calculate train loss
                loss = 0;
                for i=1:size(X_train,2)
                    y_hat = obj.net.forward(X_train(:,i));
                    loss = loss + obj.cost_function.forward(Y_train(:,i), y_hat);
                end
                obj.train_result_struct.train_loss = loss/size(X_train,2)/obj.net.output_dim;
                
                % calculate validation loss
                loss = 0;
                for i=1:size(X_validate,2)
                    y_hat = obj.net.forward(X_validate(:,i));
                    loss = loss + obj.cost_function.forward(Y_validate(:,i), y_hat);
                end
                obj.train_result_struct.validate_loss = loss/size(X_validate,2)/obj.net.output_dim;
                
                fprintf('Epoch num is %d: train loss: %.5f, validate loss:%.5f \n', epoch_num,...
                                                                                   obj.train_result_struct.train_loss,...
                                                                                   obj.train_result_struct.validate_loss);
      end
            
        function obj = update_batch(obj, X_train, Y_train)  
            sample_num = size(X_train,2);
            
            % set freeze layer
            Weight_layers_ratio = {};
            for i = 1:size(obj.net.Weight_layers,2)
                Weight_layers_ratio = [Weight_layers_ratio, {ones(size(obj.net.Weight_layers{i}))}];
            end
            Bias_Layers_ratio = {};
            for i = 1:size(obj.net.Bias_Layers,2)
                Bias_Layers_ratio = [Bias_Layers_ratio, {ones(size(obj.net.Bias_Layers{i}))}];
            end

            for i=1:size(obj.train_opt_struct.freeze_layer_arr,2)
                index = obj.train_opt_struct.freeze_layer_arr(i);
                Weight_layers_ratio{index} = zeros(size(Weight_layers_ratio{index}));
                Bias_Layers_ratio{index} = zeros(size(Bias_Layers_ratio{index}));
            end
            
                   
            % initial delta to zeros
            Weight_layers_delta = {};
            for i = 1:size(obj.net.Weight_layers,2)
                Weight_layers_delta = [Weight_layers_delta, {zeros(size(obj.net.Weight_layers{i}))}];
            end
            Bias_Layers_delta = {};
            for i = 1:size(obj.net.Bias_Layers,2)
                Bias_Layers_delta = [Bias_Layers_delta, {zeros(size(obj.net.Bias_Layers{i}))}];
            end
                
            % sum of gradient
            for i = 1:sample_num
                [W_d, B_d] = obj.net.backward(X_train(:,i),...
                                                Y_train(:,i),...
                                                obj.cost_function);
                for i = 1:size(Weight_layers_delta,2)
                    Weight_layers_delta{i} = Weight_layers_delta{i} + W_d{i};
                end
                for i = 1:size(Bias_Layers_delta,2)
                    Bias_Layers_delta{i} = Bias_Layers_delta{i} + B_d{i};
                end                                                                                                  
            end
                
            % normalize to stardard gradient
            for i = 1:size(Weight_layers_delta,2)
                Weight_layers_delta{i} = Weight_layers_delta{i}./sample_num./obj.net.output_dim;
            end
            for i = 1:size(Bias_Layers_delta,2)
                Bias_Layers_delta{i} = Bias_Layers_delta{i}./sample_num./obj.net.output_dim;
            end     
                                         
            % optimized by adaptive methods
            if (obj.adapt_method_struct.isAdapt)
                % vectorize the gradients of Weight and bias 
                gradient_vec = [];
                for i =1:numel(Weight_layers_delta)
                    gradient_vec = [gradient_vec; Weight_layers_delta{i}(:)];
                end
                for i =1:numel(Bias_Layers_delta)
                    gradient_vec = [gradient_vec; Bias_Layers_delta{i}(:)];
                end    

                obj.adapt_method_struct.adapt_obj = ...
                    obj.adapt_method_struct.adapt_obj.update_state(gradient_vec);
                delta_vec = obj.adapt_method_struct.adapt_obj.compute_delta();

                % reverse gradient vector to weight and bias
                n = 0;
                for i =1:numel(Weight_layers_delta)
                    num = numel(Weight_layers_delta{i});
                    Weight_layers_delta{i} = reshape(delta_vec(n+1:n+num),...
                                                    [size(Weight_layers_delta{i},1),size(Weight_layers_delta{i},2)]);
                    n = n + num;
                end
                for i =1:numel(Bias_Layers_delta)
                    num = numel(Bias_Layers_delta{i});
                    Bias_Layers_delta{i} = reshape(delta_vec(n+1:n+num),...
                                                    [size(Bias_Layers_delta{i},1),size(Bias_Layers_delta{i},2)]);
                    n = n + num;
                end                 
            end

            % update parameters
            switch lower(obj.update_method)
                case 'gradient_descend'
                    obj.net = obj.gradient_decent(obj.net,...
                            Weight_layers_delta,... 
                            Bias_Layers_delta,...
                            obj.learning_rate,...
                            Weight_layers_ratio,...
                            Bias_Layers_ratio);            
                otherwise
                    error(fprintf('method %s is not supported', obj.train_method))
            end 
        end
        
       function net = gradient_decent(obj,...
                            net,...
                            Weight_layers_delta,... 
                            Bias_Layers_delta,...
                            learning_rate,...
                            Weight_layers_ratio,...
                            Bias_Layers_ratio)
            
            for i = 1:size(Weight_layers_delta,2)
                net.Weight_layers{i} = net.Weight_layers{i}...
                                    - (learning_rate * Weight_layers_delta{i}).*Weight_layers_ratio{i};
            end
                
            for i = 1:size(Bias_Layers_delta,2)
                net.Bias_Layers{i} = net.Bias_Layers{i} ...
                                    - (learning_rate * Bias_Layers_delta{i}).*Bias_Layers_ratio{i};
            end 
       end
        
       % update using Levenberg-Marquardt method
%         function net = LM(obj,...
%                             net,...
%                             Weight_layers_delta,... 
%                             Bias_Layers_delta,...
%                             learning_rate,...
%                             Weight_layers_ratio,...
%                             Bias_Layers_ratio)
%                         
%             % update with ratio 1 by default
%             if ~exist('Weight_layers_ratio','var') || ~exist('Bias_Layers_ratio','var')
%                 Weight_layers_ratio = {};
%                 for i = 1:size(Weight_layers_delta,2)
%                     Weight_layers_ratio = [Weight_layers_ratio, {ones(size(Weight_layers_delta{i}))}];
%                 end
%                 Bias_Layers_ratio = {};
%                 for i = 1:size(Bias_Layers_delta,2)
%                     Bias_Layers_ratio = [Bias_Layers_ratio, {ones(size(Bias_Layers_delta{i}))}];
%                 end
%             end
%             
%             for i = 1:size(Weight_layers_delta,2)
%                 net.Weight_layers{i} = net.Weight_layers{i}...
%                                     - (learning_rate * Weight_layers_delta{i}*weight).*Weight_layers_ratio{i};
%             end
%                 
%             for i = 1:size(Bias_Layers_delta,2)
%                 net.Bias_Layers{i} = net.Bias_Layers{i} ...
%                                     - (learning_rate * Bias_Layers_delta{i}*weight).*Bias_Layers_delta{i};
%             end 
%         end
        
        
%         % Z-score normalization
%         function [x_normalized, mean_vec, std_vec] = z_score_normalize(obj,x)
%             mean_vec = mean(x,2);
%             std_vec = std(x,0,2);
%             x_normalized = [];
%             for i = 1:size(x,2)
%                 x_normalized = [x_normalized,(x(:,i) - mean_vec) ./ std_vec];
%             end
%         end
        
     
    end
end

