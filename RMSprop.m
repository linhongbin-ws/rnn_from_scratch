classdef RMSprop
    properties
        G_t = [];
        epsilon = 1e-8;
        lamda = 0.9;
    end
    
    methods
        function obj = RMSprop()
        end
  
        
        % compute the delta w.r.t. parameters which is used for updating network
        function obj = update_state(obj, gradient)
            G_ = gradient*gradient.';
            if(isempty(obj.G_t))
                obj.G_t = G_;
            else
                obj.G_t = obj.G_t*obj.lamda + G_*(1-obj.lamda);
            end
        end
        
        function ada_grad = compute_gradient(obj, gradient)
            divider = diag(diag(obj.G_t)) + diag(obj.epsilon*ones(size(gradient)));
            ada_grad = eye(size(divider))/sqrt(divider)*gradient;
        end
    end
end
