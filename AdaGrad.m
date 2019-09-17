classdef AdaGrad
    properties
        G_t = [];
        epsilon = 1e-8;
    end
    
    methods
        function obj = AdaGrad()
        end
  
        
        % compute the delta w.r.t. parameters which is used for updating network
        function obj = update_state(obj, gradient)
            G_ = gradient*gradient.';
            if(isempty(obj.G_t))
                obj.G_t = G_;
            else
                obj.G_t = obj.G_t + G_;
            end
        end
        
        function ada_grad = compute_gradient(obj, gradient)
            divider = diag(diag(obj.G_t)) + diag(obj.epsilon*ones(size(gradient)));
            ada_grad = eye(size(divider))/sqrt(divider)*gradient;
        end
    end
end
