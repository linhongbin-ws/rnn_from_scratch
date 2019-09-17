classdef Adam
    properties
        m_t = [] % mean of momentumn at time t
        v_t = [] % variance of momentumn at time t
        epsilon = 1e-8 % small constant at dedominator
        beta_1 = 0.9 % decay rate of mean 
        beta_2 = 0.999 % decay rate of variance
        gradient_dim 
    end
    
    methods
        function obj = Adam()
        end
        
        % update the state of Adam
        function obj = update_state(obj, gradient)
            if(isempty(obj.m_t))
                assert(isvector(gradient), 'gradient should be a vector');
                obj.m_t = (1-obj.beta_1).*gradient;
                obj.v_t = (1-obj.beta_2).*gradient.^2;
                obj.gradient_dim = size(gradient,1);
            else
                assert(size(gradient,1) == obj.gradient_dim, 'dimesion of gradient do not match');
                obj.m_t = obj.beta_1.*obj.m_t + (1-obj.beta_1).*gradient;
                obj.v_t = obj.beta_2.*obj.v_t + (1-obj.beta_2).*gradient.^2;                
            end
        end
        
        % compute the delta w.r.t. parameters which is used for updating network
        function delta = compute_gradient(obj, gradient)
            m_t_hat = obj.m_t ./(1-obj.beta_1);
            v_t_hat = obj.v_t ./(1-obj.beta_2);
            delta = m_t_hat ./ (sqrt(v_t_hat) + obj.epsilon);
        end
    end
end
