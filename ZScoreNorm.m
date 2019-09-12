classdef ZScoreNorm
    methods     
        function [x_normalized, mean_vec, std_vec] = normalize(obj,x_mat)
            mean_vec = mean(x_mat,2);
            std_vec = std(x_mat,0,2);
            x_normalized = [];
            for i = 1:size(x_mat,2)
                x_normalized = [x_normalized,(x_mat(:,i) - mean_vec) ./ std_vec];
            end
        end
        
        function x_norm = forward(obj, x_mat, mean_vec, std_vec)
            x_norm = [];
            for i =1:size(x_mat,2)
                norm = (x_mat(:,i) - mean_vec)./std_vec;
                x_norm = [x_norm, norm];          
            end
        end
        
        function x_mat = backward(obj, x_norm_mat, mean_vec, std_vec)
            x_mat = [];
            for i =1:size(x_norm_mat,2)
                x = x_norm_mat(:,i).*std_vec+mean_vec;
                x_mat = [x_mat, x];          
            end
        end
    end
end

