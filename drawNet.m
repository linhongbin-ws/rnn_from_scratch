function frame = drawNet(f, net)
    persistent center_mat_list
    persistent pWeightLine_list_list

    % set f handle to current figure
    figure(f);

    layer_num = [net.input_dim, net.hidden_layer_size_arr, net.output_dim];
    Weight_layers = net.Weight_layers;
    
    % find the maximum abosolute value of weights & make weight to absolte value
    max_val = 0;
    for i =1:numel(Weight_layers)
        max_val = max([abs(Weight_layers{i}(:)); max_val]);
    end
    
    % normalize the weight value
    for i =1:numel(Weight_layers)
        Weight_layers{i} = Weight_layers{i}./max_val;
    end
    
    if (isempty(get(f, 'Children')))
        % general property
        D = 1;% diameter of neural circle
        vertical_interval = 1; % interval distance between two neurons vertically
        horizontal_interval = 10; % interval distance between two neurons horizontally

        hold on
        y_max_vec = [];
        y_min_vec = [];
        center_mat_list = {};
        x_vec = horizontal_interval+D:horizontal_interval+D:(horizontal_interval+D)*size(layer_num,2) 
        % plot circles representing neurons
        for i = 1:size(layer_num,2)
            length =  (D + vertical_interval)*(layer_num(i)-1);
            y_vec = -length/2:vertical_interval+D:length/2;
            y_max_vec = [y_max_vec;y_vec(end)];
            y_min_vec = [y_min_vec;y_vec(1)];
            center_mat = [ones(layer_num(i),1)*x_vec(i),y_vec.' ]
            viscircles(center_mat,D/2*ones(size(center_mat,1),1),'Color','k');
            center_mat_list = [center_mat_list, {center_mat}];
        end


        pWeightLine_list_list = {};
        pWeightLine_list = {};
        % plot lines representing weight values
        for i = 1:size(layer_num,2)-1
            for j = 1:layer_num(i)
                for k = 1:layer_num(i+1)
                    start_point = center_mat_list{i}(j,:);
                    end_point = center_mat_list{i+1}(k,:);
                        if(Weight_layers{i}(k,j)>0)
                        LinColor(1) = Weight_layers{i}(k,j);
                        LinColor(2) = 0;
                        LinColor(3) = 0;
                        LinColor(4) = Weight_layers{i}(k,j);
                    else
                        temp = abs(Weight_layers{i}(k,j));
                        LinColor(1) = 0;
                        LinColor(2) = 0;
                        LinColor(3) = temp;
                        LinColor(4) = temp;
                    end
                    p = plot([start_point(1)+D/2,end_point(1)-D/2], [start_point(2),end_point(2)],'Color',LinColor,'LineWidth',0.01);
                    pWeightLine_list{k, j} = p;
                end         
            end
            pWeightLine_list_list{i} = pWeightLine_list;
        end
        
        % configure the figure
        xlim([0 (horizontal_interval+D)*(size(layer_num,2)+1)])
        ylim([min(y_min_vec)-D max(y_max_vec)+D]);
        hold off
        daspect([1 1 1])
        box off
        set(gca,'xtick',[])
        set(gca,'xticklabel',[])
        set(gca,'ytick',[])
        set(gca,'yticklabel',[])  
        
    else
        % update lines representing weight values
        for i = 1:size(layer_num,2)-1
            for j = 1:layer_num(i)
                for k = 1:layer_num(i+1)
                    start_point = center_mat_list{i}(j,:);
                    end_point = center_mat_list{i+1}(k,:);
                    if(Weight_layers{i}(k,j)>0)
                        pWeightLine_list_list{i}{j,k}.Color(1) = Weight_layers{i}(k,j);
                        pWeightLine_list_list{i}{j,k}.Color(2) = 0;
                        pWeightLine_list_list{i}{j,k}.Color(3) = 0;
                        pWeightLine_list_list{i}{j,k}.Color(4) = Weight_layers{i}(k,j);
                    else
                        temp = abs(Weight_layers{i}(k,j));
                        pWeightLine_list_list{i}{j,k}.Color(1) = 0;
                        pWeightLine_list_list{i}{j,k}.Color(2) = 0;
                        pWeightLine_list_list{i}{j,k}.Color(3) = temp;
                        pWeightLine_list_list{i}{j,k}.Color(4) = temp;
                    end
                end
            end
        end  
        drawnow
    end
    frame =  getframe(gcf);
end

