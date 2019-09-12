function drawNet(f, net)
    set(groot,'CurrentFigure',f);
    D = 1;
    vertical_interval = 1;
    horizontal_interval = 10;
    hold on
    
    layer_num = [net.input_dim, net.hidden_layer_size_arr, net.output_dim];
    Weight_layers = net.Weight_layers;
    
    max_val = 0;
    for i =1:numel(Weight_layers)
        Weight_layers{i} = abs(Weight_layers{i});
        max_val = max([Weight_layers{i}(:); max_val]);
    end
    
    for i =1:numel(Weight_layers)
        Weight_layers{i} = Weight_layers{i}./max_val;
    end

    y_max_vec = [];
    y_min_vec = [];
    center_mat_list = {};
    x_vec = horizontal_interval+D:horizontal_interval+D:(horizontal_interval+D)*size(layer_num,2) 
    for i = 1:size(layer_num,2)
        length =  (D + vertical_interval)*(layer_num(i)-1);
        y_vec = -length/2:vertical_interval+D:length/2;
        y_max_vec = [y_max_vec;y_vec(end)];
        y_min_vec = [y_min_vec;y_vec(1)];
        center_mat = [ones(layer_num(i),1)*x_vec(i),y_vec.' ]
        viscircles(center_mat,D/2*ones(size(center_mat,1),1),'Color','k');
        center_mat_list = [center_mat_list, {center_mat}];
    end

    for i = 1:size(layer_num,2)-1
        for j = 1:layer_num(i)
            for k = 1:layer_num(i+1)
                start_point = center_mat_list{i}(j,:);
                end_point = center_mat_list{i+1}(k,:);
                p = plot([start_point(1)+D/2,end_point(1)-D/2], [start_point(2),end_point(2)],'k','LineWidth',0.01);
                p.Color(4) = Weight_layers{i}(k,j);
            end
        end
    end


    xlim([0 (horizontal_interval+D)*(size(layer_num,2)+1)])
    ylim([min(y_min_vec)-D max(y_max_vec)+D]);
    hold off
    daspect([1 1 1])
    box off
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca,'ytick',[])
    set(gca,'yticklabel',[])
    hAxes = gca;    
end

