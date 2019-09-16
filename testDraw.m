net = FFNN(1,1,[50 50],'activation_fun_str_list',{'tanh','sigmoid','purelin'});

f = figure('Renderer', 'painters', 'Position', [0 0 1920 1080])

for i = 1:10
    for j = 1:numel(net.net.Weight_layers)
        net.net.Weight_layers{j} =  randn(size(net.net.Weight_layers{j}));
    end
    
    record(i) = drawNet(f, net.net);
end

figure(2)
daspect([1 1 1])
box off
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])  
movie(record,1,1)

