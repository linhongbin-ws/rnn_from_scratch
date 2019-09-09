net = FFNN(6,5,[30],'activation_fun_str_list',{'tanh','purelin'});
% generate_sim_data_CAD;
net = net.train(input_mat(1:6,:), output_mat(2:6,:),...
                'EpochNum',2000,...
                'LearningRate',3,...
                'FreezeLayer', []);
%t_hat = net.predict();
% figure
% hold on
% plot(x,t,'-k');
% plot(x,t_hat,'-b');
% hold off

