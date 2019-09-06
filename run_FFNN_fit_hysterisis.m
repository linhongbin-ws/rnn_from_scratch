load('./data/Real_Joint4_10Reps/Real_Joint4_10Reps_pos.mat');
load('./data/Real_Joint4_10Reps/Real_Joint4_10Reps_tor.mat');
train_input_mat = input_mat(4,:);
train_output_mat = output_mat(4,:);

delay = 5;
feature = [train_input_mat(:,delay:end);  train_input_mat(:,1:end-delay+1)];
labels = train_output_mat(:,delay:end);

net = FFNN(size(feature,1),size(labels,1),[30],'activation_fun_str_list',{'tanh','purelin'});
net = net.train(feature, labels,...
                'EpochNum',200,...
                'LearningRate',0.9);
labels_hat = net.predict(feature)

test_traj = linspace(min(train_input_mat), max(train_input_mat), 900)
test_traj = test_traj(:,1+50:end-50);
test_traj = [test_traj, flip(test_traj)];
test_input = [test_traj(:,delay:end);  test_traj(:,1:end-delay+1)];
test_output = net.predict(test_input);

test_traj2 = [train_input_mat; train_input_mat]
labels_hat2 = net.predict(test_traj2)

x = rad2deg(train_input_mat(:,delay:end));

%%
figure
hold on
scatter(x,labels,10,'k', 'filled');
plot(x, labels_hat,'-b','LineWidth',4);
xlabel('{\it q_4} (Deg)','Interpreter','tex')
ylabel(['$\tau_','0'+4,'$ (Nm)'],'Interpreter','latex','fontweight','bold');  
%plot(rad2deg(test_traj(:,delay:end)), test_output, '-r')
%plot(rad2deg(train_input_mat), labels_hat2,'-g')
%plot(x, y_hat_list2{1}, '-g')
legend('Measured Torque','Estimated torque')
set(gca,'FontSize',20)
hold off


