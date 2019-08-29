load('./data/Real_Joint4_10Reps/Real_Joint4_10Reps_pos.mat');
load('./data/Real_Joint4_10Reps/Real_Joint4_10Reps_tor.mat');
train_input_mat = input_mat(4,:);
train_output_mat = output_mat(4,:);


fixWindowLength = 8;
train_input_cell = {};
train_output_cell = {};
% cut data into cells with fix window
for i = 1:size(train_input_mat,2)-fixWindowLength+1
    train_input_cell = vertcat(train_input_cell, {train_input_mat(:,i:i+fixWindowLength-1)});
    train_output_cell = vertcat(train_output_cell, {train_output_mat(:,i:i+fixWindowLength-1)});
end


net = RNN(1,1,[30],'activation_fun_str_list',{'tanh','purelin'});
net = net.train(x, t,...
                'EpochNum',2000,...
                'LearningRate',3);
t_hat = net.predict(x);
figure
hold on
plot(x,t,'-k');
plot(x,t_hat,'-b');
hold off

