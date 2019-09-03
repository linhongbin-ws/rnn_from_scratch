load('./data/Real_Joint4_10Reps/Real_Joint4_10Reps_pos.mat');
load('./data/Real_Joint4_10Reps/Real_Joint4_10Reps_tor.mat');
train_input_mat = input_mat(4,1:500);
train_output_mat = output_mat(4,1:500);


fixWindowLength = 8;
train_input_cell = {};
train_output_cell = {};
% cut data into cells with fix window
for i = 1:size(train_input_mat,2)-fixWindowLength+1
    train_input_cell = vertcat(train_input_cell, {train_input_mat(:,i:i+fixWindowLength-1)});
    train_output_cell = vertcat(train_output_cell, {train_output_mat(:,i:i+fixWindowLength-1)});
end
train_input_cell = train_input_cell.';
train_output_cell = train_output_cell.';


net = RNN(1,1,[20],'activation_fun_str_list',{'tanh','purelin'});
net = net.train(train_input_cell,...
                train_output_cell,...
                'EpochNum',40,...
                'LearningRate',0.1);

test_input_cell = {train_input_mat};
y_hat_list = net.predict(train_input_cell);
y_hat = [];
for i = 1:size(y_hat_list,2)
    y_hat = [y_hat y_hat_list{i}(:,end)];
end
x = rad2deg(train_input_mat);
figure
hold on
plot(x, train_output_mat,'-k');
plot(x(fixWindowLength:end),y_hat,'-b');
hold off

