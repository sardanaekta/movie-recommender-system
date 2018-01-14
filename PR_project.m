%% main functions
load('train_all.mat');
load('test_all.mat');
Y = Rating_train; %rating matrix
R = L_train;      %indicator matrix
sz = size(Y);
num_movies = sz(1);
num_users = sz(2);
lambdas = [0.01,0.1,0.5,1,5,10,20,30,40,60,80,90,100]
num_feat = 10;
max_iter = 500;
len = length(lambdas);
error_rates = zeros(len,1);
alpha = 0.001;
threshold = 0.0001;

%% select optimal lambda
for i = 1:len
    lambda = lambdas(i);
    P = train_model(Y,R,num_feat,lambda,max_iter,alpha,threshold);
    error_rates(i) = (norm(test_R.*(P-test_Y),'fro')^2)/(norm(test_Y,'fro')^2);
end

%% plot lambda vs Error Rate
figure;
plot(lambdas,error_rates,'-o');
ylabel('error rate');
xlabel('\lambda');
title('Lambda vs Error Rate')

%% select optimal number of features
num_features_list = [1,5,10,15,20,30,40,50,100,200];
error_rates_feat = zeros(length(num_features_list),1);
lambda = 10;
for i = 1:length(num_features_list)
    num_feature = num_features_list(i);
    P = train_model(Y,R,num_feature,lambda,max_iter,alpha,threshold);
    error_rates_feat(i) = (norm(test_R.*(P-test_Y),'fro')^2)/(norm(test_Y,'fro')^2);   
end
%% plot num_feature vs Error Rate
figure;
plot(num_features_list,error_rates_feat,'-o');
ylabel('error rate');
xlabel('Number of features');
title('Number of feature vs Error Rate')

%% Predict missing ratings
opt_lambda = 20;
opt_num_feature = 10;
opt_alpha = 0.001;
opt_P = train_model(Y,R,opt_num_feature,opt_lambda,max_iter,opt_alpha,threshold);
file_id = fopen('pred_ratings.txt','wt');
for i=1:num_users
    for j=1:num_movies
        if R(j,i) == 1
            entry = Y(j,i);
        else
            entry = round(opt_P(j,i));
        end
        fprintf(file_id,'%d %d %d\n',i,j,entry);
    end
end
fclose(file_id);






