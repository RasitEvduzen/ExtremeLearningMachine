% Recursive Least Squares Based Extrame Learning Machine - Nonlinear Regression
% Written By: Rasit
% 09-Mar-2024
clc,clear all,close all;
%% Input and Output definition
% Create Data Spiral Data
B = 4;
N = 200;
Tall = [];
for i=1:N/2
    theta = pi/2 + (i-1)*[(2*B-1)/N]*pi;
    Tall = [Tall , [theta*cos(theta);theta*sin(theta)]];
end
Tall = [Tall,-Tall];
Tmax = pi/2+[(N/2-1)*(2*B-1)/N]*pi;
xtrain = [Tall]'/Tmax;
ytrain = [-ones(1,N/2), ones(1,N/2)]';

NoD = length(xtrain);
NuberOfNeuron = 50;  % Number of Neuron   (Sparse Model)
% NuberOfNeuron = NoD;  % Number of Neuron

input_w = 2*randn(NuberOfNeuron,size(xtrain,2));  % generate a random input weights
H = tanh(xtrain*input_w');     % TanH Activation Function

%% RLSE Based ELM 
xtest = []; % Test Input
ytest = []; % Test Output
for t1=-1:1e-2:1
    for t2=-1:1e-2:1
        xtest = [xtest; [t1,t2]];
    end
end

A = H;
b = ytrain;
output_w = rand(size(A,2),1);     % Random start RLSE state vector
P = 1e10*eye(size(A,2),size(A,2)); % This tunable parameter depend on activation function

figure('units','normalized','color','w')
for k=1:NoD
    [output_w,K,P] = rlse_online(A(k,:),b(k,:),output_w,P);
    Ht = tanh(xtest*input_w');
    ytest = sign((Ht*output_w));
    % Plot Result
    clf
    plot(xtest(ytest==+1,1),xtest(ytest==+1,2),'r.'),hold on,grid on
    plot(xtest(ytest==-1,1),xtest(ytest==-1,2),'b.')

    plot(xtrain(ytrain==+1,1), xtrain(ytrain==+1,2), 'k*',LineWidth=5)
    plot(xtrain(ytrain==-1,1), xtrain(ytrain==-1,2), 'y*',LineWidth=5)
    title('RLSE-ELM Nonlinear Classification')
    drawnow
end


function [x,K,P] = rlse_online(a_k,b_k,x,P)
% One step of RLSE (Recursive Least Squares Estimation) algorithm
a_k = a_k(:);
b_k = b_k(:);
K = (P*a_k)/(a_k'*P*a_k+1); % Compute Gain K (Like Kalman Gain!)
x = x + K*(b_k-a_k'*x);     % State Update
P = P - K*a_k'*P;           % Covariance Update
end
