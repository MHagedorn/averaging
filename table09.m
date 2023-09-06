% supplementary material on "Optimized convergence of stochastic gradient
% descent by weighted averaging" (2022)

% first simple case of perturbing one function x'*(D.*D.*x)
% final error in dependence of the initial error
clear all 
clc


rng(12345)                 % to make results reproducible
lambdas = 10.^(0:2:8);     % > 0, vary initial value
paropt  = [false, true];
sols    = zeros(length(paropt),length(lambdas));

for i = 1:length(lambdas)
lambda = lambdas(i);
disp(['Step ',num2str(i),'/',num2str(length(lambdas))])

n       = 100;           % dimension
kmax    = 1e5;           % number of iterations
setcond = 10;            % condition of Hessian of f is setcond
setcond = sqrt(setcond); % condition of D = A^{1/2}
rho = 1/sqrt(n); %scaling to have same norm perturbation for all dimensions

D = rand(n,1);
D = D-min(D);                    % minimum is zero now
D = D/max(D);                    % in [0,1] now
D = 1/setcond + (1-1/setcond)*D; % entries in [1/setcond, 1]
D = sort(D);                     % here D is as in Section 3.3 

c  = 1/max(D)^2;  % (here, D is the square root the matrix D in Section 2)
x0 = ones(n,1);   % initial error "evenly distributed" over all components
x0 = x0/sqrt(n);  % normalize for all dimensions
x0 = lambda*x0;   % varying the initial error

for j = 1:length(paropt)
if paropt(j) == false
    beta  = 0;
    alpha = 0;
    M     = 1;
end
if paropt(j) == true
    beta  = 2.4;
    alpha = 0;
    delta = 0;
    M     = 1 + delta * kmax;
end

x     = x0;         % ``plain'' iterates
xa    = zeros(n,1); % averages
sigma = 0;

for k = 0:kmax-1
    bk      = rho*randn(n,1); 
    nablafk = D.*D.*x + bk;
    gamma   = c * (M/(k + M))^alpha;
    wk      = (k+1)^beta;
    x       = x - gamma*nablafk;
    xa      = xa + wk*x;
    sigma   = sigma+wk;
end

xa        = xa/sigma;
xnorm     = norm(x);
xanorm    = norm(xa);

sols(j,i) = xanorm;
end
end


T = table([lambdas;sols(1,:); sols(2,:)]);
T.Properties.RowNames = {'norm(x0)','norm(x_average^final), w^0, gamma^0',...
    'norm(x_average^final), w^opt, gamma^opt'};
T.Properties.VariableNames = {'Table 9'};
T

