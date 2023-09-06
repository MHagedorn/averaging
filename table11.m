% supplementary material on "Optimized convergence of stochastic gradient
% descent by weighted averaging" (2022)

% second more realistic case of having different functions f_k
% final error in dependence of the number of iterations
clear all
clc

rng(12345)                 % to make results reproducible
kmaxs   = 10.^(3:8);       % vary number of iterations
paropt  = [false, true];
sols    = zeros(length(paropt), length(kmaxs));
Sols    = zeros(length(paropt), length(kmaxs));

for run = 1:10
    
n       = 100;          % dimension
lambda  = 100;          % initial error
setcond = 10;           % condition of Hessian of f is setcond
setcond = sqrt(setcond);% condition of D = A^{1/2}
rho     = 1/sqrt(n);    % to have same norm perturbation for all dimensions

D = rand(n,1);
D = D-min(D);                    % minimum is zero now
D = D/max(D);                    % in [0,1] now
D = 1/setcond + (1-1/setcond)*D; % entries in [1/setcond, 1]
D = sort(D);                     % here D is as in Section 3.3 
D = D*10;                        % scale D 


c  = 1/max(D)^2;  % (here, D is the square root the matrix D in Section 2)
x0 = ones(n,1);   % initial error "evenly distributed" over all components
x0 = x0/sqrt(n);  % normalize for all dimensions 
x0 = lambda * x0; % initial error


for i = 1:length(kmaxs)
    disp(['Step ',num2str(i),'\',num2str(length(kmaxs))])
    kmax = kmaxs(i);
    
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

sigma = 0;
x  = x0;          % ``plain'' iterates
xa = zeros(n,1);  % averages

for k = 0:kmax-1
    rk      = rho*randn(n,1);
    bk      = rho*randn(n,1);
    ak      = D.*rk;
    nablafk = ak*(ak.'*x) + bk;
    
    gamma = c * (M/(k + M))^alpha;
    wk    = (k+1)^beta;
    x     = x - gamma*nablafk;
    xa    = xa + wk*x;
    sigma = sigma+wk;
end

xa        = xa/sigma;
xnorm     = norm(x);	    
xanorm    = norm(xa);
sols(j,i) = xanorm;
end
end

Sols = Sols + sols;

end

Sols = Sols./10;

disp(['Norm of final iterate in dependence of the number of iterations',... 
    ' and beta:'])
T = table(Sols(:,1),Sols(:,2),Sols(:,3),Sols(:,4),Sols(:,5),Sols(:,6));
T.Properties.VariableNames = split(num2str(kmaxs));
T.Properties.RowNames = {'norm(x_average^final), w^0; gamma^0',...
    'norm(x_average^final), w^opt, gamma^opt'}
