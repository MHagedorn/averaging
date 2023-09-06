% supplementary material on "Optimized convergence of stochastic gradient
% descent by weighted averaging" (2022)

% first simple case of perturbing one function x'*(D.*D.*x)
% final error in dependence of kmax
clear all 
clc


rng(12345)         % to make results reproducible
kmaxs = 10.^(4:8); % vary number of iterations from 10^4 to 10^8
sols  = zeros(1,length(kmaxs));

for i = 1:length(kmaxs)
kmax = kmaxs(i);
disp(['Step ',num2str(i),'/',num2str(length(kmaxs))])

n       = 100;           % dimension
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
x0 = x0/norm(x0); % normalize for all dimensions


x     = x0;         % ``plain'' iterates
xa    = zeros(n,1); % averages
sigma = 0;
beta  = 2.4;        % suggested value by the theoretical considerations
alpha = 0;
delta = 0;
M     = 1 + delta * kmax;

for k = 0:kmax-1
    bk      = rho*randn(n,1); 
    nablafk = D.*D.*x + bk;
    gamma   = c * (M/(k + M))^alpha;
    wk      = (k+1)^beta;
    x       = x - gamma*nablafk;
    xa      = xa + wk*x;
    sigma   = sigma + wk;
end

xa      = xa/sigma;
xnorm   = norm(x);	    
xanorm  = norm(xa);

sols(i) = xanorm;
end


T = table([kmaxs;sols]);
T.Properties.RowNames = {'Number of iterations','norm(x_average^final)'};
T.Properties.VariableNames = {'Table 8'};
T