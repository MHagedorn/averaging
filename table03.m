% supplementary material on "Optimized convergence of stochastic gradient
% descent by weighted averaging" (2022)

% optimal parameters alpha, beta, c and delta for kmax = 10^(4.5), 
% D11 = 10^(-4) and different values of eta
clear all
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% alpha                             in [0,2]   (power in step length)
% beta                              in [0,5]   (weight w_k = k^beta)
% M = 1 + delta * kmax              with delta in [0,1]   (offset)
% c                                 in [0.1,1] (step length reduction)
% for Dnn = 1 it holds c = delta2 * (1/Dnn) = delta2 in tpk4par.m

kmax = round(10^(4.5), -2);
Dnn  = 1;       
D11  = 10^(-4); % > 0, corresponds to cond(D) = Dnn/D11 = 10^4
eta  = 0.001;   % >= 0

% bounds of alpha, beta, delta, c
options.lb = [0; 0; 0; 0.1]; % lower bounds
options.ub = [2; 5; 1;   1]; % upper bounds

% standard parameters, when modifying just one parameter
options.par_f.D11 = D11; 
options.par_f.Dnn = Dnn; 
options.par_f.k   = kmax; 
options.par_f.eta = eta; 

% for comparison the values of kappa and tau by Polyak and Juditsky
% [~, kappa0, tau0] = tpk4par([0,0,0,1], options.par_f)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% determination of the optimal parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for eta = [1, 0.1, 0.017, 0.01, 0.001]
    
    options.par_f.eta = eta; 
    disp(['For eta = ', num2str(eta),':'])

    x1 = [0;0;0;1]; % initial values
    [x1opt,~,~,~,~]        = min_f(@tpk4par, x1, options);
    [r1opt,ka1opt,tau1opt] = tpk4par(x1opt, options.par_f);
    tpkout1 = round([tau1opt,ka1opt,r1opt], 3);
    
    x2 = [0;1;0;1];
    [x2opt,~,~,~,~]        = min_f(@tpk4par, x2, options);
    [r2opt,ka2opt,tau2opt] = tpk4par(x2opt,options.par_f);
    tpkout2 = round([tau2opt,ka2opt,r2opt], 3);

    x3 = [1;0;0;1];
    [x3opt,~,~,~,~]        = min_f(@tpk4par, x3, options);
    [r3opt,ka3opt,tau3opt] = tpk4par(x3opt,options.par_f);
    tpkout3 = round([tau3opt,ka3opt,r3opt], 3);

    x4 = [1;1;0;1];
    [x4opt,~,~,~,~]        = min_f(@tpk4par, x4, options);
    [r4opt,ka4opt,tau4opt] = tpk4par(x4opt,options.par_f);
    tpkout4 = round([tau4opt,ka4opt,r4opt], 3);
    
    T = table([x1opt([1,2,4,3]); tpkout1'],... 
              [x2opt([1,2,4,3]); tpkout2'],...
              [x3opt([1,2,4,3]); tpkout3'],...
              [x4opt([1,2,4,3]); tpkout4']);
    T.Properties.RowNames = {'alpha','beta','c','delta','tau','kappa','r'};
   
    T
end