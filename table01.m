% supplementary material on "Optimized convergence of stochastic gradient
% descent by weighted averaging" (2022)

% the values of log10(tau) when alpha = beta = delta = 0 and c = 1
% for different kmax and condition numbers
clear all 
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha = 0; % in [0,2]   (power in step length)
beta  = 0; % in [0,5]   (weight w_k = k^beta)
delta = 0; % in [0,1]   (offset)
c     = 1; % in [0.1,1] (step length reduction)
eta   = 1;

Dnn   = 1;
D11   = 10^(-3);             % i.e. condition number 10^3
kmax  = round(10^(4.5), -2); % i.e. 31600 


% standard parameters, when modifying just one parameter
options.par_f.D11 = D11; 
options.par_f.Dnn = Dnn; 
options.par_f.k   = kmax; 
options.par_f.eta = eta; 
                                                    
tabTau   = zeros(13, 8);                                                     

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% calculation of the values for kappa and tau
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
kk    = 10.^(2:0.5:8);   % kmax = 10^2, 10^(2.5), .., 10^8                                     
conds = 10.^(0.5:0.5:4); % cond = 10^(0.5), 10^1, .., 10^4
for i = 1:length(kk)
   disp(['Step ',num2str(i),'/13'])
   for j = 1:length(conds)
       % specification of iteration number and condition number
       options.par_f.D11 = round(Dnn/conds(j), round(2.5 + 0.5*j));
       options.par_f.k   = round(kk(i),-(length(num2str(round(kk(i))))-3));
       
       % calculation of values of kappa and tau
       [~,~,tau]   = tpk4par([alpha, beta, delta, c], options.par_f);
       tabTau(i,j) = log10(tau); % saving results
   end    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% creating table
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
format short
rows = {'10^2','10^2.5','10^3','10^3.5','10^4','10^4.5','10^5','10^5.5',...
    '10^6','10^6.5','10^7','10^7.5','10^8'};                               
cols = {'10^0.5','10^1','10^1.5','10^2','10^2.5','10^3','10^3.5','10^4'};
tabTau = round(tabTau, 4);
tabTau = table(tabTau(:,1),tabTau(:,2),tabTau(:,3),tabTau(:,4),...
    tabTau(:,5),tabTau(:,6),tabTau(:,7),tabTau(:,8));
tabTau.Properties.RowNames      = rows;
tabTau.Properties.VariableNames = cols;

disp(['log_10(tau(1,1)) for kmax = 10^2,..,10^8 in the rows ',...
    'and condition numbers 10^0.5,..,10^4 in the columns:'])
tabTau



