function [x,y,fx,out] = min_fc(f_obj,f_con,options)
% MIN_FC, MINimize F subject to Constraints,
% i.e. smooth local minimzation without using derivatives, subject to
% linear and non-linear constraints
%
% calling routine:
%   [x,y,fx,out] = min_fc(@f_obj,@f_con,options)
%
% minimize  f_obj(x)  for  f_e(x)  = 0,   f_i(x) <= 0,
%                          Ae*x    = be, Ai*x    <= bi, 
%                          lb <= x <= ub 
%
% where the output for f_con are two vectors f_e and f_i 
% (in this order, representing equalities and inequalities).
%
% Aims:
% MIN_FC aims at finding either a LOCAL minimizer or a (feasible or
% infeasible) stationary point where the derivatives of the active and
% the violated constraints are (nearly) linearly dependent.
% Ideally, at the output of MIN_FC either the Fritz-John conditions are
% satisfied or the output is an infeasible point for which there does not 
% exist any direction improving the linearization of the infeasibility.
% (No guarantee the output of MIN_FC is close to such ideal output.) 
%
% Florian Jarre, Felix Lieder, last change Feb. 2023
% Test Version with errors -- No guarantee of any kind is given!!!
%
% Mandatory input: 
%   f_obj, a function handle for the objective function R^n --> R     
%   f_con, a function handle for the constraints R^n --> [R^p, R^q]  
% !!! Even when p=0 or q = 0, the output of f_con must consist of two   !!!
% !!! vectors [fe ,fi] = f_con(x), with fe  = zeros(0,1) if p = 0,      !!!
% !!! or fi = zeros(0,1) if q = 0.    Here, p and q do not have to be   !!!
% !!! specified; they will be determined from fe and fi.                !!!
% !!! (fe and fi must be column vectors.)                               !!!
%
% Further MANDATORY input:
%   options, a structure with the MANDATORY field
%     x0, starting point, not necessarily feasible but of correct dimension
%
%   and the further OPTIONAL fields:
%     Ae, a constraint matrix for equality constraints
%     be, right hand side  (default for Ae, be: empty)
%     Ai, a constraint matrix for inequality constraints
%     bi, right hand side  (default for Ai, bi: empty)
%     lb, lower bounds on the variable x, of dim (n,1)   (Default -Inf)
%     ub, upper bounds on the variable x, of dim (n,1)   (Default  Inf)
%         (Sparsity of the bounds is not exploited. Setting lb_i=-Inf or 
%          ub_i=Inf is more efficient than e.g., lb_i=-1e20 or ub_i=1e20.)
%     par_fobj, If f_obj depends on additional parameters (not subject
%               to optimization), then the field options.par_fobj is a
%               struct containing the input parameters.
%               Default: options.par_fobj is not provided
%     par_fcon, If f_con depends on additional parameters (not subject
%               to optimization), then the field options.par_fcon is a
%               struct containing the input parameters.
%               Default: options.par_fcon is not provided
%
%     maxit, a bound on the number of iterations; each iteration takes
%            about 2*n+15: evaluations of f_obj, f_con (Default 100*n)
%     err,   if the absolute error for a typical evaluation of f_obj or
%            f_con is known, this parameter can be set here, else it
%            is estimated (and used for the finite differences).
%     p_l,   print_level, values 1 or 2 for more printout, default 0
%
% Output:
%   x:         an approximate local minimizer / stationary point
%   y:         the associated Lagrange multipliers in the order: fe, Ae, 
%              fi, Ai, (Ai including the bounds lb,ub)
%              y is empty if there is only one degree of freedom (after
%              the elimination of linear constraints) i.e. if the problem
%              can be reduced to a line search
%   fx:        the final objective value
%   out.relKKTres:   Norm of the gradient of the Lagrangian divided by
%                    max(1,norm of the gradient of the objective function)
%   out.constr_viol = [norm(constraint violation, nonlinear   equalities),
%                      norm(constraint violation,    linear   equalities),
%                      norm(constraint violation, nonlinear inequalities),
%                      norm(constraint violation,    linear inequalities)];
%   out.iter:        number of iterations needed
%   out.fval:        number of function evaluations needed
%   out.Ae     transformed matrix of linear   equalities used for ye
%   out.Ain    transformed matrix of linear inequalities used for yin
%   out.termmsg termination message
%               1) KKT point found with 6 digits accuracy
%               2) infeasible stationary point discovered with 6 digits acc
%               3) KKT point found with 3 digits accuracy
%               4) Early termination as iteration limit has been reached
%               5) Termination without reaching convergence
%               6) Termination without improving over initial point
% !!!     The termination messages are based on estimates and are       !!!
% !!!                     NOT RELIABLE                                  !!!
%               
%
% Algorithm:
% Trust region SQP algorithm with finite difference approximation 
% of the gradient using a PSB update for the Hessian of the Lagrangian.


%RandStream.setDefaultStream(RandStream('mt19937ar','seed',100));
%RandStream.setGlobalStream(RandStream('mt19937ar','seed',100));

desired_tol = 10.e-8; % some desired stopping tolerance
                      % the actual final error typically is much larger
converged = 0;        % no convergence yet

% COMPLETE INPUT ARGUMENTS
if (nargin < 3)
   error('input for mwdnlc must contain f_obj, f_con, and options');
end
if ~isfield(options,'x0')
   error('starting point must be supplied in options.x0');
end

if ~isfield(options,'par_fobj')
   f = @(x) NaN2Inf(f_obj(x));
else
   f = @(x) NaN2Inf(f_obj(x,options.par_fobj));
end
if ~isfield(options,'par_fcon')
   fcon = @(x) NaN2Inf_con1(@(u) f_con(u), x);
else
   %fcon = @(x) NaN2Inf_con2(@(u) f_con(u), x,options.par_fcon);
   fcon = @(x) NaN2Inf_con2(@(u) f_con(u,options.par_fcon), x);
end

x0 = options.x0;
[n,dummy] = size(x0);
if dummy ~= 1
   error('x0 must be a column vector')
end

if ~isfield(options,'lb')
   options.lb = -Inf*ones(n,1);
end
if ~isfield(options,'ub')
   options.ub =  Inf*ones(n,1);
end
if ~isfield(options,'maxit')
   options.maxit = 100*n; 
end
if ~isfield(options,'err')
   err = Inf; % no accuracy known
else
   err = options.err;
end
lb = options.lb;
ub = options.ub;
if ~isfield(options,'Ae')
   mAe = 0;
   Ae = zeros(0,n);
   be = zeros(0,1);
else
   Ae = full(options.Ae);
   [mAe,dummy] = size(Ae);
   if dummy ~= n
      error('size of Ae does not match size of x0');
   end
   be = options.be;
   [dummy,dummy2] = size(be);
   if dummy ~= mAe || dummy2 ~= 1
      error('size of be does not match size of Ae');
   end
end
if ~isfield(options,'Ai')
   mAi = 0;
   Ai = zeros(0,n);
   bi = zeros(0,1);
else
   Ai = full(options.Ai);
   [mAi,dummy] = size(Ai);
   if dummy ~= n
      error('size of Ai does not match size of x0');
   end
   bi = options.bi;
   if max(abs(bi)) == Inf
      if sum(bi == -Inf) > 0
         error( ' linear constraints unsatisfiable ');
      end
      tmp = bi < Inf;
      bi  = bi(tmp);
      Ai  = Ai(tmp,:);
      [mAi,~] = size(Ai);
   end
   [dummy,dummy2] = size(bi);
   if dummy ~= mAi || dummy2 ~= 1
      error('size of bi does not match size of Ai');
   end
end
if ~isfield(options,'p_l')
   p_l = 0;  
   options.p_l = p_l; 
else
   p_l = options.p_l;
end
[dummy,dummy2] = size(lb);
if dummy ~= n || dummy2 ~= 1
   error('size of lb does not match size of x0');
end
[dummy,dummy2] = size(ub);
if dummy ~= n || dummy2 ~= 1
   error('size of ub does not match size of x0');
end

if options.maxit < 2
    disp('enforce at least two iterations');
    options.maxit = 2;
end
maxit = options.maxit;

if min(ub-lb) < 0
   error('bounds are not consistent');
end




% INCLUDE BOUNDS WITHIN THE MATRIX Ai          (change Ai and mAi)
Id = eye(n);
if min(ub) == -Inf
    error('some upper bound in min_fc at -Inf')
end
if max(lb) == Inf
    error('some lower bound in min_fc at +Inf')
end
fixedbnd = lb == ub;
if ~isempty(lb(fixedbnd))
   Ae = [Ae;Id(fixedbnd,:)];
   be = [be;lb(fixedbnd)  ];      
   % These (and other linear) equation(s) are eliminated below
   ub(fixedbnd) = Inf;
   lb(fixedbnd) = -Inf;
end
finitelbnd = lb > -Inf;
if mAi > 0 && ~isempty(lb(finitelbnd))
   Ai = [Ai;-Id(finitelbnd,:)]; 
   bi = [bi;-lb(finitelbnd)]; 
   mAi = mAi+length(lb(finitelbnd));
end
if mAi == 0 && ~isempty(lb(finitelbnd))
   Ai = -Id(finitelbnd,:); 
   bi = -lb(finitelbnd); 
   mAi = length(lb(finitelbnd));
end
finiteubnd = ub < Inf;
if mAi > 0 && ~isempty(ub(finiteubnd))
   Ai = [Ai;Id(finiteubnd,:)]; 
   bi = [bi;ub(finiteubnd)]; 
   mAi = mAi+length(ub(finiteubnd));
end
if mAi == 0 && ~isempty(ub(finiteubnd))
   Ai = Id(finiteubnd,:); 
   bi = ub(finiteubnd); 
   mAi = length(ub(finiteubnd));
end
clear fixedbnd; clear finitelbnd; clear finiteubnd; clear Id;
Ai = full(Ai); % this is needed only in Octave since Id(finitelbnd,:)
               % is a permutation matrix in Octave and thus, sparse



% ELIMINATE LINEAR EQUALITIES
Q2 = eye(n); % for the case mAe == 0
nn = n;      % number of degrees of freedom (after eliminating linear eq.)
if norm(Ae,'fro') == 0
   if norm(be) > 0
      error('inconsistent linear equations in min_fc')
   end
   mAe = 0;
   Ae = zeros(0,n); be = zeros(0,1);
end
if mAe > 0
% First, orthogonalize the rows of Ae and eliminate linearly dependent rows
   tolAe = eps^.66; % relative tolerance for estimating singularity of Ae
   [~,R,E] = qr(Ae.',0);
   rank_est = sum(abs(diag(R))>tolAe*max(abs(diag(R))));
   if rank_est < mAe
      if norm(Ae*(Ae\be)-be) > tolAe*norm(be)
         disp('linear equations in min_fc seem to be inconsistent')
         converged = 1;
         x = zeros(n,1);
         y = [];
         fx = f(x);
         out.relKKTres = 0;
         out.constr_viol = [Inf,1,1,1];
         out.iter = 0;
         out.fval = 0;
         out.Ae = Ae;
         out.Ain = Ai;
         out.termmsg = 2;
      end
   end
   Ae = Ae(E(1:rank_est),:);
   be = be(E(1:rank_est)); 
   mAe = rank_est;
   Ae = (R(1:rank_est,1:rank_est).')\Ae; % orthonormalize
   be = (R(1:rank_est,1:rank_est).')\be;
% Now, eliminate the linear equalities
   [Q,R] = qr([Ae.',zeros(n,n-mAe)]); 
% by above preprocessing Ae' has linearly independent columns, thus, 
% no permutation
   Q1 = Q(:,1:mAe);
   R1 = R(1:mAe,1:mAe);
   Q2 = Q(:,mAe+1:n);
% change starting point x0 to the projection satisfying linear equalities
   x0 = x0 + Q1*(R1.'\(be-Ae*x0)); 
   x0 = x0 + Q1*(R1.'\(be-Ae*x0)); % to reduce rounding errors
   nn = n-mAe;
end
if mAi > 0
   Ain = Ai * Q2; % only consider points x0+Q2*z
   bin = bi - Ai*x0;
else
   Ain = zeros(0,n-mAe);
   bin = zeros(0,1);
end




% CALL THE ACTUAL SOLVER
if converged == 0 % converged not {1 due to inconsistent linear equations} 
if nn == 1 % only one variable after eliminating linear equations
   bigM = 1.0e8;
   dx = Q2;
   ddx = zeros(size(x0));
   optL.lb = -Inf;
   optL.ub =  Inf;
   badcase = 0;
   if mAi > 0
      Ax = Ai*dx;
      b  = bi - Ai*x0;
      if max(Ax) > 0
         tmp = Ax > 0;
         optL.ub = min(b(tmp)./Ax(tmp));
      end
      if min(Ax) < 0
         tmp = Ax < 0;
         optL.lb = min(-b(tmp)./Ax(tmp));
      end
      if optL.lb > optL.ub
         badcase = 1;
      end
   end  
   optL.tol = 1.0e-8;
   [tx,fx,~,~,outL] = mwd11(@(t) curvimerit(f,fcon,t,x0,dx,ddx,bigM),optL);
   [fe,fi]= fcon(x0+tx*dx);
   x = tx; % used below
   out.constr_viol = [norm(fe),0,norm(fi(fi>0)),0]; 
   out.iter = outL.iter;
   out.relKKTres = 1.0e-8;
   out.dt = 1.0e-6;
   out.fval = outL.iter;
   if max(out.constr_viol) > 1.0e-6
      out.termmsg = 2; 
   else
      out.termmsg = 1;  
   end
   if badcase
      out.relKKTres = 1; 
      out.termmsg = 2;
   end
else % we now have nn > 1
   if mAe == 0 % ``normal'' case of more than one unknown and no linear 
               % equations;  just shift the starting point to zero
      [x,~,fx,out] = mwdnonlin(@(x) f(x0+x), @(x) fcon(x0+x), ...
          Ain,bin,maxit,err,desired_tol,p_l);
   else % after elimination of the LINEAR equations:
      if min(size(Q2)) == 0 % ``exceptional'' case, x unique by lin. eq.
         x = zeros(0,1);
         if options.p_l > 0
            disp('linear equations determine x uniquely')
         end
         [tmp1,tmp2] = fcon(x0);
         %y = zeros(mAi+length(tmp1)+length(tmp2),1);
         tmp3 = 0;
         if ~isempty(tmp1)
            tmp3 = max(abs(tmp1));
         end
         tmp4 = 0;
         if ~isempty(tmp2)
            tmp4 = max(0,max(tmp2));
         end
         tmp5 = 0;
         if mAi > 0
            tmp5 = max(0,max(-bin));
         end
         fx = f(x0);
         out.fval        = 1;
         out.constr_viol = [tmp3,0,tmp4,tmp5]; 
         out.iter        = 0;
         out.relKKTres   = 0;
         out.dt          = 1.0e-6;
         out.fval        = 1;
         out.termmsg     = 1; 

      else % ``normal'' case of more than one unknown after eliminating 
           % the linear equations
         [x,~,fx,out] = mwdnonlin(@(x) f(x0+Q2*x), @(x) fcon(x0+Q2*x), ...
                                  Ain,bin,maxit,err,desired_tol,p_l);
      % Note, the output x has less than n components
      end
   end
end

if mAe == 0
   x = x0+x;
else
   x = x0+Q2*x;
end
 

% RECALCULATE THE MULTIPLYERS IN THE ORIGINAL SPACE:
if nn > 1 && min(size(Q2)) > 0 %%%  not a mere line search / unique eq.
   dt              = out.dt;
   [g,Dfe,Dfi]     = update_firstder(f,fcon,x,dt);
   % above in particular when mAe > 0, but also to have most recent data
   [fe,fi]         = fcon(x);
   out.fval        = out.fval+2*n+1;
   out.constr_viol = [norm(fe),norm(Ae*x-be),norm(max(0,fi)),...
                      norm(max(0,Ai*x-bi))];
   bm              = bi - Ai*x; 
   
   Dfirownorm = 1+sum(Dfi.*Dfi,2).^.5;
   options.activefi = fi >= -out.act_tol*Dfirownorm;
   options.activeAi = bm <=  out.act_tol;
   Dtmp = [Dfe;Ae];
   %options.activefi = out.activefi;
   %options.activeAi = out.activeAi;
   [ye,yi,yA,~] = findlagmult(g,Dtmp,Dfi,fi,Ai,bm,options);
   y = [ye;yi;yA];
   out.relKKTres = norm(g+[Dfe.',Ae.',Dfi.',Ai.']*y)/max(1,norm(g));
   if isnan(out.relKKTres)
      out.relKKTres = 1;
   end
   if out.relKKTres < 1.0e-3 && max(out.constr_viol) < 1.0e-3
      out.termmsg = 3;
   end
   if out.relKKTres < 1.0e-6
      if max(out.constr_viol) < 1.0e-6
         out.termmsg = 1;
      else
         out.termmsg = 2;
      end
   end
else %%% case not a mere line search / unique eq.
   out.relKKTres = 1.0e-8; % (accuray in line search)
   y = zeros(0,1);         % no multipliers for nn==1 / unique eq.
end %%% of cases mere line search / unique eq.

out.Ain = Ain*Q2.';
out.Ae  = Ae;
out.bin = bin;
out.be  = be;
if isfield(out,'dt')
   out  = rmfield(out,'dt');
end
if isfield(out,'act_tol')
   out  = rmfield(out,'act_tol');
end
if isfield(out,'activefi')
   out  = rmfield(out,'activefi');
end
if isfield(out,'activeAi')
   out  = rmfield(out,'activeAi');
end

if out.termmsg == 1
    disp('KKT point found, about 6 digits accuracy');
end
if out.termmsg == 2
    disp('Infeasible nearly stationary point');
end
if out.termmsg == 3
    disp('KKT point found with 3 digits accuracy');
end
if out.termmsg == 4
    disp('Early termination, iteration limit reached');
end
if out.termmsg == 5
    disp('Termination without reaching convergence');
end
if out.termmsg == 6
    disp('Termination without improving over initial point');
end

end % of converged not {1 due to inconsistent linear equations} 

end % of min_fc








function [x,y,fx,out] = ...
                      mwdnonlin(f_obj,f_con,A,b,maxit,err0,desired_tol,p_l)
% subprogram for min_fc
%
% minimize  f_obj(x)  for  A*x <= b, f_eq(x) = 0, f_in(x) <= 0.
%
% where the output for f_con is [f_eq, f_in] (equalities and inequalities).
%
% Florian Jarre, Felix Lieder, April 2017
% Test Version with errors -- No guarantee of any kind is given!!!
%
% Mandatory input: 
%   f_obj,   a function handle for the objective function, R^n --> R
%   f_con,   a function handle for the constraints, R^n --> [R^me,R^mi]
%   A,       a constraint matrix, must have n columns and mA >= 0 rows,
%   b,       a right hand side with mA components
%   maxit,   a bound on the number of iterations
%   err0,    if the absolute error for a typical evaluation of
%            f_obj, f_con is known, this parameter can be set here, else 
%            it is estimated below (and used for the finite differences)
%   des_tol, desired stopping tolerance
%   p_l,     print level
%
% Output:
%   x:         an approximate local minimizer / stationary point
%   y:         the associated Lagrange multiplyer
%   fx:        the associated function value
%   out.iter:  number of iterations needed
%   out.fval:  number of function evaluations needed
%   out.relKKTres:   Norm of the gradient of the Lagrangian divided by
%                    norm of the gradient of the objective function
%   out.constr_viol = [norm(constraint violation, nonlinear equalities),
%                      zero -- nonexisting lin. equations here -- 
%                      norm(constraint violation, nonlinear inequalities),
%                      norm(constraint violation,    linear inequalities)];
%   out.dt            step size used for finite difference approximations
% 
%
% Algorithm:
% Trust region SQP algorithm with finite difference approximation 
% of the gradient using a PSB update for the Hessian of the Lagrangian.
%
% For the SQP subproblem, the Hessian is projected onto the orthogonal 
% complement of the active constraint gradients and then regularized 
% (to be positive semidefinite). 
%
% Each time the gradient is evaluated, the constraints are rescaled such 
% that the norms of the gradients of the constraints are at most one. 
% (This is done for numerical stability of the SOCP solver, and to avoid 
%  that reduction of some constraints dominates other constraints.)
% ``Currently redundant'' eq. constraints are being eliminated as well.
% Downside: The multipliers change, even their dimension is not constant.
%
% Starting point: zeros(n,1) where n is the number of columns of A




%**************************************************************************
% INITIALIZATION:
options.p_l = p_l;    % for passing to other subroutines
options.desired_tol = desired_tol;
out.termmsg = 0;      % no termination message yet
options.act_tol = 1.0e-7; % tolerance for setting active constraints

f         = @(x) f_obj(x);
fcon      = @(x) f_con(x);

[mA,n]    = size(A);  % mA = 0 if there are no linear inequalities
fn_it     = 0;        % counter for the number of function evaluations
outerit   = 0;        % counter for the number of outer iterations
converged = 0;

% project xact=0 onto the set satisfying the linear inequalities
if mA > 0
% First, normalize rows of A
   Arownormi = sum((A.*A),2); 
   tmpzero = Arownormi < 1.0e-12;
   tmpnzero = find(Arownormi >= 1.0e-12);
   if b(tmpzero) < 0
      disp('linear inequalities are ill-conditioned or inconsistent')
      converged = 1;
      out.termmsg = 2;
   else
      if ~isempty(tmpnzero)
         A = A(tmpnzero,:);
         b = b(tmpnzero);
         Arownormi = Arownormi(tmpnzero);
         mA = length(b);
      else
         A = zeros(0,n);
         b = zeros(0,1);
         Arownormi = Arownormi(0,1);
         mA = 0;
      end
      Arownormi = Arownormi.^(-1/2); % inverses of norms of rows of A
      b = b.*Arownormi;
      A = A .* (Arownormi*ones(1,n));
      if min(b) < 0
         in_linviol = max(-b); % initial violation of linear constraints
         likely_safe  = b >= 1.0e3*(1+in_linviol); % clearly satisfied 
         likely_unsafe = b < 1.0e3*(1+in_linviol); 
         % for numerical stability in mehrotramsocp first only try to 
         % satisfy the likely_unsafe inequalities
         Atmp = A(likely_unsafe,:);
         btmp = b(likely_unsafe);
         mAtmp = sum(likely_unsafe);
         Atmp = [eye(mAtmp),zeros(mAtmp,1),Atmp]; 
         Ktmp.l = mAtmp; Ktmp.q = n+1;
         ctmp = [zeros(mAtmp,1);1;zeros(n,1)];
         [xact,~,~,outm] = mehrotramsocp( Atmp,btmp,ctmp,Ktmp,options );
         xact = xact(mAtmp+2:end);
         if outm.done > 1
            disp(...
           'Failure to generate a point satisfying the linear constraints') 
            converged = 1;
            out.termmsg = 2;
         end
         surprize = A(likely_safe,:)*xact-b(likely_safe,:)>0;
         if sum(surprize) > 0 % try to satisfy all inequalities
            Atmp = [eye(mA),zeros(mA,1),A]; 
            Ktmp.l = mA; Ktmp.q = n+1;
            ctmp = [zeros(mA,1);1;zeros(n,1)];
            [xact,~,~,outm] = mehrotramsocp( Atmp,btmp,ctmp,Ktmp,options );
            xact = xact(mA+2:end);
            if outm.done > 1
               disp(...
           'Failure to generate a point satisfying the linear constraints') 
               converged = 1;
               out.termmsg = 2;
            end
         end
      else
         xact = zeros(n,1);
      end
   end
else
   xact = zeros(n,1);
end

fact = f(xact); 
[fe,fi] = fcon(xact);
fn_it = fn_it + 1;  
if isempty(fi)
%   mi = 0;
   fi = zeros(0,1); % make sure that empty function values have  
end                 % the right dimensions
if isempty(fe)
%   me = 0;
   fe = zeros(0,1);
end
x_00 = xact;
f_00 = fact; fe_00 = fe; fi_00 = fi; % at the end compare with these values

[dummy1,dummy2] = size(fact);
if dummy1 ~= 1 || dummy2 ~= 1
    error('output of the objective function not a real number')
end

% initialize the sizes me and mi of the output of fe and fi
[me,dummy2] = size(fe);  % number of (nonlinear)   equality constraints
if me > 0 && dummy2 ~= 1
    error('output of the fe not a column vector')
end
[mi,dummy2] = size(fi);  % number of (nonlinear) inequality constraints
if mi > 0 && dummy2 ~= 1
    error('output of the fi not a column vector')
end

% END OF INITIALIZATION:
%**************************************************************************






%**************************************************************************
% DETERMINE FINITE DIFFERENCE STEP LENGTH AND FIRST DERIVATIVE APPROX'NS:
dt  = (1+norm(xact))*eps^(.65); % finite difference for the first step 
dtt = 2*dt;
Dfe = zeros(me,n);       % storage allocation, also for me = 0 (!)
Dfi = zeros(mi,n);       % storage allocation, also for mi = 0 (!)


% estimate the error of the function evaluation of f, fe and fi near xact
U = rando(n); 

fval = zeros(n,2);
g = zeros(n,1); % gradient of f, column vector with n entries 
if me > 0
   fep  = zeros(me,n);
   fepp = zeros(me,n);
end
if mi > 0
   fip  = zeros(mi,n);
   fipp = zeros(mi,n);
end

err = eps; % lower estimate for the error in a function evaluation

for i = 1:n % forward difference gradient and error estimate for f & fcon
   xup  = xact+dt *U(:,i);
   xupp = xact+dtt*U(:,i);
   fval(i,1) = f(xup );
   fval(i,2) = f(xupp);
   err = max(err, abs(fval(i,1)-0.5*(fact+fval(i,2))));
   g(i) = (fval(i,2)-fact)/dtt; % actually no need to store fval(:,1:2)
   [feup ,fiup ] = fcon(xup );
   [feupp,fiupp] = fcon(xupp);
   
   if me > 0
      fep(:,i)  = feup;
      fepp(:,i) = feupp;
      err = max(err, norm(fep(:,i)-0.5*(fe+fepp(:,i)),Inf));
      Dfe(:,i) = (fepp(:,i)-fe)/dtt;
   end
   if mi > 0
      fip(:,i)  = fiup;
      fipp(:,i) = fiupp;
      err = max(err, norm(fip(:,i)-0.5*(fi+fipp(:,i)),Inf));
      Dfi(:,i) = (fipp(:,i)-fi)/dtt;
   end
end
fn_it = fn_it+2*n+1;

if err0 < Inf % relate err to given input estimate err0
   if err0 < err
      disp('either the function has large second derivative or')
      disp('options.err was too optimistic; it is being increased')
   else
      if err0 > 10*err
         disp('options.err is more pessimistic than estimate generated')
         disp('by the algorithm; a more optimistic estimate is used');
         err = 10*err;
      else
         err = err0;
      end
   end
end
dt      = 0.1*err^(1/3); % finite difference for the rest of the algorithm
out.dt  = dt;

% possibly redo the gradient (and initialize the Hessian)
H = zeros(n);
if err < 0.002*eps^(.65) % 2*eps^(.65) was used for the finite difference
   g = U*g;              % hence, about >= 3 digits accuracy
   if me > 0
      Dfe = Dfe*U.';
   end
   if mi > 0
      Dfi = Dfi*U.';
   end
else
   [g,Dfe,Dfi] = update_firstder(f,fcon,xact,dt);
end
Dfeu = Dfe; % unscaled version
Dfiu = Dfi; 
dummy = norm(Dfe,'fro')+norm(Dfi,'fro');
if isnan(dummy) || dummy == Inf
   disp('constraints not finite at or very close to the initial point')
   converged = 1;
end

[fe,fi,Dfe,Dfi,me,mi,spar,conv2] = rescale(fe,fi,Dfe,Dfi,p_l);
converged = max(converged,conv2);
bmAx = b-A*xact;

xold    = xact; % used in the stopping test, also when there is no update
                % of xact or of the derivatives

% DONE WITH FINITE DIFFERENCE STEP LENGTH AND FIRST DERIVATIVE APPROX'NS
%**************************************************************************




bigM = 1000;
lastddxnorm = 0.1; % Arbitrary estimate of last SO correction
options.residual = 1;
if converged == 0  %%% Estimate the Lagrange multipliers
options.act_tol = max(5.0e-7,2*lastddxnorm); % tolerance for setting act.
if mi > 0
   options.activefi = fi > -options.act_tol;
else
   options.activefi = zeros(0,1); 
end
if mA > 0
   options.activeAi = bmAx < options.act_tol;
else
   options.activeAi = zeros(0,1); 
end
bmAx             = b-A*xact;             % should always be nonnegative
[ye,yi,yA,~] = findlagmult(g,Dfe,Dfi,fi,A,bmAx,options);

earlystop = norm(g+Dfe.'*ye+Dfi.'*yi+A.'*yA) < desired_tol && ...
            norm(fe)+norm(max(0,fi)) < desired_tol;
% KKT conditions almost satisfied, but active constraints may be wrong

if earlystop
   if mi > 0
      options.activefi = fi > -desired_tol;
   end
   if mA > 0
      options.activeAi = bmAx < desired_tol;
   end
   [ye,yi,yA,converged2] = ...
               findlagmult(g,Dfe,Dfi,fi,A,bmAx,options);
   earlystop = norm(g+Dfe.'*ye+Dfi.'*yi+A.'*yA) < desired_tol && ...
               norm(fe)+norm(max(0,fi)) < desired_tol;
   converged2 = max(converged2, earlystop);
   if converged2 == 1
      if p_l >= 1
         disp('stop since starting point is (nearly) stationary');
      end
      out.termmsg = 1;
   end
   converged = converged || converged2;
end
bigM = 10*max(sqrt(me+mi+mA),(norm(ye)^2+norm(yi)^2+norm(yA)^2)^.5);
end %%% of Estimate the Lagrange multipliers




delta = 1.0; % initial trust region radius 

if me > n
   disp('more nonlinear equality constraints than degrees of freedom')
   disp('initial point either satisfies the Fritz-John conditions') 
   disp('or it is an infeasible stationary point')
   converged = 1;
   out.termmsg = 2;
end




%**************************************************************************
% MAIN LOOP
while  converged == 0  &&  outerit < maxit  % MAIN LOOP 
   outerit = outerit + 1;

   
% Project the Hessian of the Lagrangian and make it positive definite:
   if isnan(norm(H,'fro')) || norm(H,'fro') == Inf
      H = 1.0e-6*eye(n); 
   end
   eigH = eig(H);
   if min(eigH) <= 1.0e-6*(1+norm(H,'fro'))
      actfi = yi > 0;
      actAi = yA > 0;
      Atmp  = full([Dfe;Dfi(actfi,:);A(actAi,:)]); % empty matrices 
                                     % sometimes are sparse empty matrices
      options.Hactfi = actfi; % Remember which constraints were considered
      options.HactAi = actAi; % active when forming Hsqp
      [mtmp,~] = size(Atmp);
      if mtmp > 0
         [Q,~,~] = qr(Atmp.','vector');
         Q2 = Q(:,mtmp+1:end);
         Hsqp = Q2*(Q2.'*H*Q2)*Q2.';
         Hsqp = 0.5*(Hsqp+Hsqp.');
         lambdamin = min(0,min(eig(Hsqp)));
         Hsqp = Hsqp + (1.0e-6*(norm(Hsqp,'fro')+1)-lambdamin)*eye(n);
      else
         Hsqp = H + (1.0e-6*(norm(H,'fro')+1)-min(0,min(eigH)))*eye(n);
      end
   else
      Hsqp = H;
      actfi = yi < -1;
      actAi = yA < -1;
      options.Hactfi = actfi; % No constraints were considered
      options.HactAi = actAi; % active when forming Hsqp
   end
   if isnan(norm(Hsqp,'fro')) || norm(Hsqp,'fro') == Inf
      Hsqp = 1.0e-6*eye(n);
   end

   
% Compute SQP trust region correction
   options.delta      = delta;
   [dx,ye,yi,yA,~,outs] = sqpsocp(g,Hsqp,Dfe,fe,Dfi,fi,A,b-A*xact,options);
   options.residual = max(norm(g+Dfe.'*ye+Dfi.'*yi+A.'*yA),...
                         (norm(fe)^2+norm(max(0,fi))^2)^.5);
   converged = max(converged, outs.converged);
   if out.termmsg == 0 && outs.converged == 1
      out.termmsg = 2; 
   end

   
   if converged == 0 % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Possibly compute a second order correction   
   if outs.viol_opt <= 1.0e-8*max( 1, norm(fe) + norm(fi) ) && outerit > 1
      [fenewu,finewu] = fcon(xact+dx);
      fn_it = fn_it + 1;  
      [fenew,finew] = rescalef(fenewu,finewu,spar);
      bmAxdx = b-A*(xact+dx);
      [ddx,out_so]  = so_corr(fenew,Dfe,finew,Dfi,bmAxdx,A,options);
      if out_so.mu > 0.1
         converged = 1;
         if p_l > 0
            disp('stop since (shifted) MFCQ appears to be nearly violated')
         end
         out.termmsg = 2; 
      end
   else
      ddx = zeros(n,1);
   end
   lastddxnorm = norm(ddx); % Arbitrary estimate of last SO correction
   options.act_tol = max(5.0e-7,2*lastddxnorm);

   
% Compute range for possible step lengths
   if outerit == 1
      l_max = Inf; % in the first step no SO-correction and allow l_max > 1
   else
      l_max = 1; 
   end
   if mA >0 && outerit == 1
      crit_vec  = A*dx;
      crit_comp = crit_vec > 0;
      btmp = b-A*(xact+dx+ddx);
      if min(btmp) < 0
         if p_l > 0
            disp('second order corr. violates feasibility of lin. constr.')
            tmp1 = sprintf('violation : %0.5g ', min(btmp) );
            disp(tmp1)
         end
         btmp = max(1.0e-10,btmp); % assume the above are rounding errors
      end
      if sum(crit_comp) > 0
         l_max = 1+min(btmp(crit_comp)./crit_vec(crit_comp));
      end
   end
   l_min = 0; % allow only steps in positive direction of dx
   
   
% line search along the curve xact + lambda*dx+min(1,max(1,lambda))^2*ddx:
   optL.lb   = l_min;     % lower bound for curvi-linear search
   optL.ub   = l_max;
   optL.xact = 1;         % starting point for curvi-linear search
   optL.tol  = 1.0e-4;
   if outerit <= 2
       optL.tol = 1.0e-8;
   %else
   %    optL.xact = 0.99; % this does not seem beneficial
   end
   meritold = fact + bigM*(norm(fe)^2+norm(max(fi,0))^2)^.5;
   fcons     = @(x) f_cons(@(u,v) f_con(u), x,spar); % scaled constraints

   [lambda,merit,~,~,out1] = ...
                   mwd11(@(t) curvimerit(f,fcons,t,xact,dx,ddx,bigM),optL);
                                            
   if merit == Inf || merit >= meritold
      if options.p_l > 0
         disp('inconsistency in the line search for an SQP step')
         tmp = sprintf('merit function at old iterate %0.5g',meritold);
         disp(tmp)
         tmp = sprintf(...
                    'difference to merit function at new iterate %0.5g',...
                     merit-meritold);
         disp(tmp)
      end
      lambda = 0;
   end
   xnew = xact + lambda*dx + min(1,max(0,lambda))^2*ddx;
   fn_it = fn_it + out1.iter; % number of function evaluations
   if options.p_l > 1
      tmp = sprintf(...
'delta, norm(dx), and step length in line search %0.5g , %0.5g , %0.5g',...
      delta,norm(dx),lambda);
      disp(tmp);
   end
   tmp = min(b-A*xnew)/(1+norm(b));
   if tmp < -1.0e-8 % line search did not maintain linear inequalities-fail
      converged = 1;
      out.termmsg = 5; 
      tmp1 = min(b-A*xact);
      if options.p_l > 0
         tmp2 = sprintf(...
          'violation of linear constraints before SQP step %0.5g ', -tmp1);
         disp(tmp2)
         tmp2 = sprintf(...
          'violation of linear constraints after  SQP step %0.5g ', -tmp );
         disp(tmp2)
      end
      xnew = xact;
   end
   if lambda == 0 || ( abs(lambda) < 1.0e-3 && outerit > 1 )
                   % the very first trust region radius may be very poor
      if converged == 0 && options.p_l > 0
         disp('stop due to very short step along the SQP solution')
      end
      converged = 1;
   else
      xold    = xact; % accept the line search
      xact    = xnew;
   end
% end of line search along xact + lambda*dx+min(1,max(1,lambda))^2*ddx

   
 % Reduce or increase the trust region radius
   if lambda < 0.9 && lambda > 0
      delta = lambda*norm(dx);
   end
   if lambda >= 0.9 && converged == 0
      delta = max(delta,2*lambda*norm(dx));
   end
   % reduce delta for numerical stability (unnecessarily???) 
   delta = min(delta,10*norm(dx));
   end % of converged == 0 % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   
   if converged == 0 % converged may have been changed above
% update objective gradient and constraint Jacobians 
      gold    = g;
      Dfeoldu = Dfeu;
      Dfioldu = Dfiu;
      [g,Dfeu,Dfiu] = update_firstder(f,fcon,xact,dt);
      dummy = norm(Dfe,'fro')+norm(Dfi,'fro');
      if isnan(dummy) || dummy == Inf
         disp('constraints not finite at current iterate')
         converged = 1;
      end
      bmAx    = b - A*xact;
      fetmp = fe; fitmp = fi;
      [fe,fi] = fcon(xact);
      if p_l > 1
         tmp = sprintf('fe,fi before/after SQP %0.5g %0.5g %0.5g %0.5g',...
              norm(fetmp),norm(fe),norm(max(0,fitmp)),norm(max(0,fi)));
         disp(tmp);
      end
      fact = f(xact); 
      fn_it = fn_it + 2*n+1;
      largerDfiold = spar.largerDfi;
      [fe,fi,Dfe,Dfi,~,~,spar,conv2] = rescale(fe,fi,Dfeu,Dfiu,p_l);
      converged = max(converged, conv2);
      if out.termmsg == 0 && conv2 == 1
         out.termmsg = 2; 
      end

% Find Lagrange multiplyers (muliplyers from the SQP sub problem may be 
% unreliable due to update of the derivative / trust region constraint)
      if converged == 0
         if norm(spar.largerDfi-largerDfiold) == 0
            options.activefi = outs.activefi;
         else
            options.activefi = fi > -max(5.0e-7,2*lastddxnorm);
         end
         if mA > 0
            options.activeAi = outs.activeAi;
         end
         [ye,yi,yA,converged2] = findlagmult(g,Dfe,Dfi,fi,A,bmAx,...
                                             options);
         converged = max(converged,converged2); % zero or one
         bigM  = max(bigM,10*(norm(ye)^2+norm(yi)^2+norm(yA)^2)^.5);

% update the Hessian of the Lagrangian 
         [Dfeolds,Dfiolds] = rescaleDf(Dfeoldu,Dfioldu,spar);
         gLold = gold+Dfeolds.'*ye+Dfiolds.'*yi; 
                  
% gradient of Lagrangian at xold with new multiplyers
% NOTE: The linear constraints are ignored
% since they give no contribution to the Hessian of the Lagrangian
         gL = g+Dfe.'*ye+Dfi.'*yi; 
         H = update_hess(xact,xold,gL,gLold,H,dt); 
      end
   end
% of update objective gradient and constraint Jacobians

   if max(A*xact -b) > 1.0e-8*(1+norm(b))
      disp('Warning, linear constraints are not satisfied ')
      %keyboard
   end 
   if p_l > 1
      disp(' ')
   end
   if norm(xold-xact) < desired_tol && outerit > 1
       converged = 1;
   end
end 
% end of MAIN LOOP
%**************************************************************************
[fe,fi] = fcon(xact);
fn_it = fn_it + 2*n+1;
feu = fe;   % unscaled version
fiu = fi; 
[fe,fi,Dfe,Dfi,~,~,~,conv2] = rescale(fe,fi,Dfeu,Dfiu,p_l);
if out.termmsg == 0 && conv2 == 1
   out.termmsg = 2; 
end
if p_l >= 2
   disp('main loop completed ') 
end


x  = xact;
if ~exist('ye','var')
   ye = zeros(me,1);
   Dfe = zeros(me,n);
end
if ~exist('yi','var')
   yi = zeros(mi,1);
   Dfi = zeros(mi,n);
   options.activefi = yi > 0; 
end
if ~exist('yA','var')
   yA = zeros(mA,1);
   options.activeAi = yA > 0; 
end
y  = [ye;yi;yA];
fx = f(x);
[ye,yi,yA,~] = findlagmult(g,Dfe,Dfi,fi,A,bmAx,options);
act_fi = 1:length(spar.largerDfi);
act_fi = act_fi(spar.largerDfi);
act_fi = act_fi(options.activefi).';
act_fi1 = zeros(length(spar.largerDfi),1);
act_fi1(act_fi) = 1;
out.activefi = logical(act_fi1);
%out.activefi = options.activefi;
out.activeAi = options.activeAi;

%keyboard

out.relKKTres = norm(g+Dfe.'*ye+Dfi.'*yi+A.'*yA)/(norm(g)+1.0);
if isnan(out.relKKTres)
   out.relKKTres = 1;
end
out.constr_viol = [norm(fe),0,norm(max(fi,0)),norm(max(A*xact-b,0))];
out.iter  = outerit;
out.fval  = fn_it;

if fx   + bigM*(norm(feu  )^2+norm(max(fiu,  0))^2)^.5 >= ...
   f_00 + bigM*(norm(fe_00)^2+norm(max(fi_00,0))^2)^.5
   if norm(x-x_00) > 0 && out.termmsg == 0
      out.termmsg = 6;
   end
   x = x_00;
   fx = f_00;
   y = [];
   out.relKKTres = 1;
   out.constr_viol =[norm(fe_00),0,...
                     norm(max(fi_00,0)),norm(max(A*x_00-b,0))];
   if p_l > 0
      disp('no improvement over initial point was found ');
   end
end

if out.termmsg == 0
   finalacc = max(out.relKKTres,max(out.constr_viol));
   if finalacc > 1.0e-3 
      out.termmsg = 5; 
      %keyboard
   end
   if outerit >= maxit
      out.termmsg = 4; 
   end
   if finalacc <= 1.0e-3
      out.termmsg = 3; 
   end
   if finalacc <= 1.0e-6
      out.termmsg = 1;
   end
end
out.act_tol = options.act_tol;


end % of mwdnonlin 








function [fen,fin,Dfen,Dfin,me,mi,spar,converged] = ...
                                     rescale(fe,fi,Dfe,Dfi,p_l)
% In case that the equalities have linearly dependent gradients find an 
% equivalent reduced set of equality constraints or decide that the
% linearized constraints cannot be satisfied.
% In the latter case, set converged = 1.
% Also normalize the inequalities to norm at most 1.
% Store the scaling parameters in spar
                                 
[me,n]     = size(Dfe);
mi         = length(fi);
Dferownorm = sum((Dfe.*Dfe),2); 
Dferownorm = Dferownorm.^(1/2);        % norms of rows of Dfe
%Dferownorm = ones(size(Dferownorm));  % to check influence of scaling
spar.Dferownorm = Dferownorm;
Dfirownorm = sum((Dfi.*Dfi),2); 
Dfirownorm = Dfirownorm.^(1/2);        % norms of rows of Dfi
%Dfirownorm = ones(size(Dfirownorm));  % to check influence of scaling
spar.Dfirownorm = Dfirownorm;
converged  = 0;
if sum(Dferownorm) == Inf || isnan(sum(Dferownorm))
   converged = 1;
end
if sum(Dfirownorm) == Inf || isnan(sum(Dfirownorm))
   converged = 1;
end

if isempty(fi)
   mi = 0;
   fi = zeros(0,1); % make sure that empty function values have  
end                 % the right dimensions
if isempty(fe)
   me = 0;
   fe = zeros(0,1);
end

if me > 0  
   % first eliminate equality constraints with tiny gradients
   tinyDfe   = Dferownorm <  1.0e-8; % zero up to truncation errors
   largerDfe = Dferownorm >= 1.0e-8;
   spar.tinyDfe   = tinyDfe;
   spar.largerDfe = largerDfe;
   if sum(tinyDfe) > 0
      unsatisfyable = abs(fe(tinyDfe))-1.0*Dferownorm(tinyDfe) > 0;
      % here, up to truncation error, Dfe is zero but fe is not
      if sum(unsatisfyable) > 0
         disp('failure to satisfy nonlinear equality constraints')
         converged = 1;
      else % entries of abs(fe(tinyDfe)) are less than 1.0e-8 ==> omitt 
         Dfe        = Dfe(largerDfe,:);
         fe         = fe(largerDfe);
         Dferownorm = Dferownorm(largerDfe);    
         me         = sum(largerDfe); % now, me could be zero
      end
   end
end

spar.Q = eye(me);

if me > 0 && converged == 0
   Dfe = Dfe .* ((1./Dferownorm)*ones(1,n));
   % tiny Dferownorms (including zero) have been eliminated above
   fe  = fe  .*  (1./Dferownorm);
   
   [Q,R,~] = qr(Dfe,0); % only the first columns of Q are computed
   if me == 1
      DR = R(1,1); % if me == 1 then diag(R) is a matrix with diagonal R
   else
      DR = diag(R);
   end
   positiveR = abs(DR) >  1.0e-8;
   tinyR     = abs(DR) <= 1.0e-8;
   spar.tinyR     = tinyR;
   spar.positiveR = positiveR;
   if sum(tinyR) > 0
      %disp( ' *** nearly lin. dep. nonlin. eq. constr. *** ')
      spar.Q = Q;
      tmp = Q.'*fe;
      if norm(tmp(tinyR)) >= 1.0e-8
         disp('failure to satisfy nonlinear equality constraints')
         converged = 1;
      else % entries associated with tinyR are less than 1.0e-8 ==> omitt
         if p_l > 0
            disp('nearly linearly dependent nonlinear equations')
         end
         Dfe = Q.'*Dfe;
         fe  = Q.'*fe;
         Dfe = Dfe(positiveR,:);
         fe  =  fe(positiveR);
         me  = sum(positiveR);
         converged = 1;
      end
   end
else
    spar.positiveR = zeros(0,1);
end

if mi > 0 && converged == 0
   % first eliminate inequality constraints with tiny gradients
   tinyDfi   = Dfirownorm <  1.0e-8; % zero up to discretization error
   largerDfi = Dfirownorm >= 1.0e-8;
   spar.tinyDfi   = tinyDfi;
   spar.largerDfi = largerDfi;
   if sum(tinyDfi) > 0
      unsatisfyable = fi(tinyDfi)-1.0*Dfirownorm(tinyDfi) > 0;
      if sum(unsatisfyable) > 0
         disp('failure to satisfy nonlinear inequality constraints')
         converged = 1;
      else % entries of fi(tinyDfi) are less than 1.0e-8 ==> omitt 
         Dfi        = Dfi(largerDfi,:);
         fi         = fi(largerDfi);
         Dfirownorm = Dfirownorm(largerDfi);    
         mi         = sum(largerDfi); % now, mi could be zero
      end
   end
else
   spar.largerDfi = zeros(0,1);
end

if mi > 0 && converged == 0
   Dfi = Dfi .* ((1./Dfirownorm)*ones(1,n));
   fi  = fi  .*  (1./Dfirownorm);
end

fen       = fe;
fin       = fi;
Dfen      = Dfe;
Dfin      = Dfi;

end








function [fen,fin] = rescalef(fe,fi,spar)
% scale fe and fi as given by the parameters in spar
                                 
me     = length(fe);
mi     = length(fi);

if me > 0  
   if ~isempty(spar.tinyDfe)
      fe         = fe(spar.largerDfe);
      me         = sum(spar.largerDfe); % now, me could be zero
   end
end

if me > 0 
   fe  = fe  .*  (1./spar.Dferownorm(spar.largerDfe));
   if ~isempty(spar.tinyR) 
      if sum(spar.tinyR) > 0
         fe  = spar.Q.'*fe;
         % me  = length(positiveR);
      end
   end
end

if mi > 0 
   if ~isempty(spar.tinyDfi)
      fi         = fi(spar.largerDfi);
      mi         = sum(spar.largerDfi); % now, mi could be zero
   end
end

if mi > 0 
   fi  = fi  .*  (1./spar.Dfirownorm(spar.largerDfi));
end

fen       = fe;
fin       = fi;

end








function [Dfen,Dfin] = rescaleDf(Dfe,Dfi,spar)
% scale Dfe and Dfi as given by the parameters in spar
                                 
[me,n]     = size(Dfe);
[mi,~]     = size(Dfi);

if me > 0  
   if ~isempty(spar.tinyDfe)
      Dfe        = Dfe(spar.largerDfe,:);
      me         = sum(spar.largerDfe); % now, me could be zero
   end
end

if me > 0 
   Dfe  = Dfe  .*  ((1./spar.Dferownorm(spar.largerDfe))*ones(1,n));
   if ~isempty(spar.tinyR)
      if sum(spar.tinyR) > 0
         Dfe  = spar.Q.'*Dfe;
         % me  = length(positiveR);
      end
   end
end

if mi > 0 
   if ~isempty(spar.tinyDfi)
      Dfi        = Dfi(spar.largerDfi,:);
      %mi         = length(spar.largerDfi); % now, mi could be zero
   end
end

if mi > 0 
   Dfi  = Dfi  .*  ((1./spar.Dfirownorm(spar.largerDfi))*ones(1,n));
end

Dfen       = Dfe;
Dfin       = Dfi;

end








function [fen,fin] = f_cons(f_con, x,spar) % scaled constraints
% same as rescalef directly applied to f_con

[fe,fi] = f_con(x);
me      = length(fe);
mi      = length(fi);

if me > 0  
   if ~isempty(spar.tinyDfe)
      fe         = fe(spar.largerDfe);
      me         = sum(spar.largerDfe); % now, me could be zero
   end
end

if me > 0 
   fe  = fe  .*  (1./spar.Dferownorm(spar.largerDfe));
   if sum(spar.tinyR) > 0
      fe  = spar.Q.'*fe;
      % me  = length(positiveR);
   end
end

if mi > 0 
   if ~isempty(spar.tinyDfi)
      fi         = fi(spar.largerDfi);
      mi         = sum(spar.largerDfi); % now, mi could be zero
   end
end

if mi > 0 
   fi  = fi  .*  (1./spar.Dfirownorm(spar.largerDfi));
end

fen       = fe;
fin       = fi;

end






function [dx,out] = so_corr(fe,Dfe,fi,Dfi,bmAx,A,options)
% Solve for Second Order correction
% min { ||dx|| | fe + Dfe*dx = 0, fi + Dfi*dx <= 0, A*dx <= bmAx }
% bmAx: ``b minus A x''
% Do this only when viol_opt is zero (or sufficiently small) !!!

[me,n] = size(Dfe);
[mi,~] = size(Dfi);
[mA,~] = size(A);

if me+mi > 0
   Atmpfull = [zeros(me,mi+mA+1),Dfe
               eye(mi),zeros(mi,mA+1),Dfi
               zeros(mA,mi),eye(mA),zeros(mA,1),A];
   btmp = [-fe;-fi;bmAx];
   ctmp = [zeros(mi+mA,1);1;zeros(n,1)];
   Ktmp.l = mi+mA;
   Ktmp.q = n+1;
   if norm(btmp) == 0
      dx           = zeros(n,1);
      out.mu       = 0;
      out.done     = 1;
      out.residual = 0;
   else
      [dx,~,~,out] = mehrotramsocp( Atmpfull,btmp,ctmp,Ktmp,options );
      dx = dx(mi+mA+2:end);
      if out.done > 1 && options.p_l > 0
         disp(' inaccurate solution of MSOCP subproblem in so_corr ')
      end
      if options.p_l > 1
         disp('second order correction computed via msocp') 
      end
   end
else
   dx           = zeros(n,1);
   out.mu       = 0;
   out.done     = 1;
   out.residual = 0;
end

end








function [ye,yi,yA,converged] = findlagmult(g,Dfe,Dfi,fi,A,bmAx,options)
% determine approximate Lagrange multiplyers at xact such that 
% || g + Dfe'*ye + Dfi'*yi + A'*yA || is small 

[me,n]  = size(Dfe);
[mi,~]  = size(Dfi);
[mA,~]  = size(A);
if mi > 0
   activefi = options.activefi;
else
   activefi = zeros(0,1); 
end
if mA > 0
   activeAi = options.activeAi;
else
   activeAi = zeros(0,1); 
end

converged       = 0; 
mAi = sum(activefi) + sum(activeAi);

if me >= n
   yi = zeros(mi,1);
   yA = zeros(mA,1);
   if isnan(norm(Dfe,'fro')) || norm(Dfe,'fro') == Inf
      ye = zeros(me,1);
   else
      ye = - pinv(Dfe*Dfe.')*(Dfe*g); 
   end
   converged = 1;
end

if mAi > 0 && converged == 0
    
   Atmpi = [Dfi(activefi,:);A(activeAi,:)];
   % btmpi = [Dfi(activefi,:);A(activeAi,:)]*xact-[zeros(mi,1);b(activeAi)]
   % constraints Atmpi*xact <= btmpi may be active 
   if me > 0 % eliminate equality constraints before computing multipliers
      [Q,R,eqr] = qr(Dfe,'vector'); 
      T1    = 1:me;
      T2    = me+1:n;
      R1 = R(:,T1);
      dr1 = abs(diag(R1));
      if min(dr1)/max(dr1) < 1.0e-8 || max(dr1) == 0
         converged = 2; %(nearly) stationary point (feasible or infeasible)
         Dfeorig   = Dfe;
         meorig    = me;
         Qorig     = Q;
         %eqrorig   = eqr;
         Dfe       = Q'*Dfe;
         largeR1   = dr1 >= 1.0e-8*max(dr1);
         Dfe       = Dfe(largeR1,:);
         [me,~]    = size(Dfe);
         [Q,R,eqr] = qr(Dfe,'vector');
         T1        = 1:me;
         T2        = me+1:n;
         R1        = R(:,T1);
         R1it      = pinv(R1.'); 
      else
         R1it = inv(R1.');
      end
      Atmpi = Atmpi(:,eqr);
      R2sT  = R(:,T2).'*R1it;
      gtmp  = g(eqr);
      Atmpfull = [Atmpi(:,T2).'-R2sT*Atmpi(:,T1).',...
                  zeros(n-me,1), R2sT, -eye(n-me)];
      btmp = -gtmp(T2)+R2sT*gtmp(T1);
      ctmp = [zeros(mAi,1);1;zeros(n,1)];
   else
      Atmpfull = [Atmpi.',zeros(n,1),-eye(n)];
      btmp = -g;
      ctmp = [zeros(mAi,1);1;zeros(n,1)];
   end
   Ktmp.l = mAi;
   Ktmp.q = n+1;
   if norm(btmp) > 0
      [ysact,~,~,out] = mehrotramsocp( Atmpfull,btmp,ctmp,Ktmp,options );
   % solution with components  [yi(activefi);yA(activeAi);t;s]
   % where norm(s) is minimized
      if out.done > 1 && options.p_l > 0
         disp(' inaccurate solution of MSOCP subproblem in findlagmult ')
      end
      if options.p_l > 1
         disp('Lagrange multipliers computed via msocp') 
      end
      yi = zeros(mi,1);
      yA = zeros(mA,1);
      yi(activefi) = ysact(1:sum(activefi));
      yA(activeAi) = ysact(sum(activefi)+1:mAi);
      if me > 0
         s1 = ysact(mAi+2:mAi+me+1);
         g1 = gtmp(1:me);
         ye = Q*(R1it*(s1-g1-Atmpi(:,T1).'*ysact(1:mAi))); 
      else
         ye = zeros(0,1);
      end
      yA = max(0,yA); % to eliminate rounding/truncation errors
      yi = max(0,yi);
   
      Dfirownorm = sum((Dfi.*Dfi),2); 
      Dfirownorm = Dfirownorm.^(1/2); % norms of rows of Dfi
      hiddeninactivefi = yi.*Dfirownorm < -0.1*fi*norm(g);
      yi(hiddeninactivefi) = 0;
      hiddeninactiveA  = yA < 0.1*bmAx*norm(g);
      yA(hiddeninactiveA)  = 0;
   else
      yA = zeros(mA,1);
      yi = zeros(mi,1);
      ye = zeros(me,1);
   end
   
end 
if converged == 2 % reintroduce the deleted contraints and rotate back
   converged = 1;
   tmp = ye;
   ye = zeros(meorig,1);
   ye(largeR1) = tmp;
   ye = Qorig.'*ye;
   Dfe = Dfeorig;
   me = meorig;
end
if mAi == 0 && converged == 0
    
   yi = zeros(mi,1);
   yA = zeros(mA,1);
   if me > 0
      ye = - (Dfe*Dfe.')\(Dfe*g);
   else
      ye = zeros(0,1);
   end
   
end

end








function [gnew,Dfenew,Dfinew] = update_firstder(f,fcon,xnew,dt)
% update the first derivative of f and of the Lagrangian
%
% Input:
%        a function handle f
%        a function handle fcon
%        xnew: the point at which the derivatives are searched for
%        dt: step length for finite difference approximation
%
% Output:
%        gnew:  approximation of gradient of f at xnew
%        Dfenew approximation of Jacobian D fe at xnew
%        Dfinew approximation of Jacobian D fi at xnew
%
% Use finite differences to update gradient and Jacobians
%
% Test Version with errors -- No guarantee of any kind is given!!!
%

n = length(xnew);
[fenew,finew] = fcon(xnew); me = length(fenew); mi = length(finew);
dtt = 2*dt;

U = rando(n); 

fval = zeros(n,2); % just to allocate storage
if me > 0
   fep  = zeros(me,n); % just to allocate storage
   fepp = zeros(me,n);
end
if mi > 0
   fip  = zeros(mi,n);
   fipp = zeros(mi,n);
end

for i = 1:n
   xdown = xnew-dt*U(:,i);
   xup   = xnew+dt*U(:,i);
   fval(i,1) = f(xdown);
   fval(i,2) = f(xup);
   [fedown,fidown] = fcon(xdown);
   [feup  ,fiup  ] = fcon(xup  );
   if me > 0
      fep(:,i)  = fedown;
      fepp(:,i) = feup;
   end
   if mi > 0
      fip(:,i)  = fidown;
      fipp(:,i) = fiup;
   end
end

gnew   = U*(fval(:,2)-fval(:,1))/dtt;
if me > 0
   Dfenew = ((fepp-fep)* U.')/dtt;
else
   Dfenew = zeros(0,n);
end
if mi > 0
   Dfinew = ((fipp-fip)* U.')/dtt;
else
   Dfinew = zeros(0,n);
end

end







function Hnew = update_hess(xact,xold,gLact,gLold,H,dt)
% update the Hessian of the Lagrangian
%
% Input:
%        xnew: the point at which the derivatives are searched for
%        xact: the point at which approximations of the deriv. are given
%        ye multiplyer for nonlinear equality constraints
%        yi multiplyer for nonlinear inequality constraints
%        gLact approximation of gradient of Lagrangian at xact (new yi, ye)
%        gLnew approximation of gradient of Lagrangian at xnew (new yi, ye)
%        H  approximate Hessian of Lagrangian at xact
%        dt: step length for finite difference approximation
%
% Output:
%        Hnew: approximation of Hessian of Lagrangian at xnew
%
% Use PSB to update the Hessian
%
% Test Version with errors -- No guarantee of any kind is given!!!
%

%n = length(xact);

dx = xact-xold; dxdx = sum(dx.^2);

if dxdx > 0.1*dt^2 % PSB update only for sufficiently long steps; else
                   % the finite difference error may falsify the update
   dg = gLact-gLold;
   ddg = dg-H*dx;
   dH = (ddg*dx.'+dx*ddg.')*(1.0/dxdx) - ((ddg.'*dx/dxdx^2)*dx)*dx.';
   H = H + dH;
end


Hnew = 0.5*(H+H.');

dummy = norm(Hnew,'fro');
if isnan(dummy) || dummy == Inf
   disp('Hessian not finite at current iterate')
end
%Hnew = zeros(size(Hnew)); % For testing steepest descent

end








function U = rando(n)
% generate a random orthogonal n x n - matrix U
% XXX made determininstic for reproducability
if n <= 0
   error('check input for rando');
end

if n <= 1
   %U = randi(2,1,1); % a random number either 1 or 2
   %U = 2*U-3;        % a random number either 1 or -1
   U = 1;
else
   %a = randn(n);[U,~]=qr(a);
   U = eye(n); 
end

end












function [dx,ye,yi,yA,ydelta,out] = sqpsocp(g,H,Dfe,fe,Dfi,fi,A,b,options)
%
% Approximately solve the tust region SQP problem 
%
% minimize g'*dx + 0.5*dx'*H*dx 
%
% s.t. Dfe*dx   = -fe
%      Dfi*dx  <= -fi
%       A*dx   <= b
%      ||dx||  <= delta  (Euclidean norm)
%
% It is assumed that H is regularized to be positive semidefinite
% and that b >= 0.
% Sparsity is not exploited !!!
%
%
% Approach:
% If ||f_E||+||f_I|| > 1.0e-8, first solve the problem
%
% minimize ||Dfe*dx+fe||^2 + ||(Dfi*dx+fi)_+||^2
% s.t. A*dx   <= b
%      ||dx|| <= 0.9*delta,
%
% the optimal value be denoted by viol_opt^2 
% (viol_opt: smallest possible violation for trust region radius 0.9*delta)
% Else (when ||f_E||+||f_I|| <= 1.0e-8) viol_opt := 0.
% 
% Then, minimize g'*dx + 0.5*dx'*H*dx                           (***)
% s.t. ||Dfe*dx+fe||^2 + ||(Dfi*dx+fi)_+||^2 <= viol_opt^2
%       A*dx                                 <= b
%      ||dx||                                <= delta
%
% When viol_opt <= 1.0e-8 * max{ 1, ||f_E||+||f_I||}, the first constraint
% in (***) is replaced with Dfe*dx = -fe and Dfi*dx  <= -fi.
%
%
% Input:
% All quantities as listed above, where Dfe, Dfi, or A may have zero rows.
%
% Output: 
% An approximate solution x,
% Lagrange multiplyer estimates for the   equality  constraints (ye)
% Lagrange multiplyer estimates for the inequality  constraints (yi)
% Lagrange multiplyer estimates for the constraints  ``A*x<=b'' (yA)
% Lagrange multiplyer estimate  for the trust region constraint (ydelta)
% out, a structure containing  out.viol_opt, out.viol_red, 
%         out.KKT_reduct, out.TR_influence, and out.sign_viol
%      -- ``optimal'' constraint violation (= violation that is aimed for)
%      -- associated reduction of constraint violation
%      -- reduction of KKT violation ||g+H*dx+sum(y_l*Df_l')||/||g||
%      -- relative influence of trust region constraint (for KKT violation)
%      -- sign violation (of KKT multiplyers)
%      -- converged (=1 if there was a problem forcing to stop the algor.)
% 
% This is an experimental code without any guarantees 
% - free for use at your own risk
% - in return please report bugs to: jarre@hhu.de
%
% Florian Jarre, Felix Lieder, April, 2017
%
% Subprogram used:  mehrotramsocp
%
% Mehrotra predictor corrector method for Mixed Second Order Cone Programs.


% Set parameters:
infeas_reduction = 0.9;    % infeasibility to be reduced by at least 90%
                           % of what could be achieved.
desired_tol      = options.desired_tol; 
delta            = options.delta;
out.converged    = 0;
t_a              = 0; % third attempt in solving the SOCP problem
% t_a = 1 yields higher accuracy in few cases, but more often it
% yields less accuracy or a higher number of iteratons


% Initialization: partial consistency check of the input:
if ~isempty(b) 
   if min(b) < -1.0e-8;
      warning(' linear inequalities of SQP subproblem not satisfied ')
      tmp = sprintf('violation of linear ineqalities: %5g',max(-b));
      disp(tmp);
      out.converged = 1;
      dx = zeros(size(g ));
      ye = zeros(size(fe));
      yi = zeros(size(fi));
      yA = zeros(size(b ));
      ydelta = 0;
   end
end
if out.converged == 0 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n = length(g);
[d1,d2] = size(H);
if abs(d1-n)+abs(d2-n) > 0; error('g and H do not match'); end
[me,d] = size(Dfe);
if me == 0; Dfe = zeros(0,n); fe = zeros(0,1); d = n; end
if d~=n;  error('H and Dfe do not match'); end
[mi,d] = size(Dfi); 
if mi == 0; Dfi = zeros(0,n); fi = zeros(0,1); d = n; end
if d~=n;  error('H and Dfi do not match'); end
[mA,d] = size(A);   
if mA == 0; A = zeros(0,n); b = zeros(0,1); d = n; end
if d~=n;  error('H   and A do not match'); end


% Step zero: eliminate inequalities that cannot be active in this SQP step
Dfirownorm = sum((Dfi.*Dfi),2); 
Dfirownorm = Dfirownorm.^(1/2); % norms of rows of Dfi
criticalfi = fi > - delta*Dfirownorm;
miold = mi;
if sum(criticalfi) > 0
   fi  =  fi(criticalfi);
   Dfi = Dfi(criticalfi,:);
   mi  = length(fi);
else
   fi  = zeros(0,1);
   Dfi = zeros(0,n);
   mi  = 0;
end
options.Hactfi = options.Hactfi(criticalfi);
if isempty(options.Hactfi)
    options.Hactfi = zeros(0,1);
end
% below it is assumed that the rows of A have norm one!
criticalA = b < delta;
mAold = mA;
if sum(criticalA) > 0
   b  =  b(criticalA);
   A  = A(criticalA,:);
   mA  = length(b);
else
   b  = zeros(0,1);
   A  = zeros(0,n);
   mA  = 0;
end
options.HactAi = options.HactAi(criticalA);
if isempty(options.HactAi)
    options.HactAi = zeros(0,1);
end


% first, determine viol_opt (by how much the infeasibility can be reduced)
z1 = zeros(mi,1);
Asoc = [zeros(me,mA+mi+1),-Dfe,zeros(me,1+mi),eye(me)
        zeros(mi,mA),-eye(mi),z1,-Dfi,z1,eye(mi),zeros(mi,me)
        eye(mA),zeros(mA,mi+1),A,zeros(mA,1+mi+me)
        zeros(1,mA+mi),1,zeros(1,n+1+mi+me)];
    
if mi + mA == 0
   if me > 0 %%%
%   dx = -pinv(Dfe.'*Dfe)*(Dfe.'*fe); 
   dx = -(Dfe.'*pinv(Dfe*Dfe.')*fe); % chol(Dfe*Dfe.') not always defined
   dx = dx * min(1,delta*infeas_reduction/norm(dx));
   viol_opt = norm(fe+Dfe*dx);
   tmp11    = viol_opt/(eps+norm(fe));
   if tmp11 > .999
      out.converged = 1;
      if options.p_l > 0
         tmp = sprintf(...
        'stop due to poor reduction of infeasibility in SQP: %0.5g',tmp11);
         disp(tmp)
      end
   end
   if options.p_l > 1
      tmp = sprintf(...
         'predicted reduction of (eq.) constraint violation: %0.5g',tmp11);
      disp(tmp)
   end
   else %%%
      viol_opt = 0; 
      tmp11 = 0;
   end %%%
else % Use SOCP solver to determine viol_opt
   if norm(fe) + norm(max(0,fi)) > desired_tol
      delta1 = infeas_reduction*delta;
      bsoc = [fe;fi;b;delta1];
      csoc = [zeros(mA+mi+1+n,1);1;zeros(mi+me,1)];
      K.l = mA+mi;
      K.q = [1+n,1+mi+me];
      %options.tol = 1.0e-10; 
      [ x,~,~,out_m ] = mehrotramsocp( Asoc,bsoc,csoc,K,options ); 
      if out_m.done > 1 && options.p_l > 0
         disp(' inaccurate solution of first MSOCP subproblem in sqpsocp ')
      end
      viol_opt = csoc.'*x; % anticpated constraint violation after a step
                           % of length delta * infeas_reduction
      tmp11    = viol_opt/(eps+(norm(fe)^2 + norm(max(0,fi))^2)^.5) ;
      if tmp11 > .999
         out.converged = 1;
         if options.p_l > 0
            tmp = sprintf(...
        'stop due to poor reduction of infeasibility in SQP: %0.5g',tmp11);
            disp(tmp)
         end
      end
      if options.p_l > 1
         tmp =  sprintf(...
             'predicted reduction of constraint violation: %0.5g',tmp11);
         disp(tmp)
      end
   else
      viol_opt = 0;
      tmp11    = 0;
      if options.p_l > 1
         disp('current constraint violation below desired accuracy')
      end
   end
end % of Use SOCP solver to determine viol_opt
out.viol_opt = viol_opt; % targeted reduction of linearized infeasibility
out.viol_red = tmp11;    % relative reduction of linearized infeasibility


% second, solve the SQP subproblem
[U,L] = eig(H); L=U*(max(L,0).^.5)*U'; L = 0.5*(L+L');
% Now, L^2 = H if H is positive definite

if viol_opt > desired_tol * max( 1, norm(fe) + norm(fi) )
   viol_opt = max([viol_opt, 1.0e-3*(norm(fe) + norm(fi)) , desired_tol]);
   % to avoid that Slater's condition is almost violated
   % do not aim for an imrovement of more than 3 digits 
   Asoc = [Asoc,zeros(me+mi+mA+1,2+n)
           zeros(1,mA+mi+1+n),1,zeros(1,mi+me+2+n)
           zeros(1,mA+mi+1+n+1+mi+me),1,-1,zeros(1,n)
           zeros(n,mA+mi+1),sqrt(2)*L,zeros(n,3+mi+me),eye(n)];
   bsoc = [fe;fi;b;delta;viol_opt;2;zeros(n,1)];
   csoc = [zeros(mA+mi+1,1);g;zeros(1+mi+me,1);0.5;0.5;zeros(n,1)];
   K.l = mA+mi;
   K.q = [1+n,1+mi+me,2+n];
   %options.tol = 1.0e-10; 
   [ x,y,~,out_m ] = mehrotramsocp( Asoc,bsoc,csoc,K,options ); 
   if out_m.done > 1 && options.p_l > 0
      disp(' inaccurate solution of second MSOCP subproblem in sqpsocp ')
   end
   if options.p_l > 1
      disp('determine SQP step via MSOCP with 3 SOC-cones')
   end
else % viol_opt is considered zero
   Asoc = Asoc(:,1:mA+mi+1+n); 
   Asoc = [Asoc,zeros(me+mi+mA+1,2+n) 
           zeros(1,mA+mi+1+n,1),1,-1,zeros(1,n)
           zeros(n,mA+mi+1),sqrt(2)*L,zeros(n,2),eye(n)]; 
   bsoc = [fe;fi;b;delta;2;zeros(n,1)]; 
   csoc = [zeros(mA+mi+1,1);g;0.5;0.5;zeros(n,1)]; 
   K.l = mA+mi; 
   K.q = [1+n,2+n]; 
   %options.tol = 1.0e-10; 
   [ x,y,~,out_m ] = mehrotramsocp( Asoc,bsoc,csoc,K,options ); 
   if out_m.done > 1 && options.p_l > 0
      disp(' inaccurate solution of third MSOCP subproblem in sqpsocp ')
   end
   if options.p_l > 1
      disp('determine SQP step via MSOCP with 2 SOC-cones')
   end
end


% translate output back to the SQP format:
dx   = x(mA+mi+2:mA+mi+1+n);
grad = g+H*dx; % new anticipated gradient of the Lagrangian

ye =     y(         1:me);
yi = abs(y(      me+1:me+mi));
yA = abs(y(   me+mi+1:me+mi+mA));
yd = abs(y(me+mi+mA+1));

v1 = Dfe.'*ye; % zero vector if E is empty
v2 = Dfi.'*yi; % zero vector if I is empty
v3 =   A.'*yA; % zero vector if A is empty
v4 =   dx*yd;

tmp = [v1,v2,v3,v4].'*[v1,v2,v3,v4];
if isnan(norm(tmp,'fro')) || norm(tmp,'fro') == Inf
   z = zeros(4,1);
else
   if norm(dx) > 0.99*delta %consider trust region constraint as active
%   z = -[v1,v2,v3,v4] \ grad;
      z = -pinv(tmp) * [v1,v2,v3,v4].' * grad;
   else
%   z = -[v1,v2,v3] \ grad;
      z = -pinv([v1,v2,v3].'*[v1,v2,v3]) * [v1,v2,v3].' * grad;
      z(4) = 0;
   end
end

ye = ye*z(1);
yi = yi*z(2);
yA = yA*z(3);
yd = yd*z(4);

yi = max(0,yi);
yA = max(0,yA);
yd = max(0,yd);
ydelta = yd;

Dfirownorm = sum((Dfi.*Dfi),2); 
Dfirownorm = Dfirownorm.^(1/2); % norms of rows of Dfi
hiddeninactivefi = yi.*Dfirownorm < -0.1*(fi+Dfi*dx)*norm(g); % cf. below
yi(hiddeninactivefi) = 0;
hiddeninactiveA  = yA < 0.1*b*norm(g);
yA(hiddeninactiveA)  = 0;




% third, test whether the SQP step suffers from rounding errors 
% --- and correct rounding errors, if deemed necessary ---
tmp1 = norm(grad+Dfe.'*ye+Dfi.'*yi+A.'*yA+dx*yd)/(norm(g)+eps);
activefi = yi.*Dfirownorm >= -0.1*(fi+Dfi*dx)*norm(g);        % cf. above
activeA  = yA             >=  0.1*b*norm(g);
tmpA = options.HactAi;
sameact = sum(options.Hactfi == activefi) + sum(tmpA == activeA);


if tmp1 > 1.0e-02*options.residual && t_a && ...
        viol_opt < 0.1*(eps+norm(fe)^2+norm(fi(fi>0))^2)^.5 && ...
        sameact == sum(activefi)+sum(activeA) && mi+mA > 0 
   % The SQP subproblem is solved to low accuracy in the objective term,
   % but the (linearized) constraint violation is not an issue
   % and the active constraints did not change, 
   % therefore try to resolve for higher accuray in the objective term
   if options.p_l > 0
      tmp = sprintf('initial reduction of SQP-KKT violation: %0.5g',tmp1);
      disp(tmp)
      disp('Resolve SQP problem with fixed active constraints');
   end
   % add active inequalities (from fi and A) to fe
   inactivefi = ~activefi;
   inactiveA  = ~activeA;
   fic        =  fi(inactivefi); % corrected fi (i.e. without active part)
   Dfic       = Dfi(inactivefi,:);
   mic        = sum(inactivefi);
   bc         = b(inactiveA);
   Ac         = A(inactiveA,:);
   mAc        = sum(inactiveA);
   fec        = [ fe;  fi(activefi);  -b(activeA)];
   Dfec       = [Dfe; Dfi(activefi,:); A(activeA,:)];
   mec        = me+sum(activefi)+sum(activeA);
   z1c        = zeros(mic,1);
   Asoc       = [zeros(mec,mAc+mic+1),-Dfec,zeros(mec,1+mic),eye(mec)
           zeros(mic,mAc),-eye(mic),z1c,-Dfic,z1c,eye(mic),zeros(mic,mec)
           eye(mAc),zeros(mAc,mic+1),Ac,zeros(mAc,1+mic+mec)
           zeros(1,mAc+mic),1,zeros(1,n+1+mic+mec)];
   % following: same as the SQP step above but with corrected data 
   if viol_opt > desired_tol * max( 1, norm(fe) + norm(fi) )
      if options.p_l > 1
         disp('case 1, resolve with equality and inequality constraints')
      end
      % in this case viol_opt has been increased to gain at most 3 digits
      %H = H + 1.0e-4*norm(H,'fro')*eye(n); % stronger regularization
      % ----- since trust region constraint is expected to be active
      %L = chol(H); 
      %L = L.'; 
      Asoc = [Asoc,zeros(mec+mic+mAc+1,2+n)
              zeros(1,mAc+mic+1+n),1,zeros(1,mic+mec+2+n)
              zeros(1,mAc+mic+1+n+mic+mec+1),1,-1,zeros(1,n)
              zeros(n,mAc+mic+1),sqrt(2)*L,zeros(n,3+mic+mec),eye(n)];
      bsoc = [fec;fic;bc;delta;viol_opt;2;zeros(n,1)]; 
	  csoc = [zeros(mAc+mic+1,1);g;zeros(1+mic+mec,1);0.5;0.5;zeros(n,1)];
      K.l = mAc+mic;
      K.q = [1+n,1+mic+mec,2+n];
      [ x,y,~,out_m ] = mehrotramsocp( Asoc,bsoc,csoc,K,options ); 
      
      %if norm(Asoc*x-bsoc) > 1.0e-8; out_m.done = 2; end
      
      if out_m.done > 1 && options.p_l > 0
         disp(' inaccurate solution of 4. MSOCP subproblem in sqpsocp 2')
      end
      % translate output back to the SQP format (same as above):
      dx   = x(mAc+mic+2:mAc+mic+1+n);
      grad = g+H*dx;
      ye =     y(         1:me);
   
      yi = zeros(mi,1); % yi and yA need to be reordered
      yA = zeros(mA,1);
      %yi(inactivefi) = abs(y(       mec+1:mec+mic)); % should be zero 
      %yA(inactiveA)  = abs(y(   mec+mic+1:mec+mic+mAc));
      safi = sum(activefi);
      saA  = sum(activeA);
      yi(  activefi) = abs(y(me+1:me+safi));
      yA(  activeA)  = abs(y(me+safi+1:me+safi+saA));
   
      yd = abs(y(me+mi+mA+1));
      v1 = Dfe.'*ye; % zero vector if E is empty
      v2 = Dfi.'*yi; % zero vector if I is empty
      v3 =   A.'*yA; % zero vector if A is empty
      v4 =   dx*yd;
      if norm(dx) > 0.99*delta %consider trust region constraint as active
         % z = -[v1,v2,v3,v4] \ grad;
         z = -pinv([v1,v2,v3,v4].'*[v1,v2,v3,v4]) * [v1,v2,v3,v4].' * grad;
      else
         % z = -[v1,v2,v3] \ grad;
         z = -pinv([v1,v2,v3].'*[v1,v2,v3]) * [v1,v2,v3].' * grad;
         z(4) = 0;
      end
      ye = ye*z(1);
      yi = yi*z(2);
      yA = yA*z(3);
      yd = yd*z(4);
      yi = max(0,yi);
      yA = max(0,yA);
      yd = max(0,yd);
      ydelta = yd;
      hiddeninactivefi = yi.*Dfirownorm < -0.1*fi*norm(g);
      yi(hiddeninactivefi) = 0;
      hiddeninactiveA  = yA < 0.1*b*norm(g);
      yA(hiddeninactiveA)  = 0;
      
      % if max(Dfi*dx+fi) > 1.0e-6 || -min(b-A*dx) > 1.0e-8
      % disp('inaccurate solution of SQP problem')
      
   else % viol_opt is considered zero
      if options.p_l > 1
         disp('case 2, resolve with equality constraints only')
      end
      tmpx = - pinv([H+yd*eye(n), Dfec.';Dfec, zeros(mec,mec)]) * [g;fec];
      dx = tmpx(1:n);
      ndx = norm(dx);
      if ndx > 0
         dx = dx * min(1, delta/ndx);
      end
      grad = g+H*dx;
      ye = tmpx(n+1:n+me);
      yi = zeros(mi,1); % yi and yA need to be reordered
      yA = zeros(mA,1);
      safi = sum(activefi);
      saA  = sum(activeA);
      yi(  activefi) = abs(tmpx(n+me+1:n+me+safi));
      yA(  activeA)  = abs(tmpx(n+me+safi+1:n+me+safi+saA));
   end

end % of correction --- if deemed necessary ---




tmp1 = norm(grad+Dfe.'*ye+Dfi.'*yi+A.'*yA+dx*yd)/(norm(grad)+eps);
tmp2 = norm(dx*yd)/norm(grad);
tmp2b = ((norm(fe+Dfe*dx)^2+norm(max(fi+Dfi*dx,0))^2)/...
         (norm(fe)^2+norm(max(fi,0))^2+eps))^.5;
if length(yi)+length(yA)>0
   tmp3 = max(0,-min([min(yi).',min(yA).'])); % min(yi).' (with transpose!)
                                            % also works when yi is empty
else
   tmp3 = 0;
end
tmp4 = max(b-A*dx);
if min(tmp4) < -1.0e-6 % this is stronger violation than the tolerance
                       % aimed for, possible correction in subsequent iter.
   if options.p_l > 1
      disp(' stop due to rounding errors ')
      tmp = sprintf(...
         'violation of linear inequalities after SQP step: %0.5g',-min(b));
      disp(tmp)
   end
   out.converged = 1;
end
if mi+me>0
   tmp5 = ( norm(fe+Dfe*dx)^2+norm( max(fi+Dfi*dx,0) )^2 )^.5;
   if tmp5 > 1.0e-6 % again, allow stronger violation than aimed for
      tmp5 = tmp5/( norm(fe)^2+norm(max(fi,0))^2 )^.5;
      if tmp5 > 0.5*(1+tmp11)
         out.converged = 1; 
         if options.p_l > 0
            disp(' stop due to rounding errors ')
            tmp = sprintf(...
     'constraint violation in SQP step larger than predicted: %0.5g',tmp5);
            disp(tmp)
         end
      end
   end
end
if options.p_l > 1
   tmp = sprintf('rel. reduction of SQP-KKT violation:       %0.5g',tmp1);
   disp(tmp)
   tmp = sprintf('rel. influence of trust region constraint: %0.5g',tmp2);
   disp(tmp)
   tmp = sprintf('rel. reduction of constraint violation:    %0.5g',tmp2b);
   disp(tmp)
   tmp = sprintf('sign violation of KKT multipliers:         %0.5g',tmp3);
   disp(tmp)
   if yd < 0
      tmp = sprintf('sign violation for yd:                  %0.5g',-yd);
      disp(tmp)
   end
end
out.KKT_reduct   = tmp1;
out.TR_influence = tmp2;
out.con_reduct   = tmp2b;
out.sign_viol    = tmp3;


% last step, include eliminated inequalities with the multiplyers
tmp = yi;
yi  = zeros(miold,1);
yi(criticalfi) = tmp;
tmp = yA;
yA  = zeros(mAold,1);
yA(criticalA) = tmp;
out.activefi = yi > 0;
out.activeAi = yA > 0;

end %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end








function [ x,y,s,out ] = mehrotramsocp( A,b,c,K,options )
%MEHROTRAMSOCP 
% Mehrotra predictor corrector method for Mixed Second Order Cone Programs.
% Using dense arithmetic, i.e. sparsity is not exploited.
% 
% min c^T x | A * x = b, x \in K
% 
% where K is a structure such that 
%
%       K.l \ge 0 is the number of nonnegative variables and
%       K.q is a vector with the dimensions (\ge 2 each) of the SO-cones,
%       if there are no nonnegative variables, K.l is 0  or undefined,
%       if there are no SO-cones,              K.q is [] or undefined.
%
% and x \in K iff 
%     x(1:K.l) \ge 0,
%     || x(K.l+2:K.l+K.q(1)) || <= x(K.l+1),
%     followed by the SO-cones of dimensions K.q(2) ... K.q(end).
%
% Mandtory input:
%   A, b, c of dimensions conforming with above problem formulation.
% Optional input:
%   K,       (default is all nonnegative variables)
%   options, (a structure with 
%      options.tol,   some relative stopping tolerance (default 1.0e-10)
%      options.maxit, stopping the algorithm with an error message after
%                     maxit iterations)
%      options.p_l,   where p_l (printlevel) > 0 for debugging purposes
%
% NOTE: The input format of sedumi (Jos Sturm) is similar but allows 
%       much more general data!
%
% Output:
%   x,   approximate primal solution 
%   y,s, approximate dual solution (ideally A'*y+s=c, s \in K, x'*s=0)
%   out, structure with fields
%        mu:   final value of mu
%        done: stopping information
%             1 normal termination
%             2 ill-conditioning of A*W*A'
%             3 short steps in interior-point algorithm
%             4 back solve failed (highly inaccurate)
%             5 iteration limit reached
%             6 trivial solution x = 0, optimal if there is an opt. sol.
%        residual: final primal-dual residual (for the normalized problem)
%
% This is an experimental code without any guarantees - 
% - free for use at your own risk.
%
% Florian Jarre, March 17, 2016
%



% Complete the input:
[m,n] = size(A);
if (nargin < 4)
   K.l = n;
end
if ~isfield(K,'l')
   K.l = 0;
end
if ~isfield(K,'q')
   K.q = [];
end
if (nargin < 5)
   options.tol   = 1.0e-12;
   options.maxit = 300; 
else
   if ~isfield(options,'tol')
      options.tol   = 1.0e-12;
   end
   if ~isfield(options,'maxit')
      options.maxit = 300;
   end
end

dummy = K.l + sum(K.q);
if dummy ~= n; error(' dimensions of K and of A do not match '); end;
pS = length(K.q); % number of second order cones.
pK = pS + K.l;    % overall number of cones, scalar product of e^Te.

eee        = zeros(n,1); % the central element ``e''
eee(1:K.l) = 1;
i          = K.l;
for j = 1:pS
   eee(i+1)=1; i=i+K.q(j);
end


% Preprocessing and initialization:
done = 0; % not done yet
recover_y = 0; % y does not need to be recovered separately
if norm(A,'fro')==Inf || isnan(norm(A,'fro')) || norm(b)==Inf || ...
        isnan(norm(b)) || norm(c)==Inf || isnan(norm(c))
   done = 8;
   bnorm_old = 1;
   cnorm_old = 1;
   residual = 1;
else
   AAt   = A*A.';
   [R,p] = chol(AAt);  % needs minimum degree ordering for sparse problems
   if p > 0 
      disp(' AAt not positive definite in mehrotramsocp'); 
      if norm(A,'fro') == 0
         if norm(b) > 0
            disp('inconsistent linear equations in mehrotramsocp')
            done = 7;
            residual = 1;
         else
            done = 6; % A = 0 and b = 0
            residual = 0;
         end
      end
% Orthogonalize the rows of A and eliminate linearly dependent rows
      tolA = eps^.75; % tolerance for estimating singularity of A
      [~,R,E] = qr(A.',0);
      rank_est = sum(abs(diag(R))>tolA*max(abs(diag(R))));
      if rank_est < m
         if norm(A*(A\b)-b) > tolA*norm(b)
            disp('linear equations in mehrotramsocp seem inconsistent')
            residual = 1; 
            done = 8;
         end
      end
      Aold = A; mold = m;
      recover_y = 1; % recompute y of correct dimension at the end
      A = A(E(1:rank_est),:);
      b = b(E(1:rank_est)); 
      m = rank_est;
      A = (R(1:rank_est,1:rank_est).')\A; % orthonormalize
      b = (R(1:rank_est,1:rank_est).')\b;
      R = eye(m);
   end;
   clear AAt;

   A = R.'\A; %for stability make (nearly) orthonormal rows (dense case!)
   b         = R.'\b;
   bnorm_old = norm(b);
   dyold     = A*c;           %remember A'*dyold is subtracted from c below
   c         = c - A.'*dyold; % change c to lie in the null space of A
   R_old     = R;             % remember R (to rescale y in the end)
   cnorm_old = norm(c);

   if bnorm_old == 0
      done = 6;
      bnorm_old = 1;
      residual = 1; % since the dual solution might not be feasible
   end
   if cnorm_old == 0
      c = eee;
      cnorm_old = norm(c);
      if options.p_l > 0
         disp('zero objective function in mehrotramsocp changed to e^T*x');
      end
   end

   b = b / bnorm_old;
   c = c / cnorm_old;
   nb1 = norm(b);
   nc1 = norm(c);
end

x0        = eee;           x  = x0;
y0        = zeros(m,1);    y  = y0;
s0        = x0;            s  = s0;
mu0       = x.'*s/pK;      mu = mu0;
res0      = norm(A*x-b);
res1      = norm(A.'*y+s-c);
quot0     = 10.e-2*mu/(res0+1);
quot1     = 1.0e-2*mu/(res1+1);
iter      = 0;
%initialit = ceil(log(n)/log(10)); % first some plain centering
initialit = 8; 



% Main loop:
while done == 0
   iter = iter+1;
       
   %str_g = sprintf('mu = %0.5g',mu); disp(str_g)
   
% Compute the scaling point of x and s
   w   = zeros(n,1);
   eta = zeros(n,1);
   if K.l > 0 
      if min(min(x(1:K.l)),min(s(1:K.l))) <= 0 
         disp(' iterate in mehrotramsocp is not positive ');
         done = 3;
         correction = eps-2*min(min(x(1:K.l)),min(s(1:K.l)));
         x(1:K.l) = x(1:K.l)+correction;
         s(1:K.l) = s(1:K.l)+correction;
      end
      eta(1:K.l) = (x(1:K.l)./s(1:K.l)).^(0.5);
   end
   i = K.l;
   for j = 1:pS
      njj  = K.q(j);
      xbar = x(i+1:i+njj); 
      sbar = s(i+1:i+njj);
      detx = xbar(1)^2-xbar(2:njj).'*xbar(2:njj);
      dets = sbar(1)^2-sbar(2:njj).'*sbar(2:njj);
      if min([detx,dets,x(i+1),s(i+1)]) <= 0
          if options.p_l > 0
             disp(' iterate in mehrotramsocp is not in interior of SOC ');
          end
          done = 3;
          xbar(1) = max(xbar(1),norm(xbar(2:end))*(1+1.0e-10)+1.0e-10);
          sbar(1) = max(sbar(1),norm(sbar(2:end))*(1+1.0e-10)+1.0e-10);
          detx = xbar(1)^2-xbar(2:njj).'*xbar(2:njj);
          dets = sbar(1)^2-sbar(2:njj).'*sbar(2:njj);
          x(i+1) = xbar(1);
          s(i+1) = sbar(1);
      end
      xbar  = xbar/sqrt(detx);
      sbar  = sbar/sqrt(dets);
      sbari = -sbar; sbari(1) = sbar(1);
      gamma = 1/sqrt(2*(1+xbar.'*sbar));
      w(i+1:i+njj)   = gamma*(xbar+sbari);
      eta(i+1:i+njj) = (detx/dets)^.25;
      eta(i+1)       = -eta(i+1); % to allow a simple rank-1-update for W
      i=i+K.q(j);
   end

% Compute AW
   AW = A .* (ones(m,1)*eta.'); % avoid sparse computation in octave
   i = K.l;
   for j = 1:pS
      njj             = K.q(j);
      wj              = w(i+1:i+njj); 
      wj(1)           = wj(1)+1;
      wj              = sqrt(-eta(i+1)/wj(1)) * wj; % eta(i+1) is negative
      AW(:,i+1:i+njj) = AW(:,i+1:i+njj) + (A(:,i+1:i+njj)*wj)*wj.';
      i               = i+njj;
   end

   AWAt   = AW*AW.';
   [R,p] = chol(AWAt);  
   if p > 0; R = 1.0e16*eye(size(AWAt)); done = 2; end;

% predictor step
   p = b-A*x;
   q = c-(y.'*A).'-s;
   if iter <= initialit
      if norm(p) < 0.1 && norm(q) < 0.1
         initialit = iter-1; % no need for continuing phase 1
      end
   end
   v = wmult(w,eta,s,K);
   residualfirst = 0; % the priority is not to first reduce the residual
   if mu < max(res0*quot0,res1*quot1)
      residualfirst = 1; % now, priority is to reduce the residual first
      r = max(res0*quot0,res1*quot1)*jinv(v,K)-v; % some centering
   else
      r = -v; % affine scaling 
   end
   if iter <= initialit
      r = jinv(v,K)-v; % keep mu at mu = 1
   end
   rhs  = p + AW*(wmult(w,eta,q,K)-r);
   
   [dxp,~,dsp,failback] = backsolve(A,AW,R,rhs,p,q,r,w,eta,K);   
   if failback == 1
      done = 4;
   end
   
% corrector step
   alxmax = alphamax(x,dxp,K);
   alsmax = alphamax(s,dsp,K);
   if residualfirst == 1 % make sure residual reduces not ... 
      munew = max(res0*quot0,res1*quot1); % ... much slower than mu
      target = munew;
   else
      munew  = (x+alxmax*dxp).'*(s+alsmax*dsp)/pK;
      target = munew; % this target could be reached by the predictor step
      munew  = min(0.5*mu,mu*(munew/mu)^3);
   end
   tmp    = jmult(winvmult(w,eta,dxp,K),wmult(w,eta,dsp,K),K);
   theta  = 1; % possible damping of the Mehrotra corrector
   r      = munew*jinv(v,K)-v-theta*linvmult(v,tmp,K);
   if iter <= initialit
      theta = 1; % possible damping of the Mehrotra corrector
      r = jinv(v,K)-v-theta*linvmult(v,tmp,K);
   end

   rhs    = p + AW*(wmult(w,eta,q,K)-r);
   
   [dx,dy,ds,failback] = backsolve(A,AW,R,rhs,p,q,r,w,eta,K);
   if failback == 1
      done = 4;
   end
   
   if norm(dx) == 0 
      alxmax = 1;
   else
      alxmax = alphamax(x,dx,K);
   end
   if norm(ds) == 0 
      alsmax = 1;
   else
      alsmax = alphamax(s,ds,K);
   end
   eta_x  = min(0.95,0.5+0.5*alxmax); 
   eta_s  = min(0.95,0.5+0.5*alsmax); 
   eta_x  = min(eta_x,eta_s); eta_s=eta_x; %same damping in primal and dual
   if iter <= initialit
      eta_x = min(eta_x,0.5); eta_s = min(eta_s,0.5);
   end
   alx    = min(1,eta_x * alxmax);
   als    = min(1,eta_s * alsmax);
      
   xold=x;yold=y;sold=s;muold=mu; % remember these for possible safety step
   x  = x + alx*dx;
   s  = s + als*ds;
   y  = y + als*dy;
   if alx < 0.001; done = 3;  end
   if als < 0.001; done = 3;  end
   mu = x.'*s/pK;
   if iter > initialit && mu >= 0.75*(target+muold) && ...
                          residualfirst == 0
% slow reduction of mu, but not because munew had been increased
      if options.p_l >= 2
         str_g = sprintf('safety step in mehrotramsocp, mu = %0.5g',mu);
         disp(str_g)
      end
      %keyboard
      r   = muold*jinv(v,K)-v;
      rhs = p + AW*(wmult(w,eta,q,K)-r);
      [dxp,dyp,dsp,failback] = backsolve(A,AW,R,rhs,p,q,r,w,eta,K); 
      if failback == 1
         done = 4;
      end
      alxmax = alphamax(xold,dxp,K);
      alsmax = alphamax(sold,dsp,K);
      x    = xold + 0.5*alxmax*dxp;
      y    = yold + 0.5*alsmax*dyp;
      s    = sold + 0.5*alsmax*dsp;
      %[iter,muold,target,mu,x'*s/pK] % For debugging purposes
      mu   = x.'*s/pK;
   end
   
   res0      = norm(A*x-b);
   res1      = norm(A.'*y+s-c);
   angle     = x.'*s/((norm(x)+1)*norm(s)+1);
   residual = max([res0/nb1,res1/nc1,angle]); 
   if iter >= options.maxit
      done = 5;
   end
   if residual <= options.tol 
      done = 1; %normal termination
   end
   
end % of Main loop


if residual <= max(1.0e-8,10*options.tol) 
   done = 1; %still consider this a normal termination
end

if options.p_l > 0
if done == 2
   disp(' mehrotramsocp: early termination due to ill-conditioning ')
   tmp = sprintf('Final residual: %5.0g',residual);
   disp(tmp)
end
if done == 3
   disp(' mehrotramsocp: early termination due to short step length ')
   tmp = sprintf('Final residual: %5.0g',residual);
   disp(tmp)
end
if done == 4
   disp('mehrotramsocp: termination due to rounding errors in backsolve')
   tmp = sprintf('Final residual: %5.0g',residual);
   disp(tmp)
end
if done == 5
   disp(' mehrotramsocp: termination since iteration limit is reached ')
   tmp = sprintf('Final residual: %0.5g',residual);
   disp(tmp)
end
if done == 6
   disp(' in mehrotramsocp:')
   disp('x = zero is optimal, if there is a finite optimal solution')       
end
if done == 8
   disp(' in mehrotramsocp: data not finite ')
end
end % of options.p_l > 0


if done == 6
   x = zeros(n,1);
   y = zeros(m,1); % this dual solution might not be feasible
   s = zeros(n,1); % this dual solution might not be feasible
end
if done == 7
   x = pinv(A)*b;
   s = zeros(n,1); % this dual solution might not be feasible
   if recover_y == 0
      y = zeros(m,1); % this dual solution might not be feasible
   else
      y = zeros(mold,1);
   end
end
if done == 8
   disp(' in mehrotramsocp: data not finite ')
   x = x0; % return something
   y = y0; 
   s = s0; 
end




% Postprocessing, rescaling the result.
x = x * bnorm_old;
y = y * cnorm_old;
s = s * cnorm_old;

if done <= 7
   y = y + dyold; % because above, A'*y + s = cnew = cold - A'*dyold
   y = R_old\y;   % because above, A was rescaled with inv(R_old')
   if recover_y == 1
      y = pinv(Aold.')*(c-s); 
   end
end

if options.p_l > 1
   tmp = sprintf('Number of iterations in mehrotramsocp: %5.0f',iter);
   disp(tmp)
end

out.mu       = mu;
out.done     = done;
out.residual = residual;

end








function xs = jmult(x,s,K)
% compute the Jordan-product of x and s

pS = length(K.q);
xs = s;
if K.l > 0; xs(1:K.l) = x(1:K.l).*s(1:K.l); end
i = K.l;
for j = 1:pS
   njj           = K.q(j);
   xs(i+1)       = x(i+1:i+njj).'*s(i+1:i+njj);
   xs(i+2:i+njj) = x(i+1)*s(i+2:i+njj)+s(i+1)*x(i+2:i+njj);
   i             = i+njj;
end
end








function v = linvmult(x,s,K)
% multiply s with inv(L(x)) (not with L(inv(x)):

pS = length(K.q);
v = zeros(size(s));
i = K.l;
if K.l > 0
   if min(abs(x(1:K.l))) == 0; error(' linvmult divide by zero'); end
   v(1:K.l) = s(1:K.l)./x(1:K.l); 
end
for j = 1:pS
   njj           = K.q(j);
   xj            = x(i+1:i+njj); 
   xj1           = xj(1); % to avoid confusion with sign change later
   xj(1)         = -xj(1);
   detxj         = xj1^2-xj(2:njj).'*xj(2:njj);
   if xj1   == 0; error(' linvmult divide by zero (SOC1)'); end
   if detxj == 0; error(' linvmult divide by zero (SOC2)'); end
   v(i+2:i+njj)  = s(i+2:i+njj)/xj1;  
   v(i+1:i+njj)  = v(i+1:i+njj) + xj*((xj.'*s(i+1:i+njj))/(detxj*xj1)); 
   i             = i+njj;
end
end








function ws = wmult(w,eta,s,K)
% multiply s with W given by w and eta:

pS = length(K.q);
ws = eta.*s;
i = K.l;
for j = 1:pS
   njj           = K.q(j);
   wj            = w(i+1:i+njj); 
   wj(1)         = wj(1)+1;
   wj            = sqrt(-eta(i+1)/wj(1)) * wj; % eta(i+1) is negative
   ws(i+1:i+njj) = ws(i+1:i+njj) + (s(i+1:i+njj).'*wj)*wj;
   i             = i+njj;
end
end








function ws = winvmult(w,eta,s,K)
% multiply s with W^{-1} given by w and eta:

pS = length(K.q);
ws = s./eta;
i = K.l;
for j = 1:pS
   njj           = K.q(j);
   wj            = w(i+1:i+njj); 
   wj(1)         = -(wj(1)+1);
   wj            = sqrt(1/(eta(i+1)*wj(1))) * wj; % eta(i+1),wj(1) negative
   ws(i+1:i+njj) = ws(i+1:i+njj) + (s(i+1:i+njj).'*wj)*wj;
   i             = i+njj;
end
end








function si = jinv(s,K)
% compute the Jordan-inverse of s

pS = length(K.q);
si = s;
if K.l > 0; si(1:K.l) = 1./s(1:K.l); end
i = K.l;
for j = 1:pS
   njj           = K.q(j);
   sj            = s(i+1:i+njj); 
   detsj         = sj(1)^2-sj(2:njj).'*sj(2:njj);
   sj(1)         = -sj(1);
   si(i+1:i+njj) = (-1/detsj)*sj;
   i             = i+njj;
end
end








function [dx,dy,ds,fail] = backsolve(A,AW,R,rhs,p,q,r,w,eta,K)
% do a back solve with one step of iterative refinement
p_l = 2; % set p_l > 2 for debugging 

fail = 0; % assume the backsolve runs well
tmp  = R.' \ rhs;
dy   = R   \ tmp;
ds   = q - A.'*dy;
dx   = wmult(w,eta,r-wmult(w,eta,ds,K),K); 

pp = p - A*dx;
qq = q - (A.'*dy+ds);
rr = r - (winvmult(w,eta,dx,K)+wmult(w,eta,ds,K)); 

res0 = norm([p;q;r]);
res1 = norm([pp;qq;rr]);

if res1 > 1.0e-3*res0 && p_l > 2
   warning('ill conditioning in backsolve'); 
end
if res1 > 1.0e-10*res0
   %disp('refine')
   rhs = pp + AW*(wmult(w,eta,qq,K)-rr);
   tmp = R.' \ rhs;
   ddy = R   \ tmp;
   dds = qq - A.'*ddy;
   ddx = wmult(w,eta,rr-wmult(w,eta,dds,K),K); 

   dx = dx+ddx;
   dy = dy+ddy;
   ds = ds+dds;
      
   pp   = p - A*dx;
   qq   = q - (A.'*dy+ds);
   rr   = r - (winvmult(w,eta,dx,K)+wmult(w,eta,ds,K)); 
   res2 = norm([pp;qq;rr]);
         
   if res2 > min(res1,1.0e-4*res0)
      if p_l > 2
         warning('iterative refinement failed'); 
      end
      dx = zeros(size(dx));
      dy = zeros(size(dy));
      ds = zeros(size(ds));
      fail = 1; 
   end
end

end








function almax = alphamax(x,dx,K)
% compute the longest feasible step <= 2 in K

pS = length(K.q);
almax = 2;
if K.l > 0
   crit = find(dx(1:K.l) < 0);
   if ~isempty(crit)
      almax = min(almax,min(-x(crit)./dx(crit)));
      if almax < 0; disp('error0 in alphamax'); almax = 0; end
   end
end
i = K.l;
for j = 1:pS
   njj           = K.q(j);
   xj            = x(i+1:i+njj); 
   dxj           = dx(i+1:i+njj); 
   detxj         =  xj(1)^2- xj(2:njj).'* xj(2:njj);
   detdxj        = dxj(1)^2-dxj(2:njj).'*dxj(2:njj);
   tmp           = xj(1)*dxj(1)-xj(2:njj).'*dxj(2:njj);
   
   if dxj(1) < norm(dxj(2:njj)) % (step length in SOC must be checked) ***
       
   % first the numerically instable cases
   instab = 0; % no instable case detected
   if tmp == 0
      if detdxj < 0
         almax = min(almax, sqrt(abs(detxj/detdxj)));
      end
      instab = 1;
   end
   if abs(detxj*detdxj) <= 1.0e-6*tmp^2 && tmp < 0
      almax = min(almax, abs(detxj/(2*tmp)));
      instab = 1;
   end
   if detdxj >= 0 && tmp >= 0 % then almax = inf
      disp('error1 in alphamax'); % this should not happen here
      instab = 1;
   end
   
   % now the standard case
   if ~instab
      tmproot = tmp^2-detdxj*detxj;
      if tmproot < -1.0e-10*tmp^2; almax = 0; end % rounding errors
      tmproot = sqrt(abs(tmproot));
      al1 = (-1/detdxj) * (tmp + tmproot); 
      al2 = (-1/detdxj) * (tmp - tmproot);
      if min(al1,al2) <= 0
         al1 = max(al1,al2);
      else
         al1 = min(al1,al2);
      end
      if al1 < 0; error('error3 in alphamax'); end
      almax = min(almax, al1);
   end
   
   end % (end of: step length in SOC must be checked) ********************
   i = i+njj;
end
end








function y = curvimerit(fobj,fcon,t,x,dx,ddx,bigM)
% compute a point on the curve x + t*dx + t^2*ddx and evaluate the merit 
% function at this point (used for a curvilinear search where t > = 0)
%
% function [y,xt] = curvimerit(fobj,fcon,t,x,)
%
% Input:
%        a function handle fobj
%        a function handle fcon
%        a scalar t >= 0,  vectors x, dx, ddx
%        a penalty parameter bigM
%
%
% OUTPUT:
%   y   the value of the penalty function at x+t*dx+min(1,max(0,t))^2*ddx
%
% Test Version with errors -- No guarantee of any kind is given!!!
%


f  = @(x) fobj(x);
fc = @(x) fcon(x);

xt = x+t*dx+min(1,max(0,t))^2*ddx;
[fenew,finew] = fc(xt);
fnew          = f(xt);
if fnew == -Inf
   tmp = (norm(fenew)^2+norm(max(finew,0))^2)^.5;
   if tmp < 1.0e-8 % The point is (almost) feasible, keep fnew = - Inf
      y = -Inf;
   else
      y = tmp; % just look at the infeasibility
   end
else
   y = fnew + bigM*(norm(fenew)^2+norm(max(finew,0))^2)^.5;
end

end








function y = NaN2Inf(x)
% For minimization replace NaN with Inf

if isnan(x)
   y = Inf;
else
   y = x;
end

end








function [a1,b1] = NaN2Inf_con1(f_con, x)
% For minimization replace NaN with Inf

[a1,b1] = f_con(x);
if isempty(a1)
   a1 = zeros(0,1);
end
if isempty(b1)
   b1 = zeros(0,1);
end
if isnan(norm(a1))
   a1 = Inf(size(a1));
end
if isnan(norm(b1))
   b1 = Inf(size(b1));
end

end








%function [a1,b1] = NaN2Inf_con2(f_con, x,y)
function [a1,b1] = NaN2Inf_con2(f_con, x)
% For minimization replace NaN with Inf

%[a1,b1] = f_con(x,y);
[a1,b1] = f_con(x);
if isempty(a1)
   a1 = zeros(0,1);
end
if isempty(b1)
   b1 = zeros(0,1);
end
if isnan(norm(a1))
   a1 = Inf(size(a1));
end
if isnan(norm(b1))
   b1 = Inf(size(b1));
end

end








function [x,y,g,H,out] = mwd11(fin,options)
% Minimization of a continuous function f: R --> R union {Inf}
%
% last change: August 2016.
%
% Simplest calling routine
%
%   x = mwd11(@f);
%
% or, with more specifications:
%
%   [x,fx,out] = mwd11(@f,options);
%
% Test Version with errors -- No guarantee of any kind is given!!!
%
% Algorithms: Some form of bisection and spline interpolation.
%             Return the lowest point that the algorithm stumbles about
%             (in contrast to the official Matlab routine)
%
%
% INPUT: 
%   Mandatory: A function handle f,
%   Optional:  The structure options as outlined below.
%
%   options.lb   - The minimizer is restricted to the interval [lb,ub]. 
%                  (Default of lb is -Inf)
%   options.ub   - (Default of ub is  Inf)
%                  If lb or ub are specified f will be evaluated only
%                  within the given bounds
%   options.xact - Reference point for the line search. 
%                  When lb <= xact <= ub the returned value shall be at 
%                  least as good as f(xact).
%
%   options.tol  - An approximate tolerance for the minimizer x:
%                  In the nonsmooth case there exists a local minimizer xm
%                  of f satisfying |xm - x| <= tol * int_length where 
%                  int_length is the length of a sub interval of [lb,ub] 
%                  generated by the algorithm. (Default tol = 1e-8)
%                  In the smooth case the above stopping criterion is
%                  based on an estimate - and is not guaranteed!
%                  WARNING: 
%                  Direct search for smooth minimization can never get more
%                  than about half the digits of full machine precision
%
%
%
% OUTPUT:
%   x        -- some approximate minimizer
%   y        -- the associated function value
%   g        -- NaN or an estimate of f'  at x, if available from
%               spline interpolation
%   H        -- NaN or an estimate of f'' at x, if available from
%               spline interpolation
%   out.iter -- specifies the number of function evaluations needed.
%   out.acc  -- length of final interval containing x
%
%
% Subroutines used:
%           find_spline.m -- find a least squares spline through xx and ff
%           eval_spline.m -- evaluate the spline at a given point
%           min_spline.m  -- find the minimizer of the spline

done       = 0;    % We are not done yet
spline_int = 0;    % so far no use of spline interpolation
iterations = 0;    % number of function evaluations

% COMPLETE OPTIONAL INPUT AND GENERATE STARTING POINT xact in (lb,ub):

if (nargin < 2)
   options.lb   = -Inf;
   options.ub   =  Inf;
   options.xact = 0.0;
   options.tol  = 1.0e-8;
end

% Set x0 and x1: 
if ~isfield(options,'lb') 
   options.lb = -Inf; 
end

if ~isfield(options,'ub')
   options.ub =  Inf;
end

x0 = options.lb;
x1 = options.ub;
if x1 < x0
   disp('lower bound larger than upper bound in line search');
   done = 1;
   options.xact = 0.5*(x0+x1);
   xact         = 0.5*(x0+x1);
   fact         = NaN2Inf(fin(xact)); f0 = fact; f1 = fact;
   iterations   = iterations + 1;
end
if ~isfield(options,'par_f')
   f = @(x) NaN2Inf(fin(x));
else
   f = @(x) NaN2Inf(fin(x,options.par_f));
end


if x0 == x1
   if x0 == -Inf || x0 == Inf
      error('bounds in line search are inconsistent');
   end
   options.xact = x0;
   xact         = x0;
   fact         = f(xact); f0 = fact; f1 = fact;
   iterations   = iterations + 1;
   done         = 1;
   warning('Trivial line search in interval of length zero');
end

% Set xact:
if ~isfield(options,'xact') 
   options.xact = x0;
end
xact = options.xact;
if xact <= x0 || xact >= x1 % Make sure xact is in the interior of [x0,x1]
   if x0  > -Inf
      if x1 < Inf
         xact = 0.5*(x0+x1);
      else
	     xact = x0 + 0.5*abs(x0) + 1;
      end
   else % Case x0 = -Inf and then
      if x1 < Inf
         xact = x1 - 0.5*abs(x1) - 1;
      else
         xact = 0;
      end
   end
end
if done == 0
   fact = f(xact);
   iterations = iterations + 1; % Number of function evaluations
end
% Now, x0 < xact < x1  but x0 = -Inf of x1 = Inf is possible


% Set the tolerance
if ~isfield(options,'tol') 
   options.tol = 1.0e-8;
end
tol = min(0.1,max(1.0e-10,options.tol)); % Do not allow very high precision 
                                        % or very low precision


                                        
% Complete function values for x0, x1
if done == 0
   f0 = Inf;
   f1 = Inf;
end
if done == 0 && x0 > -Inf
   f0 = f(x0); iterations = iterations+1;
end
if done == 0 && x1 < Inf
   f1 = f(x1); iterations = iterations+1;
end




% GENERATE FINTE BOUNDS

dt = 1 + abs(xact)*1.0E-3; 
% (for large xact an increment by one may be too small)
if x1 < Inf
   dt = max(dt,x1-xact);
end
if x0 > -Inf
   dt = max(dt,xact-x0);
end
dtsave = dt;


% Make one of the bounds finite
if x0 == -Inf && x1 == Inf % Here, it must be ``done = 0''
   xtmp = xact+dt;
   ftmp = f(xtmp); iterations = iterations+1;
   if ftmp < fact
      x0 = xact;
      f0 = fact;
      xact = xtmp;
      fact = ftmp;
   else
      x1 = xtmp;
      f1 = ftmp;
   end
end


% Generate a finite upper bound
if x1 == Inf && f0 >= fact % Here, it must be ``done = 0''
   xtmp = xact + dt;
   ftmp = f(xtmp); iterations = iterations+1;
   itcount = 0;
   while itcount < 15 && ftmp < fact % Line search up to length 10^15
      x0 = xact;
      f0 = fact;
      xact = xtmp;
      fact = ftmp;
      dt = dt*10;
      xtmp = xact + dt;
      ftmp = f(xtmp); iterations = iterations+1; itcount = itcount+1;
   end
   if itcount >= 15
      if ftmp < fact
         xact = xtmp;
         fact = ftmp;
      end
      out.iter = iterations;
      warning('line search may be unbounded (x to Inf)');
      done = 1;
   else % itcount < 15 means ftmp >= fact
      x1 = xtmp;
      f1 = ftmp;
   end
end

if x1 == Inf && f0 < fact % Here, it must be ``done = 0''
   x1 = xact;
   f1 = fact;
   xact = 0.5*(xact+x0);
   fact = f(xact); iterations = iterations +1;
end


% Generate a finite lower bound
dt = dtsave;
if x0 == -Inf && f1 >= fact && done == 0
   xtmp = xact - dt; 
   ftmp = f(xtmp); iterations = iterations+1;
   itcount = 0;
   while itcount < 15 && ftmp < fact % line search up to length 10^15
      x1 = xact;
      f1 = fact;
      xact = xtmp;
      fact = ftmp;
      dt = dt*10;
      xtmp = xact - dt;
      ftmp = f(xtmp); iterations = iterations+1; itcount = itcount+1;
   end
   if itcount >= 15
      if ftmp < fact
         xact = xtmp;
         fact = ftmp;
      end
      out.iter = iterations;
      warning('line search may be unbounded (x to -Inf)');
      done = 1;
   else % itcount < 15 means ftmp >= fact
      x0 = xtmp;
      f0 = ftmp;
   end
end

if x0 == -Inf && f1 < fact && done == 0
   x0 = xact;
   f0 = fact;
   xact = 0.5*(xact+x0);
   fact = f(xact); iterations = iterations +1;
end


int_length = x1-x0; % Length of the interval containing the minimizer
int_length = max(int_length,10*eps*(abs(x0)+abs(x1))/tol);
tol = max(tol, 10*eps*(1+abs(x0)+abs(x1))/int_length);


% Eliminate Inf-values of f
if min([f0,fact,f1]) == Inf
   warning('No finite value of f found in line search')
   done = 1;
end
itcount = 0;
if fact == Inf && done == 0
   if f1 < f0 % look for minimizer near x1
      while fact == Inf && itcount < 15
         itcount = itcount+1;
         x0 = xact; 
         f0 = fact;
         xact = 0.1*xact+0.9*x1;
         fact = f(xact); iterations = iterations+1;
      end
      if fact == Inf
         warning('Only infinite objective values found in line search')
         done = 1;
      end 
   else % look for minimizer near x0
      while fact == Inf && itcount < 15
         itcount = itcount+1;
         x1 = xact; 
         f1 = fact;
         xact = 0.1*xact+0.9*x0;
         fact = f(xact); iterations = iterations+1;
      end
      if fact == Inf
         warning('Only infinite objective values found in line search')
         done = 1;
      end 
   end
end % Now, fact is finite
if f1 == Inf && done == 0
   while f1 == Inf && itcount < 30
      x1old = x1;
      x1 = 0.5*(xact+x1);
      f1 = f(x1); iterations = iterations+1;
   end
   if f1 == Inf
      warning('Only infinite objective values found in line search')
      done = 1;
   else
      if f1 < min(f0,fact)
         x0 = xact;
         f0 = fact;
         xact = x1;
         fact = f1;
         x1 = x1old;
         f1 = Inf;
      end
   end
end
if f0 == Inf && done == 0
   while f0 == Inf && itcount < 30
      x0old = x0;
      x0 = 0.5*(xact+x0);
      f0 = f(x0); iterations = iterations+1;
   end
   if f0 == Inf
      error('Only infinite objective values found in line search')
   else
      if f0 < min(f1,fact)
         x1 = xact;
         f1 = fact;
         xact = x0;
         fact = f0;
         x0 = x0old;
         f0 = Inf;
      end
   end
end      




% MAKE SURE f0 AND f1 ARE LARGER THAN fact 

itcountmax = 9+round(-log(tol)/log(10)); % higher precision near end points
% Number of iterations to identify a minimizer near the boundary

if fact >= min(f0,f1) && done == 0
   itcount = 0;
   if f0 < f1
      while fact >= f0 && itcount < itcountmax
         x1 = xact; f1 = fact; itcount = itcount+1;
         xact = 0.9*x0+0.1*x1; fact  = f(xact); 
      end
   else
      while fact >= f1 && itcount < itcountmax
         x0 = xact; f0 = fact; itcount = itcount+1;
         xact = 0.1*x0+0.9*x1; fact  = f(xact); 
      end
   end
   iterations = iterations + itcount;
   if itcount >= itcountmax || x0 == xact || x1 == xact
      done = 1;
      if f0 < min(f1,fact)
         xact = x0;
         fact = f0;
      end
      if f1 < min(f0,fact)
         xact = x1;
         fact = f1;
      end
      out.iter = iterations;
   end
end
% Now, either ``x1-x0 <= tol*int_length'' or ``fact < min(f0,f1)''
if xact <= x0 || x1 <= xact || f0 < fact || f1 < fact
   if done == 0
      error( ' programming error in line search 1' );
   end
end




% NOW, THE ACTUAL LINE SEARCH APPROXIMATING A MINIMIZER IN [x0,x1]

itcount = 0;                      % iteration counter
gm6 = 2/(sqrt(5)+1);              % Golden mean ratio, this is about 0.6
xfval = [x0,xact,x1;f0,fact,f1];  % Record all values of the search
iact = 2;                         % Index of xact in xfval
lref = 0;                         % last refinement not used so far
    
          
   
while done == 0 % *** MAIN LOOP ***
   spline_int = 0;
   itcount = itcount+1;

   % One golden mean search step
   if x1 - xact > xact - x0
      xa = x1 + gm6*(xact-x1); 
      fa = f(xa); iterations  = iterations + 1;  
      xfval = [xfval(:,1:iact),[xa;fa],xfval(:,iact+1:end)];
      if fa <= fact
         x0 = xact; f0 = fact; xact = xa; fact = fa; iact = iact+1;
      else
         x1 = xa; f1 = fa;
      end      
   else
      xa = x0 + gm6*(xact-x0); 
      fa = f(xa); iterations  = iterations + 1;
      xfval = [xfval(:,1:iact-1),[xa;fa],xfval(:,iact:end)];
      if fa <= fact
         x1 = xact; f1 = fact; xact = xa; fact = fa;
      else
         x0 = xa; f0 = fa; iact = iact+1;
      end
   end % of golden mean step     
   if x1 - x0 <= tol*int_length
       done = 1;
   end
      
   
   % Test whether to use spline interpolation
   n = length(xfval);
   if n >=5 && done == 0
      % Check whether the spline would have predicted fact correctly
      % Find points close to xact on both sides (2 <= iact <= end-1)
      if n == 5
         indspl = [1:iact-1,iact+1:5]; % "Spline" with 4 interpolat. points
      else
%     Strategy: Choose 5 points, (if possible) two larger than xact, two
%     smaller, and the last one the closest to xact (among the rest).
         i_set = 0; % indspl not yet set
         if iact <= 2
            if iact <= 1
                error( 'programming error in line search 2')
            end
            indspl = [1,3,4,5,6]; 
            i_set = 1; % do not change indspl any more 
         end
         n = length(xfval);
         if iact >= n-1
            if iact >= n
                error( 'programming error in line search 3')
            end
            indspl = [n-5,n-4,n-3,n-2,n];
            i_set = 1; % do not change indspl any more
         end
         if i_set == 0 % now, 3 <= iact <= n-2
            indspl0 = [iact-2,iact-1,iact+1,iact+2]; 
            tmp     = [-Inf,xfval(1,:),Inf]; % Note: tmp(iact+1)=xact
            if tmp(iact+1)-tmp(iact-2) < tmp(iact+4)-tmp(iact+1)
               indspl = [iact-3,indspl0]; 
            else
               indspl = [indspl0,iact+3]; 
            end
         end
      end % Index for spline is set
      xx = xfval(1,indspl);
      ff = xfval(2,indspl);
      ss = find_spline(xx,ff); 
      [sact,i] = eval_spline(xact,ss,xx);
      ip1 = min(1+1,length(xx)-1);
      if abs(sact-fact) < 0.1*(abs(ss(3,i))+abs(ss(3,ip1)))*...
                          (xact-xx(i))*(xx(i+1)-xact)
          spline_int = 1; % Estimate of second derivative is 80% correct
      end
   end
      
      
   while spline_int == 1 && itcount < 100 && done == 0
   %          use minimizer of the spline function (possibly several steps)
      itcount = itcount + 1;
         
      % Recompute the spline including xact
      indspl = iact-2:iact+2; 
      n = length(xfval);
      if iact <= 1 || iact >= n
         error( 'programming error in line search 4')
      end
      if iact == 2
         indspl = 1:5;
      end
      if iact == n-1
         indspl = n-4:n;
      end
      xx = xfval(1,indspl);
      ff = xfval(2,indspl);
      ss = find_spline(xx,ff); 
          
      [t,tval,i] = min_spline(ss,xx);  
      % SPLINE MINIMIZER t in [xx(i),xx(i+1)]
      if t <= x0 || t >= x1 % x0 = xfval(1,iact-1),  x1 = xfval(1,iact+1)
         spline_int = 0;
      else %%% Spline interpolation may be o.k. %%%
             
         ixf = i+indspl(1)-1; % t in [xfval(1,ixf),xfval(1,ixf+1)]
         if t < xx(i) || t > xx(i+1) || t<xfval(1,ixf) || t>xfval(1,ixf+1)
            error(' programming error in spline-line search 5')
         end
         shift_t = 0.1*min(tol*int_length, xx(i+1)-xx(i));
         % the shift is (much) less than tol*int_length 
         t = max(t,xx(i  )+shift_t);
         t = min(t,xx(i+1)-shift_t);
         tnewton = xact-0.5*ss(2,3)/ss(3,3); % Newton step for spline
         safetygap = min(abs(t-tnewton),0.1*min(x1-xact,xact-x0));
         % If xact is close to x0 or x1 and t is close to xact on the other
         % side, convergence may be slow ==> move a bit away from xact
         if xact-x0 < 0.1*(x1-xact) && t > xact
            t = t+safetygap; 
         end 
         if x1-xact < 0.1*(xact-x0) && t < xact
            t = t-safetygap; 
         end 
         ft = f(t); iterations = iterations+1;
         epp = abs(tval-ft); % The approximation error at t
         ip1 = min(1+1,length(xx)-1);
         if epp > 0.2*(abs(ss(3,i))+abs(ss(3,ip1)))*(t-xx(i))*(xx(i+1)-t)
             spline_int = 0; % Prediction not so accurate
             %disp('return to golden mean search')
         end
         
         if epp < 100*eps*(abs(ft)+1)
            done = 1; % Prediction is close to machine precision
            spline_int = 1; % keep spline interpolation for derivatives
         end
         epp = 2*epp/((t-xx(i))*(xx(i+1)-t)); % Second der. of the error
         mss = ((xx(i+1)-t)*ss(3,i)+(t-xx(i))*ss(3,ip1))/(xx(i+1)-xx(i));
         if mss > epp
            tmp = max(0.1*(xx(i+1)-xx(i)), abs(0.5*(xx(i+1)+xx(i))-t) );
            if tmp*epp/(mss-epp) < int_length*tol
               done = 1; % prediction of error in t is small
               spline_int = 1; % keep spline interpolation for derivatives
            end
         end
         xfval = [xfval(:,1:ixf),[t;ft],xfval(:,ixf+1:end)];
         if t < xact
            if ft > fact
               iact = iact+1; % xact remains same but iact increases by 1
            end
         else
            if ft < fact
               iact = iact + 1; % xact changes and also iact increases by 1
            end
         end
            
         x0   = xfval(1,iact-1);
         f0   = xfval(2,iact-1);
         xact = xfval(1,iact  ); 
         fact = xfval(2,iact  );
         x1   = xfval(1,iact+1);
         f1   = xfval(2,iact+1);
         if x1-x0 <= tol*int_length
            done = 1;
            spline_int = 1; % keep spline interpolation for derivatives
         end
      end %%% Case where spline interpolation may be o.k. %%%
      dddx = xfval(1,2:end)-xfval(1,1:end-1);
      if min(dddx) < 10*eps*(1+abs(x0)+abs(x1))
         done = 1;
      end   
   end   
   if x1-x0 <= tol*int_length || itcount >= 100
      done = 1;
   end 
   dddx = xfval(1,2:end)-xfval(1,1:end-1);
   if min(dddx) < 10*eps*(1+abs(x0)+abs(x1))
      done = 1;
   end
   
   if done == 1 % do a (final?) check
      check = 0;
      acc = max(x1-xact,xact-x0);
      if acc > int_length*tol
         check = 1;
      end
      if fact >= min(f0,f1)
         check = 0;
      end
      if min(x1-xact,xact-x0) <= 0
         check = 0;
      end
      if check > 0
         slope0 = (f0-fact)/(x0-xact); 
         slope1 = (f1-fact)/(x1-xact); 
         t0 = 0.5*(x0+xact);
         t1 = 0.5*(x1+xact);
         topt = t0-slope0*(t1-t0)/(slope1-slope0);
         if abs(xact-topt) < int_length*tol
             check = 0;
             %acc = abs(xact-topt);
         end
      end
      if check > 0
         if x1-xact > xact-x0
            t = xact + 0.9*int_length*tol;
            ft = f(t); iterations = iterations+1;
            xfval = [xfval(:,1:iact),[t;ft],xfval(:,iact+1:end)];
            if ft < fact
               done = 0;
               lref = 1;
               iact = iact + 1; % xact changes and also iact increases by 1
               x0   = xact;
               f0   = fact;
               xact = t;
               fact = ft;
            else
               x1 = t;
               f1 = ft;
            end
         else % we have x1-xact <= xact-x0
            t = xact - 0.9*int_length*tol;
            ft = f(t); iterations = iterations+1;
            xfval = [xfval(:,1:iact-1),[t;ft],xfval(:,iact:end)];
            iact = iact+1;
            if ft < fact
               done = 0;
               lref = 1;
               iact = iact - 1; % xact changes and also iact increases by 1
               x1   = xact;
               f1   = fact;
               xact = t;
               fact = ft;
            else
               x0 = t;
               f0 = ft;
            end
         end
         acc = max(x1-xact,xact-x0);
         if done == 1 && acc > int_length*tol
         % repeat the above (automatically this is the other side)
            if x1-xact > xact-x0
               t = xact + 0.9*int_length*tol;
               ft = f(t); iterations = iterations+1;
               xfval = [xfval(:,1:iact),[t;ft],xfval(:,iact+1:end)];
               if ft < fact
                  done = 0;
                  lref = 1;
                  iact = iact + 1; % xact changes, also iact decreases by 1
                  x0   = xact;
                  f0   = fact;
                  xact = t;
                  fact = ft;
               else
                  x1 = t;
                  f1 = ft;
               end
            else % we have x1-xact <= xact-x0
               t = xact - 0.9*int_length*tol;
               ft = f(t); iterations = iterations+1;
               xfval = [xfval(:,1:iact-1),[t;ft],xfval(:,iact:end)];
               iact = iact+1;
               if ft < fact
                  done = 0;
                  lref = 1;
                  iact = iact - 1; % xact changes, also iact decreases by 1
                  x1   = xact;
                  f1   = fact;
                  xact = t;
                  fact = ft;
               else
                  x0 = t;
                  f0 = ft;
               end
            end
         end
      end
   end
end % *** OF MAIN LOOP ***
x = xact; y = fact; out.iter = iterations;
spline_int = max(spline_int, lref);


if spline_int == 1 && y > -Inf
   n = length(xfval);
   if xact < xfval(1,1) || xact > xfval(1,n)
      warning(' final point out of range in spline evaluation ');
      g = NaN; % No derivative information available from Spline
      H = NaN; % (Use finite difference instead - not done here)
   else
      % Recompute the spline including xact
      indspl = iact-2:iact+2; 
      if iact <= 1 || iact >= n
         error( 'programming error in line search 6')
      end
      if iact == 2
         indspl = 1:5;
      end
      if iact == n-1
         indspl = n-4:n;
      end
      xx = xfval(1,indspl);
      ff = xfval(2,indspl);
      ss = find_spline(xx,ff); 
      i = 1;
      while xact > xx(1,i+1)
         i = i+1;
      end
      t = xact-xx(1,i);
      
      g = ss(2,i)+t*(2*ss(3,i)+t*(3*ss(4,i)));
      H = 2*ss(3,i)+t*(6*ss(4,i));
   end
else
   g = NaN; % No derivative information available from Spline
   H = NaN; % (Use finite difference instead - not done here)
end

acc = max(x1-xact,xact-x0);
if spline_int == 1
   if isnan(g)
      out.acc = acc;
   else
      out.acc = min(abs(g)/(eps+abs(H)),acc); 
   end
else
   out.acc = acc;
end

end








function s = find_spline(x,f)
% Find the least squares spline interpolant through the points x(i), f(i)
% assuming that there are at least 4 interpolation points.
% On each interval [x(j),x(j+1)] (1<=j<=n-1) the spline is represented as
% s(x) = s(1,j) + s(2,j)*(x-x(j)) + s(3,j)*(x-x(j))^2 + s(4,j)*(x-x(j))^3

% Initialize
[n,m] = size(x);
if n < m % make x a column vector
   x = x.'; tmp = n; n = m; m = tmp;
end
if m > 1 || m == 0 || n < 4
  error('input x for spline interpolation is inconsistent');
end

[nn,mm] = size(f);
if nn < mm % make f a column vector
   f = f.'; tmp = nn; nn = mm; mm = tmp;
end
if mm > 1 || mm == 0
  error('input f for spline interpolation is inconsistent');
end

if n ~= nn
  error('input x,f for spline interpolation is inconsistent');
end

dx = x(2:n)-x(1:n-1); % vector of increments of x
if min(dx) <= 0
error('vector of spline base points is assumed to be in increasing order');
end

s  = zeros(12,n); % Three splines with n-1 cubic parts each.
% The last column is a dummy to allow assigning f as the first row.
% The first spline interpolates f,
% the second and third interpolate the zero function.
% rows 1 to 4  for the spline interpolating f
% rows 5 to 8  for the spline interpolating zero; nonzero linear first term
% rows 9 to 12 for the spline interpolating zero; nonzero quadr. first term
t1 = [1,5, 9]; % indices for the constant terms
t2 = [2,6,10]; % indices for the linear terms
t3 = [3,7,11]; % indices for the quadratic terms
t4 = [4,8,12]; % indices for the cubic terms

s(1,:) = f; % the constant part of s(1:4,:) is given by f
            % (constant parts of the zero splines are zero)


% Set up the three splines; first the splines on [x(1),x(2)]
s(2,1)  = (f(2)-f(1))/dx(1); % first spline, linear on [x(1),x(2)]
s(6,1)  = 1;                 % second spline, first linear term is 1
s(8,1)  = -1/dx(1)^2;        % interoplate to zero at x(2)
s(11,1) = 1;                 % third spline, first quadratic term is 1
s(12,1) = -1/dx(1);          % interoplate to zero at x(2)

% then the interpolating splines on the remaining subintervals
for i = 2:n-1
   s(t2,i) = s(t2,i-1)+dx(i-1)*(2*s(t3,i-1)+3*dx(i-1)*s(t4,i-1));
   s(t3,i) = s(t3,i-1)+3*dx(i-1)*s(t4,i-1);
   s(t4,i) = (s(t1,i+1)-s(t1,i)-dx(i)*(s(t2,i)+dx(i)*s(t3,i)))/dx(i)^3;
end
s = s(:,1:n-1); % now remove the last column

% The norm is given by || D * \Delta * s(4,:)' ||_2 where, formally,
% D      = Diag(((dx(2:n-1)+dx(1:n-2)).^-.5); and
% \Delta = [-eye(n-2),zeros(n-2,1)]+[zeros(n-2,1),eye(n-2)];
% i.e. a weighted sum of squared jumps of the third derivatives

d  = (dx(2:n-1)+dx(1:n-2)).^-.5; % weights
Ds = (kron(ones(3,1),d.').*(s(t4,1:n-2)-s(t4,2:n-1))).';
% Ds contains the weighted jumps in the cubic terms of the splines

% orthogonalize the third with respect to the second spline
alpha     = Ds(:,3).'*Ds(:,2)/(Ds(:,2).'*Ds(:,2));
s(9:12,:) = s(9:12,:)-alpha*s(5:8,:); % updating s(10:12,:) would suffice
Ds(:,3)   = Ds(:,3)  -alpha*Ds(:,2);

% adjust first with the second spline
alpha     = Ds(:,1).'*Ds(:,2)/(Ds(:,2).'*Ds(:,2));
s(1:4,:)  = s(1:4,:)-alpha*s(5:8,:); % updating s(2:4,:) would suffice
Ds(:,1)   = Ds(:,1) -alpha*Ds(:,2);

% adjust first with the third spline
alpha     = Ds(:,1).'*Ds(:,3)/(Ds(:,3).'*Ds(:,3));
s(1:4,:)  = s(1:4,:)-alpha*s(9:12,:); % updating s(2:4,:) would suffice
%Ds(:,1)   = Ds(:,1) -alpha*Ds(:,3); % not needed

% to reduce rounding errors redo the final spline
s = s(1:4,:);
for i = 2:n-1
   s(2,i) = s(2,i-1)+dx(i-1)*(2*s(3,i-1)+3*dx(i-1)*s(4,i-1));
   s(3,i) = s(3,i-1)+3*dx(i-1)*s(4,i-1);
   s(4,i) = (f(i+1)-f(i)-dx(i)*(s(2,i)+dx(i)*s(3,i)))/dx(i)^3;
end
end






function [y,i] = eval_spline(xact,s,x)
% Evaluate the spline given by s and the partition x at the point xact
% Also return the segment in which xact is located
% Assume x in increasing order, length(x) = n and s is 1:4 by 1:n-1

n = length(x);

if xact < x(1) || xact > x(n)
    error(' point out of range in spline evaluation ');
end
i = 1;
while xact > x(i+1)
   i = i+1;
end

t = xact-x(i);
y = s(1,i)+t*(s(2,i)+t*(s(3,i)+t*(s(4,i))));
end







function [t,y,imin] = min_spline(s,x)
% Find the minimum of the spline given by s in the interval [x(1),x(end)]
% The minimizer t with value y in the interval [x(imin),x(imin+1)]
% Assume x in increasing order, length(x) = n and s is 1:4 by 1:n-1

n = length(x);

% Test support points first
[y,imin] = min(s(1,:)); t=x(imin);
dx = x(n)-x(n-1);
y_last = s(1,n-1)+dx*(s(2,n-1)+dx*(s(3,n-1)+dx*(s(4,n-1))));
if y_last < y
   y = y_last; t = x(n); imin = n-1; % Not imin = n, so x(imin+1) exists
end

% y is the smallest value so far
for i = 1:n-1
   v = s(:,i); % just for convenience
   if v(4) ~= 0 
      discr = v(3)^2-3*v(4)*v(2);
      if discr >= 0
         tmp = v(3)+sign(v(3))*sqrt(discr);
         t1 = -tmp/(3*v(4));
         if t1 > 0 && t1 < x(i+1)-x(i)
            y1 = s(1,i)+t1*(s(2,i)+t1*(s(3,i)+t1*(s(4,i))));
            if y1 < y
               y = y1; t = x(i)+t1; imin = i;
            end
         end
         t1 = -v(2)/tmp;
         if t1 > 0 && t1 < x(i+1)-x(i)
            y1 = s(1,i)+t1*(s(2,i)+t1*(s(3,i)+t1*(s(4,i))));
            if y1 < y
               y = y1; t = x(i)+t1; imin = i;
            end
         end
      end
   else
      if v(3) ~= 0 % when this is instable then the minimizer is outside 
                   % the current interval
         t1 = -0.5*v(2)/v(3);
         if t1 > 0 && t1 < x(i+1)-x(i)
            y1 = s(1,i)+t1*(s(2,i)+t1*(s(3,i)+t1*(s(4,i))));
            if y1 < y
               y = y1; t = x(i)+t1; imin = i;
            end
         end
      end
   end
end
end



