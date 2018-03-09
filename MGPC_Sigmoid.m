function [thetabold, ActSet, ExpertP, TestEst, TestEst2]=MGPC_Sigmoid(TrainX,TrainY,TestX,TestY, T, MaxOuterIteration, MaxEMite, MaxIteration_VI, MaxIteration_CG,rand_state)
% ????????????D*N
rand('state',rand_state); %??????????????
% rng('default');
% rng(2);
%-----------------------------------------------preprocessing
[TrainX, MInput, StdVariInput] = funcScaleSSL(TrainX');
TestX=TestX'-repmat(MInput,size(TestX',1),1);
TestX=TestX./(eps+repmat(StdVariInput,size(TestX,1),1));
TrainX=TrainX'; TestX=TestX';
MOutput=mean(TrainY);
TrainY=TrainY-MOutput;
TestY =TestY-MOutput;
stdY=std(TrainY);
TrainY=TrainY/stdY; TestY=TestY/stdY;

Testn=length(TestY);
TestEst=ones(MaxOuterIteration,Testn);
TestEst2=ones(MaxOuterIteration,Testn);
%================================================Variational EM for Learning MGP
%---initialize the parameters and hyperparameters
alpha0=1; d=size(TrainX,1);
SigmaX=cov(TrainX');
SigmaX=repairPD(SigmaX);
CholFa=chol(SigmaX);
Rx=CholFa\(CholFa'\eye(d));
mu0=zeros(d,1);      R0=Rx;
nu0=d;   W0=Rx/d;    W0inv=d*SigmaX;
a0=1e-2; b0=1e-4;
N=length(TrainY);
% T=2;
M=round(N/T); % T: maximum # of GPs; Mupper: maximum # of active set points for one GP
temp=[1 ones(1,d) 0.05]; %[sigma_f sigma_1 ... sigma_d sigma_b]
thetabold=repmat(temp, T, 1);
ActCandSize=M;         %randomly pick 100 examples as the actset candidate
ActCand=ones(T,ActCandSize);
for ii=1:T
    temp=randperm(N);
    ActCand(ii,:)=temp(1:ActCandSize);
end
%---k-means clustering, and initialize qzi
[IDX, Centroid]=kmeans([TrainX; TrainY]',T);
IDXT=cell(T,1); ActSet=cell(T,1); qzi=zeros(T,N);
for ii=1:T
    IDXT{ii}=find(IDX==ii);
    if length(IDXT{ii}) < (M-0.5)
        temp=randperm(N);
        IDXT{ii}=[ IDXT{ii}; temp(1:(M-length(IDXT{ii})))'];
        %         IDXT{ii} = temp(1:M)';
    end
    temp=randperm(length(IDXT{ii}));
    ActSet{ii}=IDXT{ii}(temp(1:M));
    
    qzi(ii,IDXT{ii})=1;
end
ActSetOld=ActSet;
qzi=qzi./(repmat(sum(qzi,1),T,1)); %posterior
IDXNAN = find(isnan(qzi));
qzi(IDXNAN) = 1/T;
%---initialize other posterior distributions
qnuk=zeros(T-1,2); qmuk=cell(T,1); qRk=cell(T,1); qwk=cell(T,1); qrk=cell(T,1);
% Initialization for q_f_n
qfn = zeros(N,2); %qfn(:,1) mu; qfn(:,2) sigma;
phi_T_xn = cell(T,1);

for ii=1:T
    if ii<T
        temp=qzi(ii+1:T, :);
        nu_t1=sum(temp(:)); nu_t2=sum(qzi(ii, :));
        qnuk(ii,:)=[nu_t2+1 nu_t1+alpha0]; %posterior
    end
    Rk1=Rx; Rk2=Rk1*(TrainX*qzi(ii,:)'); Rk3=Rk1*sum(qzi(ii,:));
    tempM = repairPD(R0+Rk3);
    CholFa=chol(tempM);
    R0Rk3inv=CholFa\(CholFa'\eye(d));
    qmuk{ii}.mu=R0Rk3inv*(R0*mu0+Rk2); %posterior
    qmuk{ii}.R=R0+Rk3;                 %posterior
    qmuk{ii}.Sigma=R0Rk3inv;
    
    muk1=sum(qzi(ii,:));
    mukn=repmat(qmuk{ii}.mu, 1, N)-TrainX;
    muk2=muk1*R0Rk3inv + (mukn.*repmat(qzi(ii,:),d,1))*mukn';
    
    tempM = repairPD(W0inv+muk2);
    CholFa=chol(tempM);
    qRk{ii}.W=CholFa\(CholFa'\eye(d)); %posterior
    qRk{ii}.nu=nu0+muk1;               %posterior
    
    rk1=a0/b0;
    Mmat=diag(thetabold(ii,2:end-1).^(-2));
    Mdist=Mdiagonal_distance(TrainX,TrainX(:,ActSet{ii}),Mmat);
    phi_k_xn=thetabold(ii,1)^2*exp(-(Mdist.^2)/2); phi_k_xn=phi_k_xn';
    phi_T_xn{ii}=phi_k_xn;
    rk2=rk1*phi_k_xn*(qzi(ii,:)'.*TrainY');
    rk3=rk1*(phi_k_xn.*repmat(qzi(ii,:),M,1))*phi_k_xn';
    qwk{ii}.R=phi_k_xn(:,ActSet{ii})+thetabold(ii,end)^2*eye(M)+rk3; %posterior
    tempM = repairPD(qwk{ii}.R);
    CholFa=chol(tempM);
    qwkSigma=CholFa\(CholFa'\eye(M));
    qwk{ii}.mu=qwkSigma*rk2;                                      %posterior
    qwk{ii}.Sigma=qwkSigma;
    
    wk1=muk1/2;
    Ewk_square=diag(phi_k_xn'*(qwkSigma+qwk{ii}.mu*qwk{ii}.mu')*phi_k_xn) + (TrainY.^2)' - 2*(repmat(TrainY', 1, M).*phi_k_xn')*qwk{ii}.mu;
    %     Ewk_square=diag(phi_k_xn'*(qwkSigma+qwk{ii}.mu*qwk{ii}.mu')*phi_k_xn) + (qfn(:,1).^2)+qfn(:,2) - 2*(repmat(qfn(:,1), 1, M).*phi_k_xn')*qwk{ii}.mu;
    %
    wk2=qzi(ii,:)*Ewk_square/2;
    qrk{ii}.a=a0+wk1;  %posterior
    qrk{ii}.b=b0+wk2;  %posterior
    qrk{ii}.Ewk_s=Ewk_square;
    
    
end

for ii = 1:N
    for jj = 1:T
        %         qfn(ii,1) = qfn(ii,1) + qzi(jj, ii)*(phi_T_xn{jj}(:,ii))'*qwk{jj}.mu;
        qfn(ii,1) = TrainY(ii);
        qfn(ii,2) = 0.1;
    end
end

%---if active sets during two EM processes change, do another EM process
% MaxOuterIteration=10;%maximum # of updating active sets
for outerstep=1:MaxOuterIteration
    disp(['OuterStep (updating active set)= ' num2str(outerstep) '...']);
    if outerstep>1.5
        tempdis=[];
        for ttt=1:T
            tempdis=tempdis+norm( sort(ActSet{ttt})-sort(ActSetOld{ttt}) );
        end
        if tempdis<1
            break;
        end
    end
    %==EM iterates
    tic;
    %     MaxEMite=50;
    for EMite=1:MaxEMite
        %disp(['..EM iteration= ' num2str(EMite) '...']);
        %----------iteratively update the variational distribution: E-step
        %MaxIteration_VI=50; %maximum number of variational inference during one E-step
        for VInum=1:MaxIteration_VI
            %disp(['...E-step (updating variational posterior)...VInum= ' num2str(VInum) '...']);
            tempa=psi(qnuk(:,1)); tempb=psi(qnuk(:,2)); tempab=psi(qnuk(:,1)+qnuk(:,2));
            Eln_nut=tempa-tempab; Eln_nut=[Eln_nut; 0]; %T rows, 1 column
            Eln_oneminusnut=tempb-tempab;               %T-1 rows, 1 column
            lnront=zeros(T,N);
            for ii=1:T
                if ii==1
                    first4parts=Eln_nut(ii) + (sum(psi((qRk{ii}.nu+1-[1:d])/2))+d*log(2)+log(det(qRk{ii}.W)) + psi(qrk{ii}.a)-log(qrk{ii}.b))/2;
                else
                    first4parts=Eln_nut(ii) + sum(Eln_oneminusnut(1:ii-1)) + (sum(psi((qRk{ii}.nu+1-[1:d])/2))+d*log(2)+log(det(qRk{ii}.W)) + psi(qrk{ii}.a)-log(qrk{ii}.b))/2;
                end
                temp=zeros(1,N);
                for jj=1:N
                    temp(jj)=-trace( qRk{ii}.nu*qRk{ii}.W* (qmuk{ii}.Sigma+(qmuk{ii}.mu-TrainX(:,jj))*(qmuk{ii}.mu-TrainX(:,jj))') )/2;
                end
                lnront(ii,:)=first4parts+temp-qrk{ii}.a/qrk{ii}.b/2*qrk{ii}.Ewk_s';
            end
            lnront=exp(lnront);
            qziold=qzi;
            qzi=lnront./(eps+repmat(sum(lnront,1),T,1)); %posterior
            qzidiff=qziold-qzi;
            if VInum>1.5 & norm(qzidiff,'fro')<1e-5
                qzi=qziold;
                break; %finish updating variational posterior
            end
            %=update other posterior distributions
            for ii=1:T
                if ii<T
                    temp=qzi(ii+1:T, :);
                    nu_t1=sum(temp(:)); nu_t2=sum(qzi(ii, :));
                    qnuk(ii,:)=[nu_t2+1 nu_t1+alpha0]; %posterior
                end
                Rk1=qRk{ii}.nu * qRk{ii}.W; Rk2=Rk1*(TrainX*qzi(ii,:)'); Rk3=Rk1*sum(qzi(ii,:));
                tempM = repairPD(R0+Rk3);
                CholFa=chol(tempM);
                R0Rk3inv=CholFa\(CholFa'\eye(d));
                qmuk{ii}.mu=R0Rk3inv*(R0*mu0+Rk2); %posterior
                qmuk{ii}.R=R0+Rk3;                 %posterior
                qmuk{ii}.Sigma=R0Rk3inv;
                
                muk1=sum(qzi(ii,:));
                mukn=repmat(qmuk{ii}.mu, 1, N)-TrainX;
                muk2=muk1*R0Rk3inv + (mukn.*repmat(qzi(ii,:),d,1))*mukn';
                tempM = repairPD(W0inv+muk2);
                CholFa=chol(tempM);
                qRk{ii}.W=CholFa\(CholFa'\eye(d)); %posterior
                qRk{ii}.nu=nu0+muk1;               %posterior
                
                rk1=qrk{ii}.a/qrk{ii}.b;
                Mmat=diag(thetabold(ii,2:end-1).^(-2));
                Mdist=Mdiagonal_distance(TrainX,TrainX(:,ActSet{ii}),Mmat);
                phi_k_xn=thetabold(ii,1)^2*exp(-(Mdist.^2)/2); phi_k_xn=phi_k_xn';
                %                 rk2=rk1*phi_k_xn*(qzi(ii,:)'.*TrainY');
                % Update rk2 using qfn instead.
                rk2=rk1*phi_k_xn*(qzi(ii,:)'.*qfn(:,1));
                rk3=rk1*(phi_k_xn.*repmat(qzi(ii,:),M,1))*phi_k_xn';
                qwk{ii}.R=phi_k_xn(:,ActSet{ii})+thetabold(ii,end)^2*eye(M)+rk3; %posterior
                tempM =repairPD(qwk{ii}.R);
                CholFa=chol(tempM);
                qwkSigma=CholFa\(CholFa'\eye(M));
                qwk{ii}.mu=qwkSigma*rk2;                                         %posterior
                qwk{ii}.Sigma=qwkSigma;
                
                wk1=muk1/2;
                Ewk_square=diag(phi_k_xn'*(qwkSigma+qwk{ii}.mu*qwk{ii}.mu')*phi_k_xn) + (qfn(:,1).^2)+qfn(:,2) - 2*(repmat(qfn(:,1), 1, M).*phi_k_xn')*qwk{ii}.mu;
                wk2=qzi(ii,:)*Ewk_square/2;
                qrk{ii}.a=a0+wk1;  %posterior
                qrk{ii}.b=b0+wk2;  %posterior
                qrk{ii}.Ewk_s=Ewk_square;
            end
            
            %             options = optimoptions('fminunc','SpecifyObjectiveGradient',true,'DerivativeCheck','on','Display','iter-detailed'); % indicate gradient is provided
            %             options = optimoptions('fminunc','SpecifyObjectiveGradient',true,'DerivativeCheck','off','Display','off'); % indicate gradient is provided
            options = optimoptions('fminunc','GradObj','on','Algorithm','trust-region','Display','iter-detailed');
            qfn_temp = qfn;
            qfn_temp(:,2) = sqrt(qfn_temp(:,2));
            fn_parameters = qfn_temp(:);
            %             for kk = 1:N
            % %             fn_parameters = fminunc(@(w)Lqf(w, qzi, qrk, qwk, phi_T_xn, TrainY), fn_parameters, options);
            %
            %                 options_minfunc.Method = 'lbfgs';
            %                 options_minfunc.DerivativeCheck = 'on';
            %                 options_minfunc.Display = '(iter)';
            %                 lambdaVal = 0.2;
            %                 lambda = lambdaVal*ones(1,1);
            %                 reglaFunObj = @(w)penalizedL2(w,@Lqfn, lambda, qzi, qrk, qwk, phi_T_xn, TrainY(ii), kk, M);
            % %                 @(w)penalizedL2(w,@UGM_HCRF_NLL,lambda,Xnode,Xedge,y,nodeMap, edgeMap, labelMap, edgeStruct,@UGM_Infer_logChain);
            %
            % %                 fn_parameters([ii,N+ii]) = minFunc(@(w)Lqf(w, qzi, qrk, qwk, phi_T_xn, TrainY(ii)), fn_parameters([ii,N+ii]),options_minfunc);
            %                 fn_parameters([kk,N+kk]) = minFunc(reglaFunObj, fn_parameters([kk,N+kk]),options_minfunc);
            %             end
            options_minfunc.Method = 'lbfgs';
            options_minfunc.DerivativeCheck = 'off';
            options_minfunc.Display = 'off';
            lambdaVal = 0;
            lambda = lambdaVal*ones(size(fn_parameters));
            reglaFunObj = @(w)penalizedL2(w,@Lqf, lambda, qzi, qrk, qwk, phi_T_xn, TrainY);
            %                 @(w)penalizedL2(w,@UGM_HCRF_NLL,lambda,Xnode,Xedge,y,nodeMap, edgeMap, labelMap, edgeStruct,@UGM_Infer_logChain);
            
            %                 fn_parameters([ii,N+ii]) = minFunc(@(w)Lqf(w, qzi, qrk, qwk, phi_T_xn, TrainY(ii)), fn_parameters([ii,N+ii]),options_minfunc);
            fn_parameters = minFunc(reglaFunObj, fn_parameters,options_minfunc);
            qfn_temp = reshape(fn_parameters, size(qfn));
            qfn_temp(:,2) = qfn_temp(:,2).^2;
            qfn = qfn_temp;
            
        end
        %----------update hyperparameters: M-step
        %         MaxIteration_CG=50; %maximum number of conjugate gradient ascent during one M-step
        thetabold_old=thetabold;
        for ii=1:T
            %disp(['...M-step (updating theta)...Expert= ' num2str(ii) '...']);
            pUkpTheta=cell(d+2,1);
            pPhi_k_xnpTheta=cell(d+2,1);
            A_g=qwk{ii}.Sigma + qwk{ii}.mu*qwk{ii}.mu';
            C_g=A_g;
            b_g=qrk{ii}.a/qrk{ii}.b;
            %-conjugate gradient ascent
            for CGstep=1:MaxIteration_CG
                %disp(['......M-step (updating theta)...CGstep= ' num2str(CGstep) '...']);
                g=ones(d+2,1);
                H=zeros(d+2,d+2);
                for jj=1:d+2
                    if jj==1
                        Mmat=diag(thetabold(ii,2:end-1).^(-2));
                        Mdist=Mdiagonal_distance(TrainX,TrainX(:,ActSet{ii}),Mmat);
                        phi_k_xn=thetabold(ii,1)^2*exp(-(Mdist.^2)/2); phi_k_xn=phi_k_xn';
                        Uk=phi_k_xn(:,ActSet{ii})+thetabold(ii,end)^2*eye(M);
                        tempM = repairPD(Uk);
                        CholFa=chol(tempM); Ukinv=CholFa\(CholFa'\eye(M));
                        
                        Mdist=Mdist';
                        pPhi_k_xnpTheta{jj}=2*thetabold(ii,1)*exp(-(Mdist.^2)/2);                       %partial derivative
                        pUkpTheta{jj}=pPhi_k_xnpTheta{jj}(:, ActSet{ii});                               %partial derivative
                    elseif jj==d+2
                        pUkpTheta{jj}=2*thetabold(ii,end)*eye(M); %partial derivative
                        pPhi_k_xnpTheta{jj}=zeros(M,N);           %partial derivative
                    else
                        tempx1=TrainX(jj-1,ActSet{ii}); tempx2=TrainX(jj-1,:); tempdis=L2_distance(tempx1,tempx2); tempdis=tempdis.^2;
                        pPhi_k_xnpTheta{jj}= phi_k_xn.*tempdis/(thetabold(ii,jj)^3);                    %partial derivative
                        pUkpTheta{jj}=pPhi_k_xnpTheta{jj}(:,ActSet{ii});                                %partial derivative
                    end
                    %Solve gradient g
                    first2part=trace((Ukinv-A_g)*pUkpTheta{jj});
                    %                     lastpart=diag(pPhi_k_xnpTheta{jj}'*C_g*phi_k_xn)-TrainY'.*(pPhi_k_xnpTheta{jj}'*qwk{ii}.mu);
                    lastpart=diag(pPhi_k_xnpTheta{jj}'*C_g*phi_k_xn)-qfn(:,1).*(pPhi_k_xnpTheta{jj}'*qwk{ii}.mu);
                    lastpart=2*b_g*qzi(ii,:)*lastpart;
                    g(jj)=first2part-lastpart;
                end
                if norm(g)<1e-5
                    break; %zero gradient, then finish the CG algorithm
                end
                
                p2Ukp2Theta=cell(d+2,d+2);
                p2Phi_k_xnp2Theta=cell(d+2,d+2);
                for jj=1:d+1
                    for kk=jj:d+1
                        if jj==1 & kk==1
                            p2Phi_k_xnp2Theta{1,1}=2*exp(-(Mdist.^2)/2);
                            p2Ukp2Theta{1,1}=p2Phi_k_xnp2Theta{1,1}(:,ActSet{ii});
                        end
                        if jj==1 & kk>1
                            tempx1=TrainX(kk-1,ActSet{ii}); tempx2=TrainX(kk-1,:); tempdis=L2_distance(tempx1,tempx2); tempdis=tempdis.^2;
                            p2Phi_k_xnp2Theta{1,kk}=thetabold(ii,1)*p2Phi_k_xnp2Theta{1,1}.*tempdis/(thetabold(ii,kk)^3);
                            p2Ukp2Theta{1,kk}=p2Phi_k_xnp2Theta{1,kk}(:,ActSet{ii});
                        end
                        if jj>1  & kk==jj
                            tempx1=TrainX(kk-1,ActSet{ii}); tempx2=TrainX(kk-1,:); tempdis=L2_distance(tempx1,tempx2); tempdis=tempdis.^2;
                            p2Phi_k_xnp2Theta{jj,kk}=thetabold(ii,1)^2*tempdis.*(p2Phi_k_xnp2Theta{1,1}/2).*(tempdis/(thetabold(ii,kk)^6) - 3/(thetabold(ii,kk)^4));
                            p2Ukp2Theta{jj,kk}=p2Phi_k_xnp2Theta{jj,kk}(:,ActSet{ii});
                        end
                        if jj>1 & kk>jj
                            tempx1=TrainX(kk-1,ActSet{ii}); tempx2=TrainX(kk-1,:); tempdis=L2_distance(tempx1,tempx2); tempdis=tempdis.^2;
                            tempx1=TrainX(jj-1,ActSet{ii}); tempx2=TrainX(jj-1,:); tempdis2=L2_distance(tempx1,tempx2); tempdis2=tempdis2.^2;
                            p2Phi_k_xnp2Theta{jj,kk}=thetabold(ii,1)^2*(p2Phi_k_xnp2Theta{1,1}/2).*tempdis.*tempdis2/(thetabold(ii,jj)^3)/(thetabold(ii,kk)^3);
                            p2Ukp2Theta{jj,kk}=p2Phi_k_xnp2Theta{jj,kk}(:,ActSet{ii});
                        end
                    end
                end
                for jj=1:d+2
                    p2Phi_k_xnp2Theta{jj,d+2}=zeros(M,N);
                    p2Ukp2Theta{jj,d+2}=p2Phi_k_xnp2Theta{jj,d+2}(:,ActSet{ii});
                    if jj==d+2
                        p2Ukp2Theta{jj,d+2}=2*eye(M);
                    end
                end
                %solve H. Pay attention to the symmetry property
                for jj=1:d+2
                    for kk=jj:d+2
                        first2part=trace((Ukinv-A_g)*p2Ukp2Theta{jj,kk} - Ukinv*pUkpTheta{kk}*Ukinv*pUkpTheta{jj});
                        %                         lastpart=diag( p2Phi_k_xnp2Theta{jj,kk}'*C_g*phi_k_xn + pPhi_k_xnpTheta{kk}'*C_g*pPhi_k_xnpTheta{jj} ) - TrainY'.*(p2Phi_k_xnp2Theta{jj,kk}'*qwk{ii}.mu);
                        lastpart=diag( p2Phi_k_xnp2Theta{jj,kk}'*C_g*phi_k_xn + pPhi_k_xnpTheta{kk}'*C_g*pPhi_k_xnpTheta{jj} ) - qfn(:,1).*(p2Phi_k_xnp2Theta{jj,kk}'*qwk{ii}.mu);
                        lastpart=2*b_g*qzi(ii,:)*lastpart;
                        H(jj,kk)=first2part-lastpart;
                    end
                end
                %make use of the symmetry
                H=H+H'-diag(diag(H));
                
                if CGstep==1
                    u_dir=g;
                    u_last=u_dir;
                    g_last=g;
                else
                    beta_dir=g'*(g-g_last)/(eps+u_last'*(g-g_last));
                    u_dir=g-beta_dir*u_last;
                    u_last=u_dir;
                    g_last=g;
                end
                temp=thetabold(ii,:)' - (g'*u_dir)*u_dir/(eps+u_dir'*H*u_dir);
                thetabold(ii,:)=abs(temp'); %the standard variance used is positive, though the negative counterpart is also computationally equivalent.
            end
        end
        if norm(thetabold_old-thetabold, 'fro')<1e-5
            thetabold=thetabold_old;
            break; %EM coverges
        end
    end
    toc;
    %----------after EM converges, update active set.
    ActSetOld=ActSet;
    %----------If active set changes, also update other parameters/hyperparameters by running EM again
    tic
    for ii=1:T
        rk1=qrk{ii}.a/qrk{ii}.b;
        Mmat=diag(thetabold(ii,2:end-1).^(-2));
        MdistAll=Mdiagonal_distance(TrainX,TrainX,Mmat);
        phi_k_xnAll=thetabold(ii,1)^2*exp(-(MdistAll.^2)/2); phi_k_xnAll=phi_k_xnAll';
        ActSet{ii}=[];
        IndexCand=ActCand(ii,:);
        for jj=1:M
            Rdet=ones(ActCandSize-jj+1,1);
            for kk=1:(ActCandSize-jj+1)
                indexUsed=[ActSet{ii}; IndexCand(kk)];
                rk3=rk1*(phi_k_xnAll(indexUsed,:).*repmat(qzi(ii,:),jj,1))*phi_k_xnAll(indexUsed,:)';
                Rdet(kk)=det( phi_k_xnAll(indexUsed,indexUsed)+thetabold(ii,end)^2*eye(jj)+rk3 ); %determinant
            end
            [Tempy,TempI]=sort(Rdet);
            ActSet{ii}=[ActSet{ii}; IndexCand(TempI(end))];
            IndexCand(TempI(end))=[];
        end
    end
    toc;
    fprintf('Predict on test data (outerstep=%d)', outerstep );
    Testn=length(TestY);
    nuk=qnuk(:,1)./sum(qnuk,2); nuk=[nuk; 1];
    nuproduct=cumprod(1-nuk);
    betak=[nuk(1); nuk(2:end).*nuproduct(1:end-1)];
    InputPdf=ones(Testn,T);
    Outputy=ones(T,Testn);
    Outputy2=ones(T,Testn);
    for ii=1:T
        tempM =repairPD(qRk{ii}.nu*qRk{ii}.W);
        CholFa=chol(tempM);
        tempSigma=CholFa\(CholFa'\eye(d));
        tempSigma  = (tempSigma+tempSigma')/2;
        InputPdf(:,ii)=mvnpdf(TestX',qmuk{ii}.mu',tempSigma+eye(d)*eps);
        
        Mmat=diag(thetabold(ii,2:end-1).^(-2));
        Mdist=Mdiagonal_distance(TestX,TrainX(:,ActSetOld{ii}),Mmat);
        phi_k_xn=thetabold(ii,1)^2*exp(-(Mdist.^2)/2); phi_k_xn=phi_k_xn';
        %     Approximating integral of sigmoid(f)N(f|mu, sigma);
        %     In section 4 (p.220) of PRML.
        kpa = (1+pi/8*(qrk{ii}.a/qrk{ii}.b)^-1)^(-1/2);
        Outputy2(ii,:)=qwk{ii}.mu'*phi_k_xn;
        Outputy(ii,:)=1./(1+exp(-qwk{ii}.mu'*phi_k_xn*kpa));
    end
    InputPdf=InputPdf';
    tempP=InputPdf.*repmat(betak,1,Testn);
    ExpertP=tempP./(repmat(sum(tempP,1),T,1));
    TestEst(outerstep,:)=sum(ExpertP.*Outputy,1);%decisions
    TestEst2(outerstep,:)=sum(ExpertP.*Outputy2,1);
end
ActSet=ActSetOld;
%================================================predict on test data
disp(['Predict on test data...']);
Testn=length(TestY);
nuk=qnuk(:,1)./sum(qnuk,2); nuk=[nuk; 1];
nuproduct=cumprod(1-nuk);
betak=[nuk(1); nuk(2:end).*nuproduct(1:end-1)];
InputPdf=ones(Testn,T);
Outputy=ones(T,Testn);
Outputy2=ones(T,Testn);
for ii=1:T
    tempM=repairPD(qRk{ii}.nu*qRk{ii}.W);
    CholFa=chol(tempM);
    tempSigma=CholFa\(CholFa'\eye(d));
    tempSigma  = (tempSigma+tempSigma')/2;
    InputPdf(:,ii)=mvnpdf(TestX',qmuk{ii}.mu',tempSigma+eye(d)*eps);
    
    Mmat=diag(thetabold(ii,2:end-1).^(-2));
    Mdist=Mdiagonal_distance(TestX,TrainX(:,ActSet{ii}),Mmat);
    phi_k_xn=thetabold(ii,1)^2*exp(-(Mdist.^2)/2); phi_k_xn=phi_k_xn';
    %     Approximating integral of sigmoid(f)N(f|mu, sigma);
    %     In section 4 (p.220) of PRML.
    kpa = (1+pi/8*(qrk{ii}.a/qrk{ii}.b)^-1)^(-1/2);
    Outputy2(ii,:)=qwk{ii}.mu'*phi_k_xn;
    Outputy(ii,:)=1./(1+exp(-qwk{ii}.mu'*phi_k_xn*kpa));
end
InputPdf=InputPdf';
tempP=InputPdf.*repmat(betak,1,Testn);
ExpertP=tempP./(repmat(sum(tempP,1),T,1));
TestEst_final=sum(ExpertP.*Outputy,1);%decisions
TestEst2_final=sum(ExpertP.*Outputy2,1);

disp('Okay');
