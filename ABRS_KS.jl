using Parameters,SparseArrays,LinearAlgebra,BasisMatrices,Dierckx,ForwardDiff

function doABRS(AM,X̄,x̄,T,Σ_Θ,ρ_Θ,δx=0.0001)
    X0 = [AM.Θ̄, AM.K̄] ; 
    JM1_KΘ = ForwardDiff.jacobian(x -> FirmFiscalBlock_ΘK(AM,x), X0)

    # step 2 is common for all the Jacobian in the Het Agent block
    Et1_RWT = computestep2(AM,T)
    
    ## JtR 
    Dt1_R = computestep1(AM,X̄,x̄,"R",T,δx)
    Ft1_R = computestep3(AM,T,Et1_RWT,Dt1_R)
    Jt1_R = computestep4(AM,T,Ft1_R)
    
    #.*J1_KΘ[:,1]
    ## JtW
    Dt1_W = computestep1(AM,X̄,x̄,"W",T,δx)
    #Et1_W = computestep2(AM,T)
    Ft1_W = computestep3(AM,T,Et1_RWT,Dt1_W)
    Jt1_W = computestep4(AM,T,Ft1_W)
    

    dZt1 = [sqrt(Σ_Θ[1])*ρ_Θ[1]^(t-1) for t in 1:T]
    dKt,dRt,dWt, J_K, J_Θ  = computestep5(Jt1_R[2:T+1,:], Jt1_W[2:T+1,:],JM1_KΘ, dZt1)
    return dKt,dRt,dWt
end;

function FirmFiscalBlock_ΘK(AM,X)
    @unpack α,δ,N̄ = AM
    Θ = X[1]
    K_ = X[2]

    R = 1 + α*Θ*K_^(α-1)*N̄^(1-α) -δ
    W = (1-α)*Θ*K_^α*N̄^(-α) 
    return [R,W]
end


function computestep2(AM,T)
    Et = zeros(T+1,AM.Nθ*AM.Ia)
    Et[1,:] = AM.āθ̄[:,1]'
    for s = 2:T+1
        Et[s,:] =Et[s-1,:]' *AM.Λ
        #[dot(ωt[:,t+1],b̄grid) for t in 1:T]
    end 
    return Et
end



function computestep1(AM,X̄,x̄,var,T,δx)

    Dt = zeros(AM.Nθ*AM.Ia,T+1)
    #Yt = zeros(AM.Nϵ*AM.Ib,T+1)

    at = copy(AM.ω̄)
    ā′ = copy(AM.ω̄)
    Xt0 = X̄[1:2].*ones(2,T+1)
    Rt = Xt0[1,:]
    Wt = Xt0[2,:]
    #_,cfs0 = compute_policy_path_alt(AM,Rt,Wt,Tt) #policy without shock
    if var == "R"
        Rt[T+1] += δx
    elseif var == "W"
        Wt[T+1] += δx
    end 
    _,cfs,c̄f′ = compute_policy_path(AM,x̄,Rt,Wt)
    iteratedistribution_forward_a!(AM,ā′,AM.R̄,AM.W̄,c̄f′)
    
    for s = 1:T+1
        #backward iterate to get policies
        #forward iterate to get path of distribution
#        ωt1 = compute_distribution_path_jacobian(AM,Rt,Wt,Tt,Vcoefs)
        #ωt1, Qt,bprime = iteratedistribution_forward_policy(AM,AM.ω̄,Rt[s],Wt[s],Tt[s],Vcoefs[:,s+1])
        #iteratedistribution_forward_alt!(AM,AM.ω̄,ωt0,Rt[s],Wt[s],Tt[s],cfs0[s])
        iteratedistribution_forward_a!(AM,at,Rt[s],Wt[s],cfs[s])
        #bt1 =reshape(bprime, AM.Nϵ*AM.Ib, 1) ; 
        #Yt[:,s] = (bt1 .- bt0) ./ δx  
        Dt[:,T+2-s] .= AM.Λ_z*((at .- ā′).*AM.ω̄./ δx)
    end 
    return Dt   #, Yt
end 


function compute_policy_path(AM,x̄,Rt,Wt)
    #first set up matrices
    @unpack Φ,Λ_z,Nθ,a′grid,σ,R̄,W̄ = AM
    luΦ = lu(Φ)
    #S = length(AM.ϵ)
    #now compute coefficients
    T = length(Rt)
    #V̄coefs = vcat([AM.V̄[s].coefs for s in 1:Nϵ]...)
    λ̄coef = x̄[2,:] #grab the λ ceoficients
    dλcoefs = zeros(length(λ̄coef),T+1)
    cfs = Vector{Vector{Spline1D}}(undef,T)
    
    λ̄coef′,c̄f′ = backwardsiterate(AM,λ̄coef,R̄,W̄,luΦ)
    
    for t in reverse(1:T)
        λcoef,cfs[t] = backwardsiterate(AM,λ̄coef .+ dλcoefs[:,t+1],Rt[t],Wt[t],luΦ)  
        dλcoefs[:,t] .= λcoef .- λ̄coef′
    end

    return λ̄coef.+dλcoefs,cfs,c̄f′
end


function backwardsiterate(AM::AiyagariModel,λcoefs′,R,W,luΦ)
    @unpack σ,β,πθ,Na,Nθ = AM

    cf = computeconsumptionpolicy(AM,λcoefs′,R,W)

    agrid = nodes(AM.abasis)[1]

    c = zeros(Na*Nθ) 
    for s in 1:Nθ
        c[(s-1)*Na+1:s*Na] = cf[s](agrid) #compute consumption at gridpoints
    end

    return luΦ\(R.*c.^(-σ)),cf
end


function computeconsumptionpolicy(AM,λcoefs′,R,W)
    @unpack σ,β,θ,a′grid,a̲, Nθ,EΦeg = AM

    Eλ′ = reshape(EΦeg*λcoefs′,:,Nθ)
    c = (β.*Eλ′).^(-1/σ) #consumption today
    a = (a′grid .+ c .- W.*exp.(θ'))./R  #Implied assets today

    cf = Vector{Spline1D}(undef,Nθ)#Vector{Spline1D}(undef,Nϵ)#implied policy rules for each productivity
    for s in 1:Nθ
        #with some productivities the borrowing constraint does not bind
        if a[1,s] > a̲ #borrowing constraint binds
            #add extra points on the borrowing constraint for interpolation
            â = [a̲;a[:,s]]
            p = sortperm(â)
            ĉ = [R*a̲-a̲ + W*exp(θ[s]);c[:,s]]
            cf[s] = Spline1D(â[p],ĉ[p],k=1)#
        else
            p = sortperm(a[:,s])
            cf[s] = Spline1D(a[p,s],c[p,s],k=1)#
        end
    end

    return cf
end


function iteratedistribution_forward_a!(AM,a′,R,W,cf)
    @unpack θ,πθ,Ia,āθ̄, Nθ = AM
    ā = āθ̄[1:Ia,1] #grids are all the same for all shocks
    c = hcat([cf[s](ā) for s in 1:Nθ]...) #consumption policy
    a′ .= max.(min.(T .+ R.*ā .+ W.*exp.(θ') .- c))[:] #max.(min.create a Ib×Nϵ grid for the policy rules
end


function prepare_grids!(AM)

    bbasis = AM.bbasis
    AM.b′grid = bgrid = nodes(bbasis)[1]
    #Precompute EΦ at basis
    #AM.EΦ′ = kron(AM.πϵ,BasisMatrix(bbasis,Direct(),AM.b′grid,[1]).vals[1])
    #AM.EΦ = kron(AM.πϵ,BasisMatrix(bbasis,Direct(),AM.b′grid).vals[1])
    #Precompute Phi
    #AM.Φ = kron(Matrix(I,AM.Nϵ,AM.Nϵ),BasisMatrix(bbasis,Direct()).vals[1])
    #AM.Φ′ = kron(Matrix(I,AM.Nϵ,AM.Nϵ),BasisMatrix(bbasis,Direct(),AM.b′grid,[1]).vals[1])
    return nothing
end 

function computestep3(AM,T,Et,Dt)
    Ft = zeros(T+1,T)
    #Ft[1,:] =AM.ω̄' * Yt ; 
    Ft[2:T+1,:] = Et[1:T, :] * Dt[:,1:T]

    return Ft
end 


function computestep4(AM,T,Ft)
    Jt = zeros(T+1,T)

    for t = 2:T+1
        Jt[t,1] = Ft[t,1]
        for s=2:T
            Jt[t,s] = Jt[t-1,s-1] .+ Ft[t,s]
        end 
    end 

    return Jt
end

function computestep5(JtR, JtW,JM_KΘ, dZ)

    #JtK is the Jacobian of Capital (supply) w.r.t change in capital demand (which change the prices R,W,T)
    #JtZ is the Jacobian of Capital (supply) w.r.t change in TFP (which change the prices R,W,T)
    #faster than do three jacobian w.r.t R,W,T respectively
    T = size(JtR)[1]
        #step 5.1
    #dU = - H_U^{-1} H_Z dZ
        #step 5.2
    #dX = M_U dU + M_Z dZ

    Lag_R_K = diagm(-1=> JM_KΘ[1,2]*ones(T-1))
    Lag_W_K = diagm(-1=> JM_KΘ[2,2]*ones(T-1))
    Lag_K_K = diagm(0 => ones(T))

    J_K = JtR * Lag_R_K  .+ JtW * Lag_W_K .- Lag_K_K  

    J_Θ = JtR * JM_KΘ[1,1]  .+ JtW * JM_KΘ[2,1]
#   dK = - JtK[2:T,2:T] \ (JtZ[2:T,2:T] * dZ)

    dK = .- J_K \ (J_Θ * dZ)
    dR = Lag_R_K*dK .+ JM_KΘ[1,1].*dZ
    dW = Lag_W_K*dK .+ JM_KΘ[2,1].*dZ
    return dK,dR,dW, J_K, J_Θ
end 



function getPathABRS(path,dX,Xbar)
    Xt=zeros(length(path))
    Xzerotht=Xbar*ones(length(path))
    Xfirstordert=zeros(length(path))

    for t in 1:length(path)
        fo=0
        for s=1:t
            fo += path[s]*dX[t-s+1]
        end


        Xfirstordert[t]= fo
    end
    Xt=DataFrame()
    Xt.zeroth=Xzerotht
    Xt.firstorder=Xfirstordert
    Xt.path=path
    Xt.X=Xzerotht+Xfirstordert

    return Xt
end

## Code to Check Accuracy


function check_accuracy(AM,X̂t)
    @unpack α,δ,N̄,K2Y,Θ̄,R̄,W̄ = AM 
    T = size(X̂t,2)
    #compute steady state values
    Y2K = 1/K2Y
    K2N = (Y2K/Θ̄)^(1/(α-1))
    K̄ = N̄*K2N
    Kt = X̂t[5,:] .+ K̄
    Rt = X̂t[1,:] .+ R̄
    Wt = X̂t[2,:] .+ W̄
    ωt = zeros(length(AM.ω̄),T+1)
    ωt[:,1] .= AM.ω̄

    λcoefs,cfs = compute_policy_path_accuracy(AM,Rt,Wt)
    for s = 1:T
        #backward iterate to get policies
        #forward iterate to get path of distribution
#        ωt1 = compute_distribution_path_jacobian(AM,Rt,Wt,Tt,Vcoefs)
        #ωt1, Qt,bprime = iteratedistribution_forward_policy(AM,AM.ω̄,Rt[s],Wt[s],Tt[s],Vcoefs[:,s+1])
        @views iteratedistribution_forward_accuracy!(AM,ωt[:,s],ωt[:,s+1],Rt[s],Wt[s],cfs[s])
    end 
    #infer path of market value of capital
    āgrid = AM.āθ̄[:,1]
    Ktt = [dot(ωt[:,t+1],āgrid) for t in 1:T] 

    return 100 .* (Kt.-Ktt)./Kt
end


function compute_policy_path_accuracy(AM,Rt,Wt)
    #first set up matrices
    @unpack Nθ,Φ,a′grid,σ,R̄,W̄,λ̄coefs = AM
    T̄ = 0. 
    luΦ = lu(Φ)
    #S = length(AM.ϵ)
    #now compute coefficients
    T = length(Rt)
    nλcoefs = length(λ̄coefs)
    λcoefs = zeros(nλcoefs,T+1)
    cfs = Vector{Vector{Spline1D}}(undef,T)#Vector{Vector{Spline1D}}(undef,T)
    λcoefs[:,T+1] = λ̄coefs #grab the λ ceoficients
    
    for t in reverse(1:T)
        λcoefs[:,t],cfs[t] = backwardsiterate(AM,λcoefs[:,t+1],Rt[t],Wt[t],luΦ)  
    end

    return λcoefs,cfs
end

function iteratedistribution_forward_accuracy!(AM,ω,ω′,R,W,cf)
    @unpack θ,πθ,Ia,āθ̄, Nθ = AM
    ā = āθ̄[1:Ia,1] #grids are all the same for all shocks
    c = hcat([cf[s](ā) for s in 1:Nθ]...) #consumption policy
    a′ =  R.*ā .+ W.*exp.(θ') .- c #create a Ib×Nϵ grid for the policy rules
    
    #make sure we don't go beyond bounds.  Shouldn't bind if bmax is correct
    a′ = max.(min.(a′,ā[end]),ā[1])
    ω = reshape(ω,Ia,:)
    ω′ .= 0.
    for s in 1:Nθ
        Qa = BasisMatrix(Basis(SplineParams(ā,0,1)),Direct(),a′[:,s]).vals[1]
        ω′ .+= kron(πθ[s,:],sparse(I,Ia,Ia))*(Qa'*ω[:,s])
    end
    #(ω'*Q)' #, Q, b′
end
