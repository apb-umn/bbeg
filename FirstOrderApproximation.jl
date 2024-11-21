using Parameters,BasisMatrices,SparseArrays,SuiteSparse,TensorOperations
BLAS.set_num_threads(1)
include("utilities.jl")
include("ZerothOrderApproximation.jl")

#Some Helper functions
import Base./,Base.*,Base.\
"""
    /(A::Array{Float64,3},B::SparseMatrixCSC{Float64,Int64})

Apply the inverse to the last dimension of a 3 dimensional array
"""
function /(A::Array{Float64,3},B::SuiteSparse.UMFPACK.UmfpackLU{Float64, Int64})
    ret = similar(A)
    n = size(ret,1)
    for i in 1:n
        ret[i,:,:] .= (B'\view(A,i,:,:)')' 
        #ret[i,:,:] .= A[i,:,:]/B
    end
    return ret
end


"""
    /(A::Array{Float64,3},B::SparseMatrixCSC{Float64,Int64})

Apply the inverse to the last dimension of a 3 dimensional array
"""
function \(A::SuiteSparse.UMFPACK.UmfpackLU{Float64, Int64},B::Array{Float64,3})
    n = size(A,2)
    sizeB = size(B)
    return reshape(A\reshape(B,n,:),sizeB)
end



#function *(A::Matrix{Float64},B::Array{Float64,3})
#    return @tensor C[i,k,l] := A[i,j]*B[j,k,l] 
#end

function *(A::Array{Float64,3},B::SparseMatrixCSC{Float64, Int64})
    k,m,n = size(A)
    return reshape(reshape(A,:,n)*B ,k,m,:) 
end

#function *(A::Matrix{Float64},B::Array{Float64,3})
#    return @tensor C[i,k,l] := A[i,j]*B[j,k,l] 
#end

function *(A::SparseMatrixCSC{T, Int64},B::Array{Float64,3}) where {T<:Real}
    k,m,n = size(B)
    return reshape(A*reshape(B,k,:),:,m,n) 
end

function *(A::Adjoint{T,SparseMatrixCSC{T, Int64}},B::Array{Float64,3}) where {T<:Real}
    k,m,n = size(B)
    return reshape(A*reshape(B,k,:),:,m,n) 
end


"""
FirstOrderApproximation{Model}

Holds all the objects necessary for a first order approximation of a 
given Model.  Will assume that derivatives (e.g. F_x, F_X etc.) exists.
"""
@with_kw mutable struct FirstOrderApproximation
    #M::Model #holds objects that we care about like H
    ZO::ZerothOrderApproximation
    T::Int #Length of IRF

    #Derivative direction
    Δ_0::Vector{Float64} = zeros(1) #Distribution direction (will assume 0 for now)
    X_0::Vector{Float64} = zeros(1)
    Θ_0::Vector{Float64} = zeros(1) #Θ direction

    #x̄_a
    x̄_a::Matrix{Float64} = zeros(0,0)
    
    #Terms for Lemma 2
    f::Vector{Matrix{Float64}} = Vector{Matrix{Float64}}(undef,1) 
    x::Vector{Array{Float64,3}} = Vector{Array{Float64,3}}(undef,1)
    
    #Terms for Lemma 4
    a::Array{Float64,3} = zeros(1,1,1) #a terms from paper
    L::SparseMatrixCSC{Float64,Int64} = spzeros(1,1) #A operator
    M::SparseMatrixCSC{Float64, Int64} = spzeros(1,1) #M operator

    
    #Terms for Corollary 2
    I::Matrix{Float64} = zeros(1,1)   #I operator
    IL::Array{Float64, 3} = zeros(1,1,1)
    ILM::Array{Float64, 3} = zeros(1,1,1)
    E::Array{Float64,3} = zeros(1,1,1) #expectations operators
    J::Array{Float64,4} = zeros(1,1,1,1)

    #Terms for Proposition 1
    BB::SparseMatrixCSC{Float64, Int64} = spzeros(1,1)
    luBB::SparseArrays.UMFPACK.UmfpackLU{Float64, Int64} = lu(sprand(1,1,1.))

    #Outputs
    Ω̂t::Matrix{Float64} =  zeros(1,1)
    x̂t::Vector{Matrix{Float64}} =  Vector{Matrix{Float64}}(undef,1)
    X̂t::Matrix{Float64} = zeros(1,1)


    Ω̂_Θt::Vector{Matrix{Float64}} =  Vector{Matrix{Float64}}(undef,1)
    x̂_Θt::Vector{Vector{Matrix{Float64}}} =  Vector{Vector{Matrix{Float64}}}(undef,1)
    X̂_Θt::Vector{Matrix{Float64}} = Vector{Matrix{Float64}}(undef,1)

    #Portfolio objects
    r::Matrix{Float64} = zeros(1,1)
    vσσ::Vector{Float64} = zeros(1)
    vσσ_raw::Vector{Float64} = zeros(1)
    v::Array{Float64,3} = zeros(1,1,1)
    v_raw::Array{Float64,3} = zeros(1,1,1)
    traders::Vector{Float64} = zeros(1)
    V::Array{Float64,2} = zeros(1,1)
    Vσσ::Float64 = 0.
    N::Array{Float64, 3} = zeros(1,1,1)
    Nv::Array{Float64,4} = zeros(1,1,1,1)
    Nvσσ::Matrix{Float64} = zeros(1,1)
    BBbase::SparseMatrixCSC{Float64, Int64} = spzeros(1,1)
    BBhat::SparseMatrixCSC{Float64, Int64} = spzeros(1,1)
    AAbase::Vector{Float64} = [;]
    AAhat::Matrix{Float64} = zeros(1,1)
    Rx_σσ::Vector{Float64} = zeros(1)
    k̄::Vector{Float64} = zeros(1)
end



"""
    FirstOrderApproximation(ZO::ZerothOrderApproximation, T)

Constructs a first-order approximation object based on a given zeroth-order approximation object.

# Arguments
- `ZO::ZerothOrderApproximation`: The zeroth-order approximation object.
- `T`: The number of time periods for truncation

# Returns
- `approx`: The first-order approximation object.
"""
function FirstOrderApproximation(ZO::ZerothOrderApproximation, T)
    @unpack n = ZO
    approx = FirstOrderApproximation(ZO = ZO, T = T)
    approx.f = Vector{Matrix{Float64}}(undef, n.sp)
    approx.x = Vector{Array{Float64, 3}}(undef, T)
    approx.J = zeros(ZO.n.x, T, ZO.n.Q, T)
    return approx
end


"""
    compute_f_matrices!(FO::FirstOrderApproximation)

Compute the f matrices for the FirstOrderApproximation object `FO`.
These are the building blocks of the x matrices.

# Arguments
- `FO::FirstOrderApproximation`: The FirstOrderApproximation object for which to compute the f matrices.

"""
function compute_f_matrices!(FO::FirstOrderApproximation)
    @unpack ZO,f = FO
    @unpack x̄,Φ̃ᵉₐ,p, dF,n,Φ̃ = ZO
    Ex̄′_a = x̄*Φ̃ᵉₐ
    x̄_a = similar(x̄)
    for j in 1:n.sp
        @views f[j] = -inv(dF.x[j] + dF.x′[j]*Ex̄′_a[:,j]*p)
        @views x̄_a[:,j] .= f[j]*dF.a[j]
    end
    FO.x̄_a = x̄_a/Φ̃
end


"""
    compute_Lemma3!(FO)

Computes the terms from Lemma 3, x_s = dx_t/dX_t+s
"""
function compute_Lemma3!(FO::FirstOrderApproximation)
    #compute_f_matrices!(FO) #maybe uncomment and then remove from compute_theta_derivative
    @unpack ZO,f,x,T = FO
    @unpack x̄,Φ̃ᵉ,Φ̃,p,n, dF = ZO
    #N = length(ẑ)
    luΦ̃ = lu(Φ̃) #precompute inverse of basis matrix
    xtemp = zeros(n.x,n.Q,n.sp) #one nx x nQ matrix for each gridpoint
    cFx′ = Vector{Matrix{Float64}}(undef,n.sp)
    
    for i in 1:n.sp
        cFx′[i] = f[i]*dF.x′[i]
        xtemp[:,:,i] .= f[i]*dF.X[i]
    end
    x[1] = xtemp/luΦ̃

    for s in 2:T
        Ex = x[s-1]*Φ̃ᵉ
        for i in 1:n.sp
            @views xtemp[:,:,i] .= cFx′[i]*Ex[:,:,i]
        end
        x[s] = xtemp/luΦ̃
    end
end


"""
    compute_Lemma4!(FO)

Computes the terms from Lemma 4, Operators L and terms a_s = M p x_s 
"""
function compute_Lemma4!(FO)
    @unpack ZO,x,T = FO
    @unpack Φ,Φₐ,p,ω̄,Λ,x̄,n = ZO
    
    #start with computing A 
    ā_a = ((p*x̄)*Φₐ)[:]  #1xIz array should work with broadcasting
    FO.L = L = deepcopy(Λ)
    for j in eachindex(ā_a)
        for index in nzrange(L,j)
            @inbounds L.nzval[index] *= ā_a[j]
        end
    end

    #Next compute a objects
    #Iz = length(ω̄)
    FO.M = Λ*(Φ'.*ω̄)
    
    FO.a = a = zeros(n.sp,n.Q,T)
    for s in 1:T
        a[:,:,s] .= (p*x[s])[1,:,:]'
    end
end

"""
    compute_Corollary2!(FO)

Constructs J object from Corollary 1 
"""
function compute_Corollary2!(FO)
    @unpack ZO,T,L,M = FO
    @unpack Φₐ,x̄,ω̄ ,n,Φ= ZO
    Lt = sparse(L')
    Mt = sparse(M')
    #Λt = sparse(ZO.Λ')
    #MΦ̃ = Φ.*ω̄'
    #Iz = n.z̄
    #compute expectations vector
    FO.IL = IL  = zeros(n.Ω,n.x,T-1)
    FO.I = x̄*Φₐ
    IL[:,:,1] = (FO.I)' 
    for t in 2:T-1
        @views IL[:,:,t] = Lt*IL[:,:,t-1]
    end
    FO.ILM = permutedims(Mt*IL,[2,3,1])#MΦ̃*(Λt*IL)
end 


function compute_Proposition1!(FO)
    #compute_Corollary2!(FO)
    @unpack ZO,x,T,J,a,ILM = FO
    @unpack Φ,p,ω̄ ,n= ZO

    #Iz = length(ω̄)
    IA = ILM*a#reshape(reshape(ILM,n.ẑ,:)'*reshape(z,n.ẑ,:),n.x,T,n.Q,T)

    IntΦ̃ = Φ * ω̄ #operator to integrate splines over ergodic
    for s in 1:T
        @views J[:,1,:,s] .= x[s]*IntΦ̃
    end

    #initialize l = 0
    for t in 2:T
        @views J[:,t,:,1] .= IA[:,t-1,:,1]
    end
    for s in 2:T
        for t in 2:T
            @views J[:,t,:,s] .= J[:,t-1,:,s-1] .+ IA[:,t-1,:,s]  
        end
    end
end

"""
    compute_BB!(FO::FirstOrderApproximation)

Computes the BB matrix
"""
function compute_BB!(FO::FirstOrderApproximation)
    @unpack ZO,T,J = FO
    @unpack dG,P,Q,n = ZO
    ITT = sparse(I,T,T)
    ITT_ = spdiagm(-1=>ones(T-1))
    ITTᵉ = spdiagm(1=>ones(T-1))
    #construct BB matrix
    if !ZO.portfolio
        FO.BB = kron(ITT,dG.x)*reshape(J,n.x*T,:)*kron(ITT,Q) .+ kron(ITT,dG.X) .+ kron(ITTᵉ,dG.Xᵉ) .+ kron(ITT_,dG.X_*P);
    else
        nRx = size(ZO.R,1)
        Rcon = [kron(zeros(T),ZO.R) kron(ITT[:,2:end],ZO.R)]
        Rcon[(T-1)*nRx+1:T*nRx,(T-1)*n.X+1:T*n.X] += ZO.T
        FO.BB = [kron(ITT,dG.x)*reshape(J,n.x*T,:)*kron(ITT,Q) .+ kron(ITT,dG.X) .+ kron(ITTᵉ,dG.Xᵉ) .+ kron(ITT_,dG.X_*P);
                 Rcon]
    end
    FO.luBB = lu(FO.BB)
end



"""
    solve_Xt!(FO::FirstOrderApproximation)

Solves for the path Xt.
"""
function solve_Xt!(FO::FirstOrderApproximation)
    @unpack ZO,T,Θ_0,Δ_0,X_0,luBB,L = FO
    @unpack x̄,Φ,dG,n,Λ,ρ_Θ = ZO

    AA = zeros(n.X,T)
    for t in 1:T
        @views AA[:,t] .+= dG.Θ*ρ_Θ^(t-1)*Θ_0
    end
    

    AA[:,1] .+= dG.X_*X_0

    Xt = -(luBB\AA[:])
    FO.X̂t = reshape(Xt,n.X,T)
end


"""
    compute_Lemma3PFand7!(FO::FirstOrderApproximation)

Compute Lemma 3PF and 7 for the given FirstOrderApproximation object.

# Arguments
- `FO::FirstOrderApproximation`: The FirstOrderApproximation object to compute Lemma 3PF and 7 for.

"""
function compute_Lemma3PFand7!(FO::FirstOrderApproximation)
    @unpack ZO,f,x,T,traders = FO
    @unpack x̄,Φ̃ᵉ,Φ̃,p,n,dF,s = ZO

    luΦ = lu(Φ̃) #precompute inverse of basis matrix
    rtemp = zeros(n.x,n.sp) #one nx x nQ matrix for each gridpoint
    for i in 1:n.sp
        rtemp[:,i] .= (f[i]*dF.k[i])[:]
    end
    r = FO.r = (luΦ'\rtemp')'

    Esr = (s*r)*Φ̃ᵉ
    Esx̄ = (s*x̄)*Φ̃ᵉ
    
    FO.vσσ_raw = (-Esx̄./Esr)'[:]
    FO.v_raw = v_raw = zeros(n.sp,n.Q,T)
    for t in 1:T
        Esx = (s*x[t])[1,:,:]*Φ̃ᵉ
        v_raw[:,:,t] = (-Esx./Esr)'
    end
end


"""
    compute_OperatorsPF!(FO::FirstOrderApproximation)

Compute the MPP and N operators for the first-order approximation.

# Arguments
- `FO::FirstOrderApproximation`: The FirstOrderApproximation object.

"""
function compute_OperatorsPF!(FO::FirstOrderApproximation)
    @unpack ZO,T,L,M,IL = FO
    @unpack ω̄,n,Φ,Λ,p= ZO
    pr = ((p*FO.r)*Φ)[:]
    MPP = deepcopy(Λ)
    for j in eachindex(pr)
        for index in nzrange(L,j)
            @inbounds MPP.nzval[index] *= pr[j]
        end
    end
    N0 = reshape((FO.r*Φ*M)',n.sp,n.x,1)

    FO.N = permutedims(cat(N0,sparse(M')*sparse(MPP')*IL,dims=3),[2,3,1]) 
end



function compute_BB_AA_PF!(FO::FirstOrderApproximation)
    @unpack ZO,T,J,M,traders,N = FO
    @unpack dG,Q,n,ρ_Θ,Φ̃,P = ZO
    luΦ′ = lu(Φ̃')
    IM = sum(M,dims=1)
    ITT = sparse(I,T,T)
    ITT_ = diagm(-1=>ones(T-1))
    ITTᵉ = spdiagm(1=>ones(T-1))

    IΘ = sparse(I,n.Θ,n.Θ)
    #Compute Nv and Nvσσ
    FO.v = v = luΦ′\(traders.*FO.v_raw)
    FO.vσσ = vσσ = luΦ′\(traders.*FO.vσσ_raw)
    FO.Nvσσ = Nvσσ = (N*vσσ)[:,:]
    FO.Nv   = Nv   = N*v
    #construct BB matrix
    #Full matrix will be nX*T*nΘ+nR
    nRx = size(ZO.R,1)
    Rcon = [kron(zeros(T),ZO.R) kron(ITT[:,2:end],ZO.R)]
    Rcon[1:nRx,(T-1)*n.X+1:T*n.X] += ZO.T
    BB = [kron(ITT,dG.x)*reshape(J,n.x*T,:)*kron(ITT,Q) .+ kron(ITT,dG.X) .+ kron(ITTᵉ,dG.Xᵉ) .+ kron(ITT_,dG.X_*P);
            Rcon]
    FO.BBbase = kron(IΘ,BB)
    #Now consruct the V objects
    FO.Vσσ = Vσσ = (IM*vσσ)[1]
    V = reshape(IM*v,1,1,n.Q,T)
    FO.V = V[1,1,:,:]
    JPP = Nv .- reshape(Nvσσ,n.x,T,1,1).*V./Vσσ
    FO.BBhat  =[kron(ITT,dG.x)*reshape(JPP,n.x*T,:)*kron(ITT,Q);0*Rcon]
    nG = size(dG.Θ,1)
    nR = size(ZO.R,1)
    AA = zeros(nG,T,n.Θ)
    Θ0 = Matrix{Float64}(I,n.Θ,n.Θ)
    for t in 1:T
        @views AA[:,t,:] = dG.Θ*Θ0
        Θ0 = ρ_Θ*Θ0
    end
    FO.AAbase = vcat(reshape(AA,:,n.Θ),zeros(nR*T,n.Θ))[:]
    FO.AAhat = vcat((dG.x*Nvσσ./Vσσ)[:],zeros(nR*T))*reshape(ZO.K*ZO.X̄,1,:)
end

"""
    find_X_Θ_residual!(FO::FirstOrderApproximationPort,R_Θ)

Computes X_Θ  given guess of first order excess returns
"""
function find_X_Θ_residual!(FO::FirstOrderApproximation,Rx_0)
    @unpack ZO,T,BBbase,BBhat,AAbase,AAhat =FO
    @unpack Σ_Θ,n = ZO

    #componenents of exposure to shocks and price of risk
    Sigmainv = inv(Rx_0*Σ_Θ*Rx_0')
    Sigma1 = Σ_Θ * Rx_0' * Sigmainv * Rx_0

    #Now construct BB
    BBPP = BBbase .+ kron(Sigma1',BBhat)
    AAPP = AAbase .+ (AAhat*Rx_0)[:]

    X̂ = -BBPP\AAPP
    FO.X̄_Θ = reshape(X̂,n.X,T,n.Θ)
    return Rx_0 .- ZO.R*FO.X̄_Θ[:,1,:]
end

"""
    find_X_Θ_residual!(FO::FirstOrderApproximationPort,R_Θ)

Computes X_Θ  given guess of first order excess returns
"""
function find_X_Θ!(FO::FirstOrderApproximation)
    @unpack ZO,T,vσσ_raw,v_raw = FO
    @unpack Σ_Θ,n = ZO
    fres(Rx_0) = find_X_Θ_residual!(FO,Rx_0)
    difftraders = 1
    while difftraders > 0
        compute_BB_AA_PF!(FO)
        Rx_0  = nlsolve(fres,[1.]).zero
        QX̂ = ZO.Q*FO.X̄_Θ
        FO.Rx_σσ= Rx_σσ = ZO.K*ZO.X̄./FO.Vσσ .- dot(FO.V[:]./FO.Vσσ,QX̂[:]./Rx_0)

        k̄ = vσσ_raw[:].*Rx_σσ .+ reshape(v_raw,n.ẑ,:)*QX̂[:]./Rx_0
        tradersnew = (ZO.Φ'*k̄).>=0 
        difftraders = sum(abs.(FO.traders.-tradersnew))
        FO.traders = tradersnew
    end
end


"""
    find_X_Θ_residual!(FO::FirstOrderApproximationPort,R_Θ)

Computes X_Θ  given guess of first order excess returns
"""
function find_X_Θ_linear!(FO::FirstOrderApproximation)
    @unpack ZO,T,vσσ_raw,v_raw = FO
    @unpack Σ_Θ,n = ZO
    difftraders = 1

    #some setup
    nRx = size(ZO.R,1)
    R0 = spzeros(nRx,size(FO.BBbase,2))
    R0[1:nRx,1:n.X] = ZO.R
    while difftraders > 0
        compute_BB_AA_PF!(FO)
        @unpack BBbase,BBhat,AAbase,AAhat = FO
        BBPP = BBbase .+ BBhat .+ AAhat*R0
        X̂_Θ = -BBPP\AAbase
        X̂_Θ = reshape(X̂_Θ,n.X,T,n.Θ)
        Rx_0 = ZO.R*X̂_Θ[:,1,:]
        QX̂ = ZO.Q*X̂_Θ
        FO.Rx_σσ= Rx_σσ = ZO.K*ZO.X̄./FO.Vσσ .- dot(FO.V[:]./FO.Vσσ,QX̂[:]./Rx_0)

        k̄ = vσσ_raw[:].*Rx_σσ .+ reshape(v_raw,n.sp,:)*QX̂[:]./Rx_0
        tradersnew = k̄.>=0 
        difftraders = sum(abs.(FO.traders.-tradersnew))
        FO.traders .= tradersnew
        FO.k̄ = ZO.Φ̃'\(FO.vσσ[:].*Rx_σσ .+ reshape(FO.v,n.sp,:)*QX̂[:]./Rx_0)[:]

        if difftraders == 0
            FO.X̂_Θt = Vector{Matrix{Float64}}(undef,n.Θ)
            for iΘ in 1:n.Θ
                FO.X̂_Θt[iΘ] = X̂_Θ[:,:,iΘ]
            end
        end
    end
end


function compute_x̂t_Ω̂t!(FO::FirstOrderApproximation)
    @unpack ZO,T,x,X̂t,L,M = FO
    @unpack Q,Φ,ω̄,p,n,Λ = ZO
    #Fill objects
    #N = length(ZO.â)
    FO.x̂t = [zeros(n.x,n.sp) for t in 1:T]
    QX̂t= Q*X̂t

    for s in 1:T
        x_s = permutedims(x[s],[1,3,2])
        for t in 1:T-(s-1)
            @views FO.x̂t[t] .+= x_s * QX̂t[:,t+s-1]
        end
    end

    #Next use ā_Z to construct Ω̂t
    Ω̂t = FO.Ω̂t = zeros(n.Ω,T)
    for t in 2:T
        ât_t = (p*FO.x̂t[t-1])[:]
        @views Ω̂t[:,t] = L*Ω̂t[:,t-1] .+ M*ât_t  # this is different from the paper because Δ is change in histogram and not CDF
    end
end



"""
    compute_Θ_derivatives!(FO)

Computes the derivatives in each Θ direction
"""
function compute_Θ_derivatives!(FO::FirstOrderApproximation)
    @unpack ZO = FO
    @unpack n = ZO

    FO.X̂_Θt =  Vector{Matrix{Float64}}(undef,n.Θ)
    FO.X_0 = zeros(n.A)
    FO.Δ_0 = zeros(n.Ω)
    compute_f_matrices!(FO)
    compute_Lemma3!(FO)
    compute_Lemma4!(FO)
    compute_Corollary2!(FO)
    compute_Proposition1!(FO)

    if ZO.portfolio
        FO.traders = ones(n.sp) #start with guess that everyone is trading the asset
        compute_Lemma3PFand7!(FO)
        compute_OperatorsPF!(FO)
        compute_BB_AA_PF!(FO)
        if n.Θ > 1
            find_X_Θ!(FO)
        else
            find_X_Θ_linear!(FO)
        end
    else
        compute_BB!(FO)
        for i in 1:n.Θ
            FO.Θ_0 = I[1:n.Θ,i] #ith basis vector
            solve_Xt!(FO)
            FO.X̂_Θt[i] = FO.X̂t
        end
    end
end

function compute_x_Θ_derivatives!(FO)
    @unpack ZO = FO
    @unpack n = ZO

    FO.Ω̂_Θt =  Vector{Matrix{Float64}}(undef,n.Θ)
    Vector{Vector{Matrix{Float64}}}(undef,1)
    FO.x̂_Θt =  Vector{Vector{Matrix{Float64}}}(undef,n.Θ)
    FO.X_0 = zeros(n.A)
    for i in 1:n.Θ
        FO.Θ_0 = I[1:n.Θ,i] #ith basis vector
        compute_x̂t_Ω̂t!(FO)
        FO.Ω̂_Θt[i] = FO.Ω̂t
        FO.x̂_Θt[i] = FO.x̂t
    end
end

## verfied that the code runs until here works before this

