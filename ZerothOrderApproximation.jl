using Parameters,BasisMatrices,SparseArrays,SuiteSparse,TensorOperations


"""
Nums stores sizes of various objects

nx  number of individual policy function
nX  number of aggregate policy functions
nQ  number of aggregate variables that appear in individual problem
nA  number of aggregate predetermined variables 
nz̄  number of points on the for storing the histogram
nẑ  number of splines
nθ  number of points on shock grid
  
"""
@with_kw mutable struct Nums
    x::Int64   = 0
    X::Int64   = 0    
    Q::Int64   = 0
    A::Int64   = 0
    Ω::Int64   = 0
    sp::Int64   = 0
    θ::Int64   = 0 
    Θ::Int64   = 0
    a::Int64   = 0

    #Portfolio
    R::Int64   = 0 
end


 """
Imputs is a stuct that contains the user inputs 
"""
@with_kw mutable struct Inputs

    ## Policy Functions
    #zeroth order policy functions
    xf::Vector{Function}    = [(a,θ)->0] 
    πθ::Matrix{Float64}     = zeros(1,1)
    #xlab::Vector{Symbol}    = [:a,:λ,:v]
    #alab::Symbol            = [:a]
    xlab::Vector{Symbol}    = [:k,:λ,:v]    # notations from the paper
    alab::Vector{Symbol}    = [:k]          # notations from the paper
    
    #basis functions
    aknots::Vector{Float64}  = zeros(1) #knot points
    ka::Int                  = 2        #order of splines

    #gridpoints
    aθ_sp::Matrix{Float64}  = zeros(1,1) #course grid for splines a_sp
    aθ_Ω::Matrix{Float64}  = zeros(1,1) #fine grid for distribution a_Ω z̄
    #kinks
    ℵ::Vector{Int}          = Int[] #\aleph objects

    ## Steady State Aggregates
    #Distribution
    ω̄ ::Vector{Float64}     = zeros(1)
    Λ::SparseMatrixCSC{Float64,Int64}   = spzeros(1,1)
    Λ_z::SparseMatrixCSC{Float64,Int64} = spzeros(1,1) #do we want this???

    #Aggregate variables
    X̄::Vector{Float64}      = zeros(1)
    # total vector of aggregate variables
    Xlab::Vector{Symbol}    = [:R,:W,:T,:ℐ,:C,:Y,:V,:K,:q]
#    Xlab::Vector{}    = [:K,:R,:W]

    # vector of past aggregate variables relevant for HH problem / G function / in matrix P
    Alab::Vector{Symbol}     = [:K,:q]
#    Alab::Vector{Symbol}    = [:K]

    # vector of present aggregate variables relevant for HH problem / F function / in matrix Q
    Qlab::Vector{Symbol}    = [:R,:W,:T]
#    Qlab::Vector{Symbol}    = [:R,:W]

    ##Stochastic Equilibrium
    #Equilibrium definition
    F::Function             = (para,θ,a,x,QX,x′)->zeros(1)
    G::Function             = (para,Ix,A_,X,Xᵉ,Θ)->zeros(1)

    #Shock Process
    Θ̄::Vector{Float64}      = ones(1)
    ρ_Θ::Matrix{Float64}    = ones(1,1)
    Σ_Θ::Matrix{Float64}    = ones(1,1)
    ρ_Υ::Float64            = 0. #stochastic volatility
    σ_Υ::Float64            = 0.

    #Portfolio objects
    portfolio::Bool              = false #flag to construct portfolio objects
    Klab::Vector{Symbol}    = []
    Rlab::Vector{Symbol}    = []
    Tlab::Vector{Symbol}    = []
    slab::Symbol            = :s
 end


"""
DerivativesF stores derivatives of F 
"""


@with_kw mutable struct DerivativesF
    x::Vector{Matrix{Float64}}      =  [zeros(1,1)]
    X::Vector{Matrix{Float64}}      =  [zeros(1,1)]
    x′::Vector{Matrix{Float64}}     =  [zeros(1,1)]
    a::Vector{Vector{Float64}}      =  [zeros(1)]
    aa::Vector{Vector{Float64}}     =  [zeros(1)]
    ax::Vector{Matrix{Float64}}     =  [zeros(1,1)]
    ax′::Vector{Matrix{Float64}}    =  [zeros(1,1)]
    xx::Vector{Array{Float64,3}}    =  [zeros(1,1,1)]
    xX::Vector{Array{Float64,3}}    =  [zeros(1,1,1)]
    xx′::Vector{Array{Float64,3}}   =  [zeros(1,1,1)]
    Xx::Vector{Array{Float64,3}}    =  [zeros(1,1,1)]
    XX::Vector{Array{Float64,3}}    =  [zeros(1,1,1)]
    Xx′::Vector{Array{Float64,3}}   =  [zeros(1,1,1)]
    x′x::Vector{Array{Float64,3}}    =  [zeros(1,1,1)]
    x′X::Vector{Array{Float64,3}}    =  [zeros(1,1,1)]
    x′x′::Vector{Array{Float64,3}}  =  [zeros(1,1,1)]

    #Portfolio objects
    k::Vector{Vector{Float64}}     = [zeros(1)]
end

"""
DerivativesG stores derivatives of G
"""

@with_kw mutable struct DerivativesG

    x::Matrix{Float64}      = zeros(1,1)
    X::Matrix{Float64}      = zeros(1,1) 
    X_::Matrix{Float64}     = zeros(1,1)
    Xᵉ::Matrix{Float64}      = zeros(1,1)
    Θ::Matrix{Float64}      = zeros(1,1)
    xx::Array{Float64,3}    = zeros(1,1,1)
    xX_::Array{Float64,3}   = zeros(1,1,1)
    xX::Array{Float64,3}    = zeros(1,1,1)
    xXᵉ::Array{Float64,3}    = zeros(1,1,1)
    xΘ::Array{Float64,3}    = zeros(1,1,1)
    X_X_::Array{Float64,3}  = zeros(1,1,1)
    X_X::Array{Float64,3}   = zeros(1,1,1)
    X_Xᵉ::Array{Float64,3}    = zeros(1,1,1)
    X_Θ::Array{Float64,3}   = zeros(1,1,1)
    XX::Array{Float64,3}    = zeros(1,1,1)
    XXᵉ::Array{Float64,3}    = zeros(1,1,1)
    XΘ::Array{Float64,3}    = zeros(1,1,1)
    XᵉXᵉ::Array{Float64,3}    = zeros(1,1,1)
    XᵉΘ::Array{Float64,3}    = zeros(1,1,1)
    ΘΘ::Array{Float64,3}    = zeros(1,1,1)
end



"""
The Zeroth order class that contains the objects that we need from the zeroth order 
"""


@with_kw mutable struct ZerothOrderApproximation
    # Nums
    n::Nums=Nums()
    # grids
    aθ_sp::Matrix{Float64}      = zeros(1,1) #course grid for splines 
    aθ_Ω::Matrix{Float64}      = zeros(1,1) #fine grid for distribution
    
    #policy functions
    x̄::Matrix{Float64} =  zeros(1,1) # policy rules

    #aggregates
    X̄::Vector{Float64} = zeros(1) #steady state aggregates

    #Shock Processes
    ρ_Θ::Matrix{Float64} = 0.8*ones(1,1)
    Σ_Θ::Matrix{Float64} = 0.014^2*ones(1,1)
    μ_Θσσ::Vector{Float64} = zeros(1)
    ρΥ::Float64 = 0.75 #Persistance of volatility shock
    σΥ::Float64 = 1.  #Size of volatility shock
    Σ_ΘΥ::Matrix{Float64} = Σ_Θ#time varying risk premium

    #masses for the stationary distribution
    ω̄::Vector{Float64} = ones(1) 
    #basis and transition matricies 
    Φ̃::SparseMatrixCSC{Float64,Int64}   = spzeros(n.sp,n.sp)
    Φ̃ₐ::SparseMatrixCSC{Float64,Int64}  = spzeros(n.sp,n.sp)
    Φ̃ᵉₐ::SparseMatrixCSC{Float64,Int64} = spzeros(n.sp,n.sp)
    Φ̃ᵉ::SparseMatrixCSC{Float64,Int64}  = spzeros(n.sp,n.sp)
    Φ::SparseMatrixCSC{Float64,Int64}  = spzeros(n.sp,n.sp)
    Φₐ::SparseMatrixCSC{Float64,Int64} = spzeros(n.sp,n.sp)
    Λ::SparseMatrixCSC{Float64,Int64}   = spzeros(n.Ω,n.Ω)
    Λ_z::SparseMatrixCSC{Float64,Int64} = spzeros(n.Ω,n.Ω)

    #kinked policy rules
    ℵ::Vector{Int}          = Int[] #ℵ objects
    
    #Objects for first order approximation
    p::Matrix{Float64} = zeros(1)'  #projection matrix
    P::Matrix{Float64} = zeros(1,1) #projection matrix X->A_ 
    Q::Matrix{Float64} = zeros(1,1) #selector matrix for prices relevant for HH problem
    Q′::Matrix{Float64} = zeros(1,1) #selector matrix for future aggregate variables
    
    # F and G 
    dF::DerivativesF = DerivativesF() 
    dG::DerivativesG = DerivativesG()

    #Portfolio Objects
    portfolio::Bool    = false
    s::Matrix{Float64} = [;;]
    R::Matrix{Float64} = [;;]
    K::Matrix{Float64} = [;;]
    T::Matrix{Float64} = [;;]
end


function create_array_with_one(n::Int, position::Int)
    arr = zeros(n)  # Create an array of zeros of length n
    arr[position] = 1  # Set the specified position to 1
    return arr
end


function construct_selector_matrix(n::Int64, indices::Vector)
    m = length(indices)
    sel_matrix = sparse(1:m, indices, 1, m, n)
    return sel_matrix
end




function construct_abasis(zgrid::Vector)::Basis{1, Tuple{SplineParams{Vector{Float64}}}}
    abasis = Basis(SplineParams(zgrid,0,2))
    return abasis
end
    


function construct_x̄s(abasis::Basis{1, Tuple{SplineParams{Vector{Float64}}}},xf::Vector{Function},n::Nums)
    #@unpack iz = inputs
    x̄f = Matrix{Interpoland}(undef,n.x,n.θ)
    x̄=zeros(n.x,n.sp)
    for i in 1:n.x
        x̄f[i,:] .= [Interpoland(abasis,a->xf[i](a,s)) for s in 1:n.θ]
        x̄[i,:]  = hcat([x̄f[i,s].coefs' for s in 1:n.θ]...)
    end
    return x̄
end

function construct_Φ̃s(abasis::Basis{1, Tuple{SplineParams{Vector{Float64}}}},aθ_sp::Matrix{Float64},aθ_Ω::Matrix{Float64},af::Function,πθ::Matrix{Float64},n::Nums)
    a_sp = unique(aθ_sp[:,1])
    θ    = unique(aθ_sp[:,2])
    a_Ω  = unique(aθ_Ω[:,1])

    N = length(a_sp)

    Φ̃ = kron(Matrix(I,n.θ,n.θ),BasisMatrix(abasis,Direct(),a_sp).vals[1])'
    Φ̃ₐ = kron(Matrix(I,n.θ,n.θ),BasisMatrix(abasis,Direct(),a_sp,[1]).vals[1])'
    Φ̃ᵉ = spzeros(N*n.θ,N*n.θ)
    Φ̃ᵉₐ = spzeros(N*n.θ,N*n.θ)
    
    for s in 1:n.θ
        for s′ in 1:n.θ
            #b′ = R̄*bgrid .+ ϵ[s]*W̄ .- cf[s](bgrid) #asset choice
            a′ = af(a_sp,θ[s])
            Φ̃ᵉ[(s-1)*N+1:s*N,(s′-1)*N+1:s′*N] = πθ[s,s′]*BasisMatrix(abasis,Direct(),a′).vals[1]
            Φ̃ᵉₐ[(s-1)*N+1:s*N,(s′-1)*N+1:s′*N] = πθ[s,s′]*BasisMatrix(abasis,Direct(),a′,[1]).vals[1]
        end
    end
    #Recall our First order code assumes these are transposed
    Φ̃ᵉ = Φ̃ᵉ'
    Φ̃ᵉₐ = (Φ̃ᵉₐ)'  

    Φ = kron(Matrix(I,n.θ,n.θ),BasisMatrix(abasis,Direct(),a_Ω).vals[1])' #note transponse again
    Φₐ =kron(Matrix(I,n.θ,n.θ),BasisMatrix(abasis,Direct(),a_Ω,[1]).vals[1])' #note transponse again
    return Φ̃,Φ̃ₐ,Φ̃ᵉ,Φ̃ᵉₐ,Φ,Φₐ
end 



function ZerothOrderApproximation(inputs::Inputs)
    @unpack xf,aknots,aθ_sp,aθ_Ω,ℵ = inputs
    @unpack ω̄,Λ, Λ_z, πθ, Θ̄, X̄ = inputs    
    @unpack xlab, alab, Xlab, Alab, Qlab = inputs
    @unpack ρ_Θ, Σ_Θ = inputs
    ZO =    ZerothOrderApproximation()
    
    n=Nums(θ = size(πθ)[1], Ω = size(aθ_Ω,1), sp = size(aθ_sp,1), x=length(xf), X=length(X̄), Q=length(Qlab),  A=length(Alab), Θ=length(Θ̄), a=1)

    ia = ((xlab .== reshape(alab,1,:))' * (1:n.x))[1]# [(xlab .== reshape(alab,1,:))' * (1:n.x) for i =1:length(alab)][1] #if we only have a single z variable
    iA = (Xlab .== reshape(Alab,1,:))' * (1:n.X) # [(Xlab .== Alab[i])' * (1:n.X) for i =1:length(Alab)]
    iQ = (Xlab .== reshape(Qlab,1,:))' * (1:n.X) #[(Xlab .== Qlab[i])' * (1:n.X) for i =1:length(Qlab)]

    abasis= Basis(SplineParams(aknots,0,inputs.ka))
    Φ̃,Φ̃ₐ,Φ̃ᵉ,Φ̃ᵉₐ,Φ,Φₐ = construct_Φ̃s(abasis,aθ_sp,aθ_Ω,xf[ia],πθ,n)

    #Now compute x̄
    x̂ = zeros(n.x,n.sp)
    for i in 1:n.x
        for j in 1:n.sp
            a,θ = aθ_sp[j,:]
            x̂[i,j] = xf[i](a,θ)
        end
    end
    ZO.n=n
    ZO.aθ_sp = aθ_sp
    ZO.aθ_Ω = aθ_Ω
    ZO.x̄ = x̂/Φ̃
    ZO.X̄ = X̄
    ZO.ω̄ = ω̄ 

    ZO.Φ̃ = Φ̃
    ZO.Φ̃ₐ = Φ̃ₐ
    ZO.Φ̃ᵉₐ = Φ̃ᵉₐ 
    ZO.Φ̃ᵉ = Φ̃ᵉ
    ZO.Φ = Φ
    ZO.Φₐ = Φₐ
    ZO.Λ = Λ
    ZO.Λ_z = Λ_z

    ZO.p = create_array_with_one(n.x,ia)'
    ZO.P = construct_selector_matrix(n.X,iA)
    ZO.Q = construct_selector_matrix(n.X,iQ)

    ZO.ℵ = ℵ

    ZO.ρ_Θ = ρ_Θ
    ZO.Σ_Θ = Σ_Θ

    if inputs.portfolio
        @unpack Klab,Rlab,Tlab,slab = inputs
        ZO.portfolio = true
        is = ((xlab .== slab)' * (1:n.x))[1]
        iR = (Xlab .== reshape(Rlab,1,:))' * (1:n.X) 
        iK = (Xlab .== reshape(Klab,1,:))' * (1:n.X) 
        iT = (Xlab .== reshape(Tlab,1,:))' * (1:n.X)
        ZO.s = create_array_with_one(n.x,is)'
        ZO.R = construct_selector_matrix(n.X,iR)
        ZO.K = construct_selector_matrix(n.X,iK)
        ZO.T = construct_selector_matrix(n.X,iT)
    end
        
    return ZO    
end


function computeDerivativesF!(ZO::ZerothOrderApproximation,inputs::Inputs)
    if ZO.portfolio == true
        computeDerivativesF_portfolio!(ZO,inputs)
    else
        computeDerivativesF_no_portfolio!(ZO,inputs)
    end
end


function computeDerivativesF_no_portfolio!(ZO::ZerothOrderApproximation,inputs::Inputs)
    @unpack n,aθ_sp,Φ̃,X̄,x̄,Φ̃ᵉ,Q = ZO
    @unpack F = inputs
    dF=DerivativesF()
    dF.x= [zeros(n.x,n.x) for _ in 1:n.sp]
    dF.x′=[zeros(n.x,n.x) for _ in 1:n.sp]
    dF.X=[zeros(n.x,n.Q) for _ in 1:n.sp] # check
    dF.a = [zeros(n.x) for _ in 1:n.sp]
    dF.aa = [zeros(n.x) for _ in 1:n.sp]
    dF.ax= [zeros(n.x,n.x) for _ in 1:n.sp]
    dF.ax′= [zeros(n.x,n.x) for _ in 1:n.sp]
    dF.xx = [zeros(n.x,n.x,n.x) for _ in 1:n.sp]
    dF.xX = [zeros(n.x,n.x,n.Q) for _ in 1:n.sp]
    dF.xx′= [zeros(n.x,n.x,n.x) for _ in 1:n.sp]
    dF.Xx = [zeros(n.x,n.x,n.Q) for _ in 1:n.sp]
    dF.XX = [zeros(n.x,n.Q,n.Q) for _ in 1:n.sp]
    dF.Xx′= [zeros(n.x,n.Q,n.x) for _ in 1:n.sp]
    dF.x′x= [zeros(n.x,n.x,n.x) for _ in 1:n.sp]
    dF.x′X= [zeros(n.x,n.x,n.x) for _ in 1:n.sp]
    dF.x′x′=[zeros(n.x,n.x,n.x) for _ in 1:n.sp]


    for j in 1:n.sp
        a_,θ = aθ_sp[j,:]
        argx̄ = x̄*Φ̃[:,j]
        argX̄= Q*X̄ #only interest rate and wages relevant
        argEx̄′ = x̄*Φ̃ᵉ[:,j]
        
        # first order
        @views dF.a[j]      = ForwardDiff.derivative(a->F(θ,a,argx̄,argX̄,argEx̄′),a_)
        @views dF.x[j]      = ForwardDiff.jacobian(x->F(θ,a_,x,argX̄,argEx̄′),argx̄)
        @views dF.x′[j]     = ForwardDiff.jacobian(x′->F(θ,a_,argx̄,argX̄,x′),argEx̄′)
        @views dF.X[j]      = ForwardDiff.jacobian(X->F(θ,a_,argx̄,X,argEx̄′),argX̄)
        
        # second order
        dF.aa[j]     = ForwardDiff.derivative(a2->ForwardDiff.derivative(a1->F(θ,a1,argx̄,argX̄,argEx̄′),a2),a_)
        dF.ax[j]    = ForwardDiff.jacobian(x -> ForwardDiff.derivative(a1->F(θ,a1,x,argX̄,argEx̄′),a_),argx̄)
        dF.ax′[j]    = ForwardDiff.jacobian(x′ -> ForwardDiff.derivative(a1->F(θ,a1,argx̄,argX̄,x′),a_),argEx̄′)
        dF.xx[j]     = reshape(ForwardDiff.jacobian(x1 -> ForwardDiff.jacobian(x2->F(θ,a_,x2,argX̄,argEx̄′),x1),argx̄),n.x,n.x,n.x)
        dF.xX[j]     = reshape(ForwardDiff.jacobian(X -> ForwardDiff.jacobian(x->F(θ,a_,x,X,argEx̄′),argx̄),argX̄),n.x,n.x,n.Q)
        dF.Xx[j]     = permutedims(dF.xX[j],[1,3,2])
        dF.XX[j]     = reshape(ForwardDiff.jacobian(X1 -> ForwardDiff.jacobian(X2->F(θ,a_,argx̄,X2,argEx̄′),X1),argX̄),n.x,n.Q,n.Q)
        dF.xx′[j]    = reshape(ForwardDiff.jacobian(x′ -> ForwardDiff.jacobian(x->F(θ,a_,x,argX̄,x′),argx̄),argEx̄′),n.x,n.x,n.x) 
        dF.x′x[j]    = permutedims(dF.xx′[j],[1,3,2])
        dF.Xx′[j]    = reshape(ForwardDiff.jacobian(x′ -> ForwardDiff.jacobian(X->F(θ,a_,argx̄,X,x′),argX̄),argEx̄′),n.x,n.Q,n.x)  
        dF.x′X[j]    = permutedims(dF.Xx′[j],[1,3,2])
        dF.x′x′[j]   = reshape(ForwardDiff.jacobian(x1′ -> ForwardDiff.jacobian(x2′->F(θ,a_,argx̄,argX̄,x2′),x1′),argEx̄′),n.x,n.x,n.x)
    end
   
    ZO.dF=dF;
end


function computeDerivativesF_portfolio!(ZO::ZerothOrderApproximation,inputs::Inputs)
    @unpack n,aθ_sp,Φ̃,X̄,x̄,Φ̃ᵉ,Q = ZO
    @unpack F = inputs
    dF=DerivativesF()
    dF.x= [zeros(n.x,n.x) for _ in 1:n.sp]
    dF.x′=[zeros(n.x,n.x) for _ in 1:n.sp]
    dF.X=[zeros(n.x,n.Q) for _ in 1:n.sp] # check
    dF.a = [zeros(n.x) for _ in 1:n.sp]
    dF.k = [zeros(n.x) for _ in 1:n.sp]

    for j in 1:n.sp
        a_,θ = aθ_sp[j,:]
        argx̄ = x̄*Φ̃[:,j]
        argX̄= Q*X̄ #only interest rate and wages relevant
        argEx̄′ = x̄*Φ̃ᵉ[:,j]
        argkRx = 0.
        
        # first order
        @views dF.a[j]      = ForwardDiff.derivative(a->F(θ,a,argx̄,argX̄,argEx̄′,argkRx),a_)
        @views dF.x[j]      = ForwardDiff.jacobian(x->F(θ,a_,x,argX̄,argEx̄′,argkRx),argx̄)
        @views dF.x′[j]     = ForwardDiff.jacobian(x′->F(θ,a_,argx̄,argX̄,x′,argkRx),argEx̄′)
        @views dF.X[j]      = ForwardDiff.jacobian(X->F(θ,a_,argx̄,X,argEx̄′,argkRx),argX̄)
        @views dF.k[j]      = ForwardDiff.derivative(kRx->F(θ,a_,argx̄,argX̄,argEx̄′,kRx),argkRx)

    end
   
    ZO.dF=dF;
end

function computeDerivativesG!(ZO::ZerothOrderApproximation,inputs::Inputs)
    if ZO.portfolio == true
        computeDerivativesG_portfolio!(ZO,inputs)
    else
        computeDerivativesG_no_portfolio!(ZO,inputs)
    end
end

function computeDerivativesG_no_portfolio!(ZO::ZerothOrderApproximation,inputs::Inputs)
    #construct F derivatives
    @unpack n, X̄, x̄, Φ,Q,P = ZO
    @unpack ω̄, Θ̄, G = inputs

    dG = DerivativesG()
    argΘ̄=Θ̄[1]

    X̄_ = P*X̄
    Ix̄ = x̄*Φ*ω̄

    #first order
    dG.x = ForwardDiff.jacobian(x->G(x,X̄_,X̄,X̄,[argΘ̄]),Ix̄) 
    dG.X_ = ForwardDiff.jacobian(X_->G(Ix̄,X_,X̄,X̄,[argΘ̄]),X̄_) 
    dG.X = ForwardDiff.jacobian(X->G(Ix̄,X̄_,X,X̄,[argΘ̄]),X̄) 
    dG.Xᵉ= ForwardDiff.jacobian(Xᵉ->G(Ix̄,X̄_,X̄,Xᵉ,[argΘ̄]),X̄)
    dG.Θ = ForwardDiff.jacobian(Θ->G(Ix̄,X̄_,X̄,X̄,Θ),[argΘ̄])

    #second order
    dG.xx   = reshape(ForwardDiff.jacobian(x2->ForwardDiff.jacobian(x1->G(x1,X̄_,X̄,X̄,[argΘ̄]),x2),Ix̄),n.X,n.x,n.x)
    dG.xX_  = reshape(ForwardDiff.jacobian(X_->ForwardDiff.jacobian(x->G(x,X_,X̄,X̄,[argΘ̄]),Ix̄),X̄_),n.X,n.x,n.A)
    dG.xX   = reshape(ForwardDiff.jacobian(X->ForwardDiff.jacobian(x->G(x,X̄_,X,X̄,[argΘ̄]),Ix̄),X̄),n.X,n.x,n.X)
    dG.xXᵉ   = reshape(ForwardDiff.jacobian(Xᵉ->ForwardDiff.jacobian(x->G(x,X̄_,X̄,Xᵉ,[argΘ̄]),Ix̄),X̄),n.X,n.x,n.X)
    dG.xΘ   = reshape(ForwardDiff.jacobian(Θ->ForwardDiff.jacobian(x->G(x,X̄_,X̄,X̄,Θ),Ix̄),[argΘ̄]),n.X,n.x,n.Θ)
    dG.X_X_ = reshape(ForwardDiff.jacobian(X2_->ForwardDiff.jacobian(X1_->G(Ix̄,X1_,X̄,X̄,[argΘ̄]),X2_),X̄_),n.X,n.A,n.A)
    dG.X_X  = reshape(ForwardDiff.jacobian(X->ForwardDiff.jacobian(X_->G(Ix̄,X_,X,X̄,Θ̄),X̄_),X̄),n.X,n.A,n.X)
    dG.X_Xᵉ = reshape(ForwardDiff.jacobian(Xᵉ->ForwardDiff.jacobian(X_->G(Ix̄,X_,X̄,Xᵉ,Θ̄),X̄_),X̄),n.X,n.A,n.X)
    dG.X_Θ  = reshape(ForwardDiff.jacobian(Θ->ForwardDiff.jacobian(X_->G(Ix̄,X_,X̄,X̄,Θ),X̄_),[argΘ̄]),n.X,n.A,n.Θ)
    dG.XX   = reshape(ForwardDiff.jacobian(X2->ForwardDiff.jacobian(X1->G(Ix̄,X̄_,X1,X̄,[argΘ̄]),X2),X̄),n.X,n.X,n.X)
    dG.XXᵉ  = reshape(ForwardDiff.jacobian(Xᵉ->ForwardDiff.jacobian(X->G(Ix̄,X̄_,X,Xᵉ,[argΘ̄]),X̄),X̄),n.X,n.X,n.X)
    dG.XΘ   = reshape(ForwardDiff.jacobian(Θ->ForwardDiff.jacobian(X->G(Ix̄,X̄_,X,X̄,Θ),X̄),[argΘ̄]),n.X,n.X,n.Θ)
    dG.XᵉXᵉ   = reshape(ForwardDiff.jacobian(Xᵉ2->ForwardDiff.jacobian(Xᵉ1->G(Ix̄,X̄_,X̄,Xᵉ1,[argΘ̄]),Xᵉ2),X̄),n.X,n.X,n.X)
    dG.ΘΘ   = reshape(ForwardDiff.jacobian(Θ2->ForwardDiff.jacobian(Θ1->G(Ix̄,X̄_,X̄,X̄,Θ1),Θ2),[argΘ̄]),n.X,n.Θ,n.Θ)

    #fixed weird forward diff bug for this derivative when $Xᵉ$ is not used
    dG.XᵉΘ   = permutedims(reshape(ForwardDiff.jacobian(Xᵉ->ForwardDiff.jacobian(Θ->G(Ix̄,X̄_,X̄,Xᵉ,Θ),[argΘ̄]),X̄),n.X,n.Θ,n.X),[1,3,2])

    ZO.dG=dG;
    
end


function computeDerivativesG_portfolio!(ZO::ZerothOrderApproximation,inputs::Inputs)
    #construct F derivatives
    @unpack n, X̄, x̄, Φ,Q,P = ZO
    @unpack ω̄, Θ̄, G = inputs

    dG = DerivativesG()
    argΘ̄=Θ̄[1]

    X̄_ = P*X̄
    Ix̄ = x̄*Φ*ω̄

    #first order
    dG.x = ForwardDiff.jacobian(x->G(x,X̄_,X̄,X̄,[argΘ̄]),Ix̄) 
    dG.X_ = ForwardDiff.jacobian(X_->G(Ix̄,X_,X̄,X̄,[argΘ̄]),X̄_) 
    dG.X = ForwardDiff.jacobian(X->G(Ix̄,X̄_,X,X̄,[argΘ̄]),X̄) 
    dG.Xᵉ= ForwardDiff.jacobian(Xᵉ->G(Ix̄,X̄_,X̄,Xᵉ,[argΘ̄]),X̄)
    dG.Θ = ForwardDiff.jacobian(Θ->G(Ix̄,X̄_,X̄,X̄,Θ),[argΘ̄])

    ZO.dG=dG;    
end
