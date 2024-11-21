using Parameters,LinearAlgebra,BasisMatrices,SparseArrays,QuantEcon,Arpack,Roots, KrylovKit
using ForwardDiff,Dierckx,Plots,NLsolve

Approx1D = Interpoland{Basis{1,Tuple{SplineParams{Array{Float64,1}}}},Array{Float64,1},BasisMatrix{Tensor,SparseArrays.SparseMatrixCSC{Float64,Int64}}}

"""
    Contains the Parameters of the Aiyagari Model
"""
@with_kw mutable struct AiyagariModel
    #SteadyState Parameters
    α::Float64 = 0.3 #curvature of production function
    σ::Float64 = 2. #Risk Aversion
    β::Float64 = 0.9649 #Discount Factor
    ϕ::Float64 = 0.
    τ_Θ::Float64 = 0.
    σ_θ::Float64 = 0.13 #standard deviation of income shock θ
    ρ_θ::Float64 = 0.966 #persistence of the productivity shocks
    πθ::Matrix{Float64} = ones(1,1) #transition matrix
    a̲::Float64  = 0. #Borrowing constraint
    amax::Float64 = 500.
    Nθ::Int = 7 #number of gridpoints for the productivity shocks
    Na::Int = 60 #number of gridpoints for splines
    curv_interp::Float64 = 2.5 #controls spacing for interpolation
    ka::Int = 2 #Spline Order
    curv_hist::Float64 = 2. #controls spacing for histogram
    Ia::Int = 1000 #number of gridpoints for histogram
    R̄::Float64 = 1.01 #Equlibrium gross interest rate
    W̄::Float64 = 1.  #Equilibrium wage.
    Iv::Float64 =0.
    K̄::Float64=0.

    K2Y::Float64 = 2.7*4 #target capital to output ratio
    Θ̄::Float64 = 1. #Level of TFP
    δ::Float64 = 0.1 #depreciation rate
    N̄::Float64 = 1. #Average labor supply

    #Helpful Objects
    a′grid::Vector{Float64} = zeros(1)#grid vector for endogenous grid
    abasis::Basis{1, Tuple{SplineParams{Vector{Float64}}}} =Basis(SplineParams(collect(LinRange(0,1,60)),0,2)) #basis for endogenous grid
    a_cutoff::Dict{Float64,Float64} = Dict{Float64,Float64}() #Stores the points at which the borrowing constraint binds
    Φ::SparseMatrixCSC{Float64,Int64} = spzeros(Na,Na)
    EΦeg::SparseMatrixCSC{Float64,Int64} = spzeros(Na,Na) # to compute expectations of marginal utility at gridpoints on potential savings
    ω̄::Vector{Float64} = ones(1)/1000 #masses for the stationary distribution
    Λ::SparseMatrixCSC{Float64,Int64} = spzeros(Ia,Ia)
    Λ_z::SparseMatrixCSC{Float64,Int64} = spzeros(Ia,Ia)

    #policy functions
    cf::Vector{Spline1D} = Vector{Spline1D}(undef,1)
    λ̄coefs::Vector{Float64} = zeros(1) 
    λf::Function = (a,θ) -> 0.0
    af::Function = (a,θ) -> 0.0
    vf::Function= (a,θ) -> 0.0

    #grids
    âgrid::Vector{Float64} = zeros(1)
    āgrid::Vector{Float64} = zeros(1)    
    θ::Vector{Float64} = ones(1) #vector of productivity levels
    âθ̂::Vector{Vector{Float64}} = Vector{Vector{Float64}}(undef,1) #gridpoints for the approximations 
    āθ̄::Matrix{Float64} = ones(1,1) #gridpoints for the distribution

end

"""
computeoptimalconsumption_λ(AM::AiyagariModel,λcoefs)

This function takes as inputs spline coeff. for λ (marginal utility function) 
and uses Carrol's EGM to compute consumption policy function.  
"""
function computeoptimalconsumption_λ(AM::AiyagariModel,λcoefs::Vector{Float64})::Vector{Spline1D}
    @unpack σ,β,θ,a̲,EΦeg,a′grid,R̄,W̄, = AM
    S = length(θ)
    # compute expectations of marginal utility at gridpoints on potential savings
    #∑π_θ(s,s′) U_c (c(a′,s′)) = ∑π_θ(s,s′)λ(a′,s′)
    #∑π_θ(s,s′)∑λcoeff^j ϕ^j(a′,s^′)=EΦeg*λcoefs
    #so EΦeg is a matrix that evaluates the basis functions on the potential savings grid and sums up across shock

    Eλ′ = reshape(EΦeg*λcoefs,:,S) #precomputing expectations
    # compute consumption today implied by EE
    cEE = (β.*Eλ′).^(-1/σ) #consumption today
    # compute asset today implied by savings and consumtion
    Implieda = (a′grid .+ cEE .- W̄.*exp.(θ'))/R̄  #Implied assets today

    # now we want to figure out the consumption policy function
    # We know that for all a∈[a̲,implied_a(a′=̲a)] we have the borrowing constraint binding
    # so we check a[1,s]. If its bigger than ̲b then consumption is given by budget constraint
    # for all a≤a[1,s]. For the rest we use the interpolation using the EE consumption
      

    cf = Vector{Spline1D}(undef,S)#implied policy rules for each productivity
    for s in 1:S
        #with some productivities the borrowing constraint does not bind
        if issorted(Implieda[:,s])
            if Implieda[1,s] > a̲ #borrowing constraint binds
                AM.a_cutoff[θ[s]] = Implieda[1,s]
                #add extra points on the borrowing constraint for interpolation
                â = [a̲;Implieda[:,s]]
                ĉ = [R̄*a̲-a̲ + W̄*exp(θ[s]);cEE[:,s]]
                cf[s] = Spline1D(â,ĉ,k=1)
            else
                AM.a_cutoff[θ[s]] = -Inf
                cf[s] = Spline1D(Implieda[:,s],cEE[:,s],k=1)
            end
        else
            p = sortperm(Implieda[:,s])
            if Implieda[1,s] > a̲ #borrowing constraint binds
                AM.a_cutoff[θ[s]] = Implieda[p[1],s]
                #add extra points on the borrowing constraint for interpolation
                â = [a̲;Implieda[p,s]]
                ĉ = [R̄*a̲-a̲ + W̄*exp(θ[s]);cEE[p,s]]
                cf[s] = Spline1D(â,ĉ,k=1)
            else
                AM.a_cutoff[θ[s]] = -Inf
                cf[s] = Spline1D(Implieda[p,s],cEE[p,s],k=1)
            end
        end
    end
    return cf
end


"""
iterateλ_eg!(AM::AiyagariModel,luΦ,λcoefs)

Iterates on the λ function
λcoef Φ = T (λcoef)
where T   operator takes λcoeff, computes consumption policy function using EGM and then uses 
the form of  computes the marginal utility function to compute the new λ policy function
and finally finds the new λcoef by solving the linear system     
"""
function iterateλ_eg!(AM::AiyagariModel,luΦ,λcoefs)
    @unpack σ,β,Φ,θ,πθ,R̄,W̄ = AM
    S = length(θ)

    agrid = AM.a′grid
    Na = length(agrid)
    #Compute optimal consumption function for current λ function stored using Φ and λcoefs
    cf = computeoptimalconsumption_λ(AM,λcoefs)     
    #Compute the new λ function using the form of marginal utility
    λ = zeros(Na*S) 
    for s in 1:S
        λ[(s-1)*Na+1:s*Na] = R̄.*cf[s](agrid).^(-σ) #compute consumption at gridpoints
    end

    # update the λcoef using the linear system Φλcoefs = λ

    λcoefs′ = luΦ\λ
    diff = norm(λcoefs.-λcoefs′,Inf)
    λcoefs .= λcoefs′
    return diff
end

"""
    solveλ_eg!(AM::AiyagariModel,λcoefs,tol=1e-8)


Solves the functional equation for λ.
"""
function solveλ_eg!(AM::AiyagariModel,λcoefs,tol=1e-8)
    #increases stability to iterate on the time dimension a few times
    diff = 1.
    luΦ = lu(AM.Φ)
    while diff > tol
        #then use newtons method
        diff = iterateλ_eg!(AM,luΦ,λcoefs)
    end
end



"""
    solvebellman(AM::AiyagariModel)

Helper function to obtainthe  value function using optimal consumption functions
"""
function solvebellman(AM::AiyagariModel)
    @unpack σ,β,Φ,θ,πθ,R̄,W̄,a′grid,cf,abasis = AM
    S = length(θ)
    V = Vector{Interpoland}(undef,1)
    V = [Interpoland(abasis,a->((1-β)*a.+1).^(1-σ)./(1-σ)./(1-β)) for s in 1:S] #initialize with value function equal to β*a
    Vcoefs = vcat([V[s].coefs for s in 1:S]...)::Vector{Float64}
    agrid=a′grid
    Na = length(agrid)
    c = zeros(Na*S) 
    EΦ = spzeros(Na*S,Na*S)
    for s in 1:S
        for s′ in 1:S
            c[(s-1)*Na+1:s*Na] = cf[s](agrid) #compute consumption at gridpoints
            a′ = R̄*agrid .+ exp(θ[s])*W̄ .- c[(s-1)*Na+1:s*Na] #asset choice
            EΦ[(s-1)*Na+1:s*Na,(s′-1)*Na+1:s′*Na] = πθ[s,s′]*BasisMatrix(abasis,Direct(),a′).vals[1][:]
        end
    end


    Jac =  Φ .-  β.*EΦ
    #the next line solves  Jac*Vcoefs=c.^(1-σ)./(1-σ)
    Vcoefs=Jac\c.^(1-σ)./(1-σ)


    for s in 1:S
        V[s].coefs .= Vcoefs[1+(s-1)*Na:s*Na]
    end
    return V 
end



"""
    setupgrids_and_VF!(AM::AiyagariModel,amax,curv=1.7)

Setup grid points for the AiyagariModel Model given parameters
"""
function setupgrids_and_VF!(AM::AiyagariModel)
    @unpack a̲,Na,ρ_θ,σ_θ,β,σ,Ia,amax,curv_interp,curv_hist,abasis = AM
    S = AM.Nθ
    xvec = LinRange(0,1,Na-1).^curv_interp  #The Na -1 to to adjust for the quadratic splines
    a′gridknots = a̲ .+ (amax - a̲).*xvec #nonlinear grid knots
    
    #Now gridpoints for θ
    mc = rouwenhorst(S,ρ_θ,σ_θ)
    πθ = AM.πθ = mc.p
    θ = exp.(mc.state_values)
    πstat = real(eigs(πθ',nev=1)[2])
    πstat ./= sum(πstat)
    AM.θ = log.(θ./dot(πstat,θ))
    AM.N̄ = dot(πstat,exp.(AM.θ))

    #Grid pointsfor the policy and value functions
    abasis = Basis(SplineParams(a′gridknots,0,2))
    AM.abasis = abasis;    
    agrid = nodes(abasis)[1]
    AM.a′grid = agrid

    #Precompute  EΦeg that is used to  compute expectations of marginal utility at gridpoints on potential savings
    #∑π_θ(s,s′) U_c (c(a\prime,s′)) = ∑π_θ(s,s′)λ(a′,s′)
    #∑π_θ(s,s′)∑λcoeff^j ϕ^j(a′,s^′)=EΦeg*λcoefs
    #so EΦeg is a matrix that evaluates the basis functions on the exogenous potential savings grid and sums up across shock

    AM.EΦeg = kron(πθ,BasisMatrix(abasis,Direct(),AM.a′grid).vals[1])
    #Precompute Phi
    AM.Φ = kron(Matrix(I,S,S),BasisMatrix(abasis,Direct()).vals[1])

    λcoefs = AM.Φ\repeat((1/β).*((1-β)*agrid.+1).^(-σ),S)
    #Grid for distriaution
    xvec = LinRange(0,1,Ia).^curv_hist 
    āgrid = a̲ .+ (amax - a̲).*xvec #nonlinear grides
    AM.āθ̄ = hcat(kron(ones(S),āgrid),kron(AM.θ,ones(Ia)))
    AM.ω̄ = ones(Ia*S)/(Ia*S)

    #cutoffs
    AM.a_cutoff = Dict{Float64,Float64}()
    return λcoefs
end

"""
    find_stationarydistribution!(AM::AiyagariModel,V)

Computes the stationary distribution 
"""
function find_stationarydistribution_λ!(AM::AiyagariModel,λcoefs)
    @unpack θ,πθ,Ia,āθ̄,R̄,W̄ = AM
    S = length(θ)
    cf = computeoptimalconsumption_λ(AM,λcoefs)::Vector{Spline1D}
    ā = āθ̄[1:Ia,1] #grids are all the same for all shocks
    c = hcat([cf[s](ā) for s in 1:S]...) #consumption policy
    a′ = R̄.*ā .+ W̄.*exp.(θ') .- c #create a Ia×S grid for the policy rules
    
    #make sure we don't go beyond bounds.  Shouldn't bind if bmax is correct
    a′ = max.(min.(a′,ā[end]),ā[1])
    
    Qs = [kron(πθ[s,:],BasisMatrix(Basis(SplineParams(ā,0,1)),Direct(),@view a′[:,s]).vals[1]') for s in 1:S]

    AM.Λ = hcat(Qs...)
    AM.ω̄ .=  real(eigsolve(AM.Λ,AM.ω̄ ,1)[2])[1]
    AM.ω̄ ./= sum(AM.ω̄)
end

"""
    find_stationarydistribution!(AM::AiyagariModel,V)

Computes the stationary distribution 
"""
function find_transitionΛ_λ!(AM::AiyagariModel,λcoefs)
    @unpack θ,πθ,Ia,āθ̄,R̄,W̄ = AM
    S = length(θ)
    cf = computeoptimalconsumption_λ(AM,λcoefs)::Vector{Spline1D}
    ā = āθ̄[1:Ia,1] #grids are all the same for all shocks
    c = hcat([cf[s](ā) for s in 1:S]...) #consumption policy
    a′ = R̄.*ā .+ W̄.*exp.(θ') .- c #create a Ia×S grid for the policy rules
    
    #make sure we don't go aeyond bounds.  Shouldn't bind if amax is correct
    a′ = max.(min.(a′,ā[end]),ā[1])
    
    Qs = [kron(πθ[s,:],BasisMatrix(Basis(SplineParams(ā,0,1)),Direct(),@view a′[:,s]).vals[1]') for s in 1:S]

    AM.Λ = hcat(Qs...)
end


function find_ω̄!(AM,tol=1e-8)  
    n =1
    diff = 1
    ω1 = copy(AM.ω̄)
    ω2 = copy(AM.ω̄)
    while diff > tol
        if n%2 ==0
            mul!(ω2,AM.Λ,ω1)
        else
            mul!(ω1,AM.Λ,ω2)
        end
        if n%100 == 0
            diff = norm(ω1-ω2,Inf)
        end
        n+=1
    end
    AM.ω̄ = ω2
end


"""
calibratesteadystate_λ!(AM::AiyagariModel)

Solves for the steady state without aggregate shocks
"""
function calibratesteadystate_λ!(AM::AiyagariModel)
    λcoefs = setupgrids_and_VF!(AM)
    @unpack Θ̄,α,N̄,K2Y,R̄ = AM
    

    #Compute objects from targeted moments
    Y2K = 1/K2Y
    AM.δ = δ = α*Y2K + 1 - R̄ #matching capital to output ratio and interest rate gives depreciation rate
    K2N = (Y2K/Θ̄)^(1/(α-1)) #relationship between capital to output and capital to labor
    AM.K̄ =  K̄ = K2N*N̄
    AM.W̄ = (1-α)*Θ̄*K2N^α

    function βres(β)
        AM.β = β[1]
        solveλ_eg!(AM,λcoefs)
        find_transitionΛ_λ!(AM,λcoefs)
        find_ω̄!(AM)
        #find_stationarydistribution_λ!(AM,λcoefs)

        return dot(AM.ω̄,AM.āθ̄[:,1]) - K̄
    end
    #Q̄ = 1/R̄
    #β= fzero(βres,Q̄^30,Q̄)
    ret = nlsolve(βres,[AM.β])
    #if !converged(ret)
    #    error("Could not find steady state")
    #end
    βres(ret.zero)
    save_policy_functions!(AM,λcoefs)
    save_Λs!(AM)
    AM.λ̄coefs = λcoefs
end



"""
calibratesteadystate_λ!(AM::AiyagariModel)

Solves for the steady state without aggregate shocks
"""
function calibratesteadystate_λ_precise!(AM::AiyagariModel)
    λcoefs = setupgrids_and_VF!(AM)
    @unpack Θ̄,α,N̄,K2Y,R̄ = AM
    

    #Compute objects from targeted moments
    Y2K = 1/K2Y
    AM.δ = δ = α*Y2K + 1 - R̄ #matching capital to output ratio and interest rate gives depreciation rate
    K2N = (Y2K/Θ̄)^(1/(α-1)) #relationship between capital to output and capital to labor
    K̄ = K2N*N̄
    AM.W̄ = (1-α)*Θ̄*K2N^α

    function βres(β)
        AM.β = β[1]
        solveλ_eg!(AM,λcoefs,1e-13)
        #find_transitionΛ_λ!(AM,λcoefs)
        #find_ω̄!(AM)
        find_stationarydistribution_λ!(AM,λcoefs)

        return dot(AM.ω̄,AM.āθ̄[:,1]) - K̄
    end
    #Q̄ = 1/R̄
    #fzero(βres,Q̄^30,Q̄)
    ret = nlsolve(βres,[AM.β])
    if !converged(ret)
        error("Could not find steady state")
    end
    βres(ret.zero)
    save_policy_functions!(AM,λcoefs)
    save_Λs!(AM)
    AM.λ̄coefs = λcoefs
end

 """
    save_policy_functions!(AM::AiyagariModel)

Saves the policy functions in the AiyagariModel object
"""

function save_policy_functions!(AM::AiyagariModel,λcoefs)
    @unpack R̄,W̄,πθ,σ,Ia,Nθ,Na,curv_interp,curv_hist,a̲,amax= AM #then unpack equilibrium objects
    cf = computeoptimalconsumption_λ(AM,λcoefs)
    AM.cf=cf
    V̄=solvebellman(AM)
    λf(a,θ) = cf[θ.==AM.θ][1](a).^(-σ)*AM.R̄
    af(a,θ) = AM.R̄*a .+ AM.W̄*exp(θ) .- AM.cf[θ.==AM.θ][1](a) #helper function for debt policy
    vf(a,θ) = V̄[θ.==AM.θ][1](a)
    AM.λf=λf
    AM.af=af
    AM.vf=vf
    xvec = LinRange(0,1,Na-1).^curv_interp  #The Na -1 to adjust for the quadratic splines
    AM.âgrid = a̲ .+ (amax - a̲).*xvec #nonlinear grid for knot points
    xvec = LinRange(0,1,Ia).^curv_hist 
    AM.āgrid = a̲ .+ (amax - a̲).*xvec #nonlinear grids for distribution
end



function save_Λs!(AM::AiyagariModel)
    @unpack R̄,W̄,θ,πθ,σ,Ia,Nθ,Na,curv_interp,curv_hist,a̲,amax,āθ̄, cf,abasis= AM #then unpack equilibrium objects
    S = length(θ)
    
    âθ̂ = hcat(kron(ones(S),nodes(abasis)[1]),kron(1:S,ones(length(abasis))))
    agrid = nodes(abasis)[1]
    N = length(agrid)
    
    #construct Λ_z
    ā = āθ̄[1:Ia,1] #grids are all the same for all shocks
    c = hcat([cf[s](ā) for s in 1:S]...) #consumption policy
    a′ = R̄.*ā .+ W̄.*exp.(θ') .- c #create a Ia×S grid for the policy rules
    
    #make sure we don't go beyond bounds.  Shouldn't bind if amax is correct
    a′ = max.(min.(a′,ā[end]),ā[1])
    ābasis = Basis(SplineParams(ā,0,1))
    f(x̂) = BasisMatrix(ābasis,Direct(),[x̂]).vals[1]

    Qa_a = spzeros(Ia*S,Ia)
    for i in 1:length(a′)
        Qa_a[i,:] = ForwardDiff.derivative(f,a′[i])
    end
    Q_a = spzeros(Ia*S,Ia*S)
    for s in 1:S
        Q_a[1+(s-1)*Ia:s*Ia,:] = kron(reshape(πθ[s,:],1,:),Qa_a[1+(s-1)*Ia:s*Ia,:]) 
    end
    
    #construct Λ_z for histogram
    AM.Λ_z = Q_a'
end


function get_grids(AM)
    @unpack âgrid,āgrid,πθ,θ= AM #then unpack equilibrium objects
    aknots = âgrid
    a_sp = nodes(SplineParams(aknots,0,AM.ka)) #construct gridpoints from knots
    a_Ω = āgrid
    nθ,nsp,nΩ = size(πθ,1),length(a_sp),length(a_Ω)
    aθ_sp = hcat(kron(ones(nθ),a_sp),kron(θ,ones(nsp)))
    aθ_Ω = hcat(kron(ones(nθ),a_Ω),kron(θ,ones(nΩ)))

    #next get kinks
    ℵ = Int[]
    for s in 1:nθ
        if AM.a_cutoff[θ[s]] > -Inf
            push!(ℵ,findlast(a_sp .< AM.a_cutoff[θ[s]])+(s-1)*nsp)
        end
    end 

    return aknots,AM.ka,aθ_sp,aθ_Ω,ℵ
end

#= 

# Choose Parameters for the run
σ = 5 #risk aversion
ϕ = 35.0 #radjustment cost
α = 0.36 # capital share

ρ_Θ = 0.8 #persistence of agg TFP
Σ_Θ= 0.014^2*ones(1,1)

Nb = 120
Ib = 1000
T=400 #truncation length

AM = AiyagariModel()
AM.τ_θ= 0.
AM.ϕ = ϕ
AM.σ = σ
AM.α = α
AM.Nb = Nb
AM.Ib = Ib

save_policy_functions!(AM)
save_agg!(AM)
save_Λs!(AM) 

 =## Stationary distribution in assets 
#plot(AM.z̄[1:Ib,1],AM.ω̄[1:Ib,1]./sum(AM.ω̄[1:Ib,1]),title = "Stationary asset distribution", label = "Lowest income")
#plot!(AM.z̄[3*Ib+1:4*Ib,1],AM.ω̄[3*Ib+1:4*Ib,1] ./ sum(AM.ω̄[3*Ib+1:4*Ib,1]),label = "Median income" )
#plot!(AM.z̄[6*Ib+1:7*Ib,1],AM.ω̄[6*Ib+1:7*Ib,1]./ sum(AM.ω̄[6*Ib+1:7*Ib,1]),label = "Top income" )
#xlims!((-1,200))
#ylims!((0,0.015))
