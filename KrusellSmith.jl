using Plots,DataFrames,LaTeXStrings,BenchmarkTools,CSV
include("AiyagariModel.jl")

# Choose Parameters for the KS Model
σ = 2.      # risk aversion
α = 0.36    # capital share
ρ_Θ = 0.995*ones(1,1)   # persistence of agg TFP
Σ_Θ= 0.014^2*ones(1,1)  # variance of agg TFP

Na = 200    # number of grid points/splines for individual policy function 
Ia = 1000   # number of grid points for agent distribution 
Nθ = 7

# Load them in the function that keep all the model parameters and objects needed for simmulation
AM = AiyagariModel()
AM.σ = σ
AM.α = α
AM.Na = Na  
AM.Ia = Ia
AM.Nθ = Nθ

AM.β=0.98
calibratesteadystate_λ!(AM);
calibratesteadystate_λ_precise!(AM)

include("ZerothOrderApproximation.jl")

# Save the model parameters to be used in the functions F and G
@with_kw mutable struct ModelParams
    a_cutoff::Dict{Float64,Float64} = Dict()
    a̲::Float64 = 0.
    β::Float64 = 0.
    σ::Float64 = 0.
    θ::Vector{Float64} = zeros(1)
    α::Float64 = 0.
    δ::Float64 = 0.
    N̄::Float64 = 0.
    
    ρ_Θ::Matrix{Float64} = zeros(1)
    Σ_Θ::Matrix{Float64} = zeros(1)
end

para=ModelParams(a_cutoff = AM.a_cutoff, a̲ = AM.a̲, β = AM.β, σ = AM.σ, θ = AM.θ, α = AM.α, δ = AM.δ, N̄ = AM.N̄, ρ_Θ = ρ_Θ, Σ_Θ = Σ_Θ);

function construct_individual_inputs!(AM::AiyagariModel, inputs::Inputs)
    @unpack λf,af= AM #then unpack equilibrium objects
    inputs.aknots,inputs.ka,inputs.aθ_sp,inputs.aθ_Ω,inputs.ℵ = get_grids(AM)
    inputs.xf=[af,λf]
    inputs.xlab=[:a,:λ]
    inputs.alab=[:a]
end

inputs = Inputs()
construct_individual_inputs!(AM,inputs)

function construct_X̄s!(AM::AiyagariModel,inputs::Inputs)
    @unpack α,δ,N̄,K2Y,Θ̄,R̄,W̄,= AM 
    Y2K = 1/K2Y
    K2N = (Y2K/Θ̄)^(1/(α-1))
    K̄ = N̄*K2N
    Ȳ = Y2K*K̄
    C̄ = Ȳ - δ*K̄
    inputs.X̄=     [R̄,  W̄, C̄, Ȳ, K̄]
    inputs.Xlab = [:R,:W,:C,:Y,:K]
    inputs.Alab = [:K]
    inputs.Qlab = [:R,:W]
end
construct_X̄s!(AM,inputs)

inputs.ω̄, inputs.Λ, inputs.Λ_z, inputs.πθ =  AM.ω̄, AM.Λ, AM.Λ_z, AM.πθ;
inputs.Θ̄, inputs.ρ_Θ, inputs.Σ_Θ = ones(1)*AM.Θ̄,ρ_Θ,Σ_Θ;


function F(para::ModelParams,θ,a_,x,X,xᵉ)
    @unpack a_cutoff,β,σ,a̲ = para

    #unpack variables
    a,λ  = x
    _,λᵉ = xᵉ
    R,W  = X

    u_c = λ/R 
    c = (u_c)^(-1/σ) #definition of λ
    
    ret = [R*a_+W*exp(θ)-c-a,
           β*λᵉ - u_c]
    if a_ < a_cutoff[θ] #check if agent is on borrowing constraint
        ret[2] = a̲-a
    end
    return ret
end
inputs.F = (θ,a_,x,X,xᵉ)->F(para,θ,a_,x,X,xᵉ)


function G(para::ModelParams,Ix,A_,X,Xᵉ,Θ)
    @unpack α,δ,N̄ = para
    Ia,_ = Ix
    R,W,C,Y,K = X
    K_, = A_
    TFP, = Θ
    #now perform operations
    rK = α*TFP*K_^(α-1)*N̄^(1-α)
    I  = K - (1-δ)*K_
    return [R - (1 - δ + rK),
            W - (1-α)TFP*K_^α*N̄^(-α),
            Y - TFP*K_^(α)*N̄^(1-α),
            Y - C - I,
            K - Ia]#
end
inputs.G = (Ix,A_,X,Xᵉ,Θ)->G(para,Ix,A_,X,Xᵉ,Θ)


ZO =ZerothOrderApproximation(inputs)

#check F 
x = ZO.x̄*ZO.Φ̃
xᵉ = ZO.x̄*ZO.Φ̃ᵉ

j = 67
Fres(j) = inputs.F(ZO.aθ_sp[j,2],ZO.aθ_sp[j,1],x[:,j],ZO.Q*ZO.X̄,xᵉ[:,j])
Fres.(1:ZO.n.sp)

Ix = ZO.x̄*ZO.Φ*ZO.ω̄
inputs.G(Ix,ZO.P*ZO.X̄,ZO.X̄,ZO.X̄,[1.])


computeDerivativesF!(ZO,inputs)
computeDerivativesG!(ZO,inputs)

T=300 #truncation length

# Load first order approximation function
include("FirstOrderApproximation.jl")

#define the FirstOrderApproximation class
FO = FirstOrderApproximation(ZO,T)

#computes the first order derivatives in the direction of the aggregate shocks
compute_Θ_derivatives!(FO)

#@benchmark compute_Θ_derivatives!(FO)

compute_x_Θ_derivatives!(FO)

iK = (inputs.Xlab.==:K)'*(1:ZO.n.X)
plot(FO.X̂_Θt[1][iK,:])


# Load second order approximation function
include("SecondOrderApproximation.jl")

#define the FirstOrderApproximation class
SO = SecondOrderApproximation(FO=FO) ; 

#computes the first order derivatives in the direction of the aggregate shocks
computeSecondOrder!(SO); 

#@benchmark computeSecondOrder!(SO)

# Simulation of second order interactions terms for a IRF of horizon 50
Tsim=50

X_ΘΘtk = compute_XZZjk(SO,1,1,1:Tsim)
#X_ZZ[0] correpond to second order approximation of a shock in the first period (0 lag)  
X_ΘΘtk[0] = SO.X̄_ΘΘ[1,1];

# Compute the size of aggregate shocks (in standard deviation):
shock   = 1         # use a shock of a size 1 sd of aggregate shock standard deviation
path    = zeros(Tsim)
path[1] = shock

# Compute the Impulse Response Functionb 
KtHA=getPath(SO,path,iK,X_ΘΘtk)
KtHA.t=1:Tsim

plot(1:Tsim,KtHA.X)
plot!(1:Tsim,KtHA.FO)


include("ABRS_KS.jl")
dKt,dRt,dWt =doABRS(AM,ZO.X̄,ZO.x̄,T,Σ_Θ,ρ_Θ);
@benchmark doABRS(AM,ZO.X̄,ZO.x̄,T,Σ_Θ,ρ_Θ)

KtABRS = getPathABRS(path,dKt,ZO.X̄[iK])
KtABRS.t=1:Tsim

plot(KtHA.t,KtHA.FO)
plot!(KtABRS.t,KtABRS.X)
plot!(KtHA.t,KtHA.X)


## Check Accuracy

#Check Accuracy
AMp = AiyagariModel()
AMp.τ_Θ= 0.
AMp.σ = σ
AMp.α = α
AMp.Na = 2*Na
AMp.Ia = 2*Ia
AMp.Nθ = Nθ

## save stuff from SS
AMp.β = 0.98
calibratesteadystate_λ_precise!(AMp)

iR,iW,iC,iY,iK = 1:5
shock = 2sqrt(ZO.Σ_Θ[1,1])
X̂t_fo = FO.X̄_Θ[1].*shock
X̂t_fo_SSJ = zeros(size(X̂t_fo))
X̂t_fo_SSJ[[iR,iW,iK],:] .= hcat(dRt,dWt,dKt)'.*shock./sqrt(ZO.Σ_Θ[1,1])
X̂t_so = X̂t_fo .+ 0.5.*SO.X̄_ΘΘ[1,1].*shock^2
#X̂t_so2 = X̂t_fo .+ 0.5.*SO2.X̄_ΘΘ[1,1].*shock^2

error_fo = check_accuracy(AMp,X̂t_fo)
error_so = check_accuracy(AMp,X̂t_so)
error_SSJ = check_accuracy(AMp,X̂t_fo_SSJ)

plot(error_fo)
plot!(error_SSJ)
plot!(error_so)