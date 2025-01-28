include("AiyagariModel.jl")
include("SecondOrderApproximation.jl")
include("ABRS_KS_CA.jl")
using DataFrames,BenchmarkTools,CSV,LaTeXStrings



## Portfolio Choice


function construct_xfs_port!(AM::AiyagariModel, inputs::Inputs)
    @unpack λf,af,vf= AM #then unpack equilibrium objects
    inputs.aknots,inputs.ka,inputs.aθ_sp,inputs.aθ_Ω,inputs.ℵ = get_grids(AM)
    inputs.xf=[af,λf,vf]
    inputs.xlab=[:a,:λ,:v]
    inputs.alab=[:a]
    inputs.slab=:λ
end

function construct_X̄s_port!(AM::AiyagariModel,inputs::Inputs)
    @unpack ω̄,Na,θ,α,δ,N̄,K2Y,Θ̄ ,Ia,āθ̄,R̄,W̄, cf,vf, πθ= AM 
    S = length(θ)
    Y2K = 1/K2Y
    K2N = (Y2K/Θ̄)^(1/(α-1))
    K̄ = N̄*K2N
    Ȳ = Y2K*K̄
    C̄ = Ȳ - δ*K̄
    v = zeros(Ia*S) 
    #compute aggregate welfare
    for i in 1:Ia*S
        v[i] = vf(āθ̄[i,1],āθ̄[i,2]) #compute consumption at gridpoints
    end
    Iv = dot(v,ω̄)
    T̄ = 0.
    q̄ = 1.
    ℐ  = δ
    inputs.X̄=     [R̄,  W̄, T̄, ℐ, C̄, Ȳ,Iv, K̄, q̄,   0.,1/R̄,K̄]
    inputs.Xlab = [:Rf_,:W,:T,:ℐ,:C,:Y,:V,:K,:q,:Rx,:Qf,:Kd]
    inputs.Alab = [:K,:q,:Qf]
    inputs.Qlab = [:Rf_,:W,:T]
    inputs.Klab = [:Kd]
    inputs.Tlab = [:Qf]
    inputs.Rlab = [:Rx]
end

function F_port(para,θ,a_,x,X,x′,kRx)
    @unpack a_cutoff,β,σ,a̲ = para

    #unpack variables
    a,λ,v = x
    _,λᵉ,vᵉ = x′
    R,W,T = X

    c = (λ/R)^(-1/σ)

    ret = [R*a_+W*exp(θ)+T+kRx-c-a,
           v - c^(1-σ)/(1-σ) - β*vᵉ,
           β*λᵉ-λ/R]
    if a_ < a_cutoff[θ]
        ret[3] = a̲-a
    end
    return ret
end

function G_port(para,Ix,X_,X,Xᵉ,Θ)
    @unpack α,δ,N̄,ϕ,τ_Θ = para
    qK,_,Iv = Ix
    Rf_,W,T,ℐ,C,Y,V,K,q,Rx,Qf,Kd = X
    K_,q_,Qf_ = X_
    TFP, = Θ

    #now perform operations
    rK = α*TFP*K_^(α-1)*N̄^(1-α)
    ϕK = ℐ +  0.5*ϕ*(ℐ-δ)^2
    τ = τ_Θ*(TFP-1.0)
    R = Rf_+Rx
    return [qK - q*K,#
            (q*(1 - δ +ℐ) + rK - ϕK)/q_ - R,#
            (1-α)TFP*K_^α*N̄^(-α)*(1-τ) - W, #
            K - (1 - δ + ℐ)*K_,#
            Y - C - ϕK*K_,#
            q - 1 - ϕ*(ℐ-δ),#
            Y - TFP*K_^(α)*N̄^(1-α),
            V - Iv,#
            T - τ*(1-α)*TFP*K_^α*N̄^(-α),
            Rf_ - 1/Qf_,
            Kd - K_]#
end



## constr inputs for HA approx
inputs_port=Inputs()
construct_xfs_port!(AM,inputs_port)

# B) Aggregate policies
construct_X̄s_port!(AM,inputs_port)

# C) Equilibrium conditions
inputs_port.F = (θ,a_,x,X,xᵉ,kRx)->F_port(para,θ,a_,x,X,xᵉ,kRx)
inputs_port.G = (Ix,A_,X,Xᵉ,Θ)->G_port(para,Ix,A_,X,Xᵉ,Θ)

# D) Risk processes and other steady state objects
inputs_port.ω̄, inputs_port.Λ, inputs_port.Λ_z, inputs_port.πθ =  AM.ω̄, AM.Λ, AM.Λ_z, AM.πθ;
inputs_port.Θ̄, inputs_port.ρ_Θ, inputs_port.Σ_Θ = ones(1)*AM.Θ̄,ρ_Θ,Σ_Θ;
inputs_port.portfolio = true
## construct zeroth order HA approx
ZOport=ZerothOrderApproximation(inputs_port)
computeDerivativesF!(ZOport,inputs_port)
computeDerivativesG!(ZOport,inputs_port)

FOport = FirstOrderApproximation(ZOport,T)#define the FirstOrderApproximation object
compute_Θ_derivatives!(FOport)



# plotting
dfX_Θ =DataFrame(FO.X̂_Θt[1]',inputs.Xlab) 
dfX_Θ.t = 1:T
dfX_Θport =DataFrame(FOport.X̂_Θt[1]',inputs_port.Xlab)
dfX_Θport.t = 1:T

#compare FO responses
plot(dfX_Θ.t,dfX_Θ.K,label="only capital")
plot!(dfX_Θport.t,dfX_Θport.K,label="capital and bond")

#plot distribution
ω̄ = reshape(ZOport.ω̄,:,ZOport.n.θ)
ω̄a = sum(ω̄,dims=2)[:]
k = ZOport.Φ'*FOport.k̄
ka = sum(reshape(k,:,ZOport.n.θ).*ω̄,dims=2)./ω̄a
a = unique(ZOport.aθ_Ω[:,1])

Ȳ = dot(ZOport.X̄,inputs_port.Xlab.==:Y)
df_port_hist = DataFrame(a=a[2:900]./(Ȳ*4),equity=(ka./a)[2:900],bonds=1 .-(ka./a)[2:900]);
plot(df_port_hist.a,df_port_hist.equity,label="equity")
plot!(df_port_hist.a,df_port_hist.bonds,label="bonds")
xlabel!("Assets/GDP")
ylabel!("Share of Assets")