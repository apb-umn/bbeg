include("FirstOrderApproximation.jl")
using TimerOutputs



"""
SecondOrderApproximation{Model}

Holds all the objects necessary for a second order approximation of a 
given Model.  Will assume F_x(M::Model,z) etc. exists.
"""
@with_kw mutable struct SecondOrderApproximation
    FO::FirstOrderApproximation
    T::Int = FO.T #Length of IRF
    

    #compenents of the derivative with respect to second state
    X_02::Vector{Float64} = zeros(1) #initial predetermined for second derivative
    Θ_02::Vector{Float64} = zeros(1) #initial exogenous states
    Ω̂k::Matrix{Float64} =  zeros(1,1)
    x̂k::Vector{Matrix{Float64}} =  Vector{Matrix{Float64}}(undef,1)
    X̂k::Matrix{Float64} = zeros(1,1)

    ##PreComputations
    #x̄_zz object computed for accuracy
    x̄_aa::Matrix{Float64} = zeros(1,1)
    Ixaaδ::Matrix{Float64} = zeros(1,1)
    x̄_aaδ::Matrix{Float64} = zeros(0,0) #kink component

    ##Lemma 2 terms
    GΘΘtk::Matrix{Float64} = zeros(1,1) #nX×T

    ##Lemma 3 terms
    xσσ::Matrix{Float64} = zeros(0,0)
    x∞::Vector{Array{Float64,3}} = Vector{Array{Float64,3}}(undef,1)
    xtk::Array{Float64,3} = zeros(0,0,0)
    Ixtkδ::Array{Float64,3} = zeros(0,0,0)
    
    #Precomputed components of xtk
    Fxk::Matrix{Matrix{Float64}} = zeros(0,0)
    FXk::Matrix{Matrix{Float64}} = zeros(0,0)
    Fx′k::Matrix{Matrix{Float64}} = zeros(0,0)
    FxkΘ::Vector{Matrix{Matrix{Float64}}} = zeros(0)
    FXkΘ::Vector{Matrix{Matrix{Float64}}} = zeros(0)
    Fx′kΘ::Vector{Matrix{Matrix{Float64}}} = zeros(0)
    #delta function components
    
    ##Lemma 4 terms
    btk::Matrix{Float64} = zeros(1,1) 
    ctk::Matrix{Float64} = zeros(1,1)

    #Corollary 2 terms
    Laa1::SparseMatrixCSC{Float64,Int64} = spzeros(1,1)
    Laa2::SparseMatrixCSC{Float64,Int64} = spzeros(1,1)
    Btk::Matrix{Float64} = zeros(1,1) 
    Ctk::Matrix{Float64} = zeros(1,1)
    IBσσ::Matrix{Float64} = zeros(1,1)

    #Proposition 1 terms
    J∞::Array{Float64,4} = zeros(0,0,0,0)

    #Lemma 3 SV terms
    xσσΥ::Matrix{Float64} = zeros(0,0)
    #Corollary 2 SV terms
    Bσσ::Matrix{Float64} = zeros(0,0) #do we need this?
    IBσσΥ::Matrix{Float64} = zeros(1,1) #do we need this?
    IBΥ::Matrix{Float64} = zeros(1,1)



    #Outputs
    X̂tk::Matrix{Float64} = zeros(1,1)
    x̂tk::Vector{Matrix{Float64}} =  Vector{Matrix{Float64}}(undef,1)
    Ω̂tk::Matrix{Float64} = zeros(1,1) 

    X̂_ΘΘ::Matrix{Matrix{Float64}} = Matrix{Matrix{Float64}}(undef,1,1)
    x̂_ΘΘ::Matrix{Vector{Matrix{Float64}}} =  Matrix{Vector{Matrix{Float64}}}(undef,1,1)
    Ixtkδ_ΘΘ::Matrix{Array{Float64,3}} =  Matrix{Array{Float64,3}}(undef,1,1)
    
    X̂_σσ::Matrix{Float64} = zeros(1,1)
    X̂Υ::Matrix{Float64} = zeros(1,1)
    x̂_σσ::Vector{Matrix{Float64}} = Vector{Matrix{Float64}}(undef,1)
    Ω̄_σσ::Matrix{Float64} = zeros(1,1)

    x̂_σσΥ::Vector{Matrix{Float64}} = Vector{Matrix{Float64}}(undef,1)
end

function Base.copy(SO::SecondOrderApproximation)
    SOtemp = SecondOrderApproximation(FO=SO.FO)
    for k in fieldnames(SecondOrderApproximation)
        setfield!(SOtemp,k,getfield(SO,k))
    end
    return SOtemp
end


"""
    computex̄_aa!(SO::SecondOrderApproximation)

Computes the coefficients of x̄_aa
"""
function computex̄_aa!(SO::SecondOrderApproximation)
    @unpack FO = SO
    @unpack f,ZO,x̄_a = FO
    @unpack n,x̄,Φ̃ᵉₐ,Φ̃ₐ,Φ̃,Φ̃ᵉ,p,dF = ZO
    B = spzeros(n.x*n.sp,n.x*n.sp)
    x̄_a =x̄_a*Φ̃# x̄*Φ̃ₐ
    Ex̄_a =x̄_a*Φ̃ᵉ#  x̄*Φ̃ᵉₐ
    Ex̄_aa = x̄_a*Φ̃ᵉₐ
    x̄_aa = similar(x̄)
    for j in 1:n.sp
        x̄_aj = x̄_a[:,j] 
        ā_aj = (p*x̄_aj)[1]
        x̄′_a = view(Ex̄_a,:,j)*z̄_aj

        x̄_zz[:,j] = f[j]*(dF.x′[j]*view(Ex̄_aa,:,j)*ā_aj*ā_aj + dF.aa[j] + 2*dF.ax[j]*x̄_aj + 2*dF.ax′[j]* x̄′_a
                    + dF.xx[j]⋅(x̄_aj,x̄_aj) + 2*dF.xx′[j]⋅(x̄_aj,x̄′_a)
                    +dF.x′x′[j]⋅(x̄′_a,x̄′_a))
    end
    SO.x̄_aa = x̄_aa/Φ̃
end



"""
    compute_Lemma2_ZZ!(SO)

Computes the GΘΘtk terms
"""
function compute_Lemma2_ZZ!(SO::SecondOrderApproximation)
    @unpack FO,Ω̂k,x̂k,X̂k,Θ_02 = SO
    @unpack Ω̂t,x̂t,X̂t,Θ_0,ZO = FO
    @unpack x̄,Φ,Φₐ,ω̄,dG,ρ_Θ,P,Q,p,n = ZO
    T = length(x̂k)

    #Compute RFOT
    GΘΘtk = SO.GΘΘtk = zeros(n.X,T)
    IntΦ̃ = Φ * ω̄ #integration operator
    for t in 1:T
        X̂t_t = X̂t[:,t]
        IntΦ̃Ω_Z,IntΦ̃Ω_Z2 = Φₐ*view(Ω̂t,:,t),Φₐ*view(Ω̂k,:,t)
        x̂It = x̂t[t]*IntΦ̃ + x̄*IntΦ̃Ω_Z
        X̂k_t = X̂k[:,t]
        x̂Ik = x̂k[t]*IntΦ̃ + x̄*IntΦ̃Ω_Z2
        if t == 1
            X̂t_t_ = FO.X_0
            X̂k_t_ = SO.X_02
        else
            X̂t_t_ = P*view(X̂t,:,t-1)
            X̂k_t_ = P*view(X̂k,:,t-1)
        end
        if t==T
            X̂t_tᵉ = zeros(n.X)
            X̂k_tᵉ = zeros(n.X)
        else
            X̂t_tᵉ = X̂t[:,t+1]
            X̂k_tᵉ = X̂k[:,t+1]
        end

        θt,θk = ρ_Θ^(t-1)*Θ_0,ρ_Θ^(t-1)*Θ_02
        GΘΘtk[:,t] = (dG.xx⋅(x̂It,x̂Ik) .+ dG.xX⋅(x̂It,X̂k_t) .+ dG.xX⋅(x̂Ik,X̂t_t)  .+ dG.xΘ⋅(x̂It,θk) .+ dG.xΘ⋅(x̂Ik,θt)
                .+ dG.xXᵉ⋅(x̂It,X̂k_tᵉ) .+ dG.xXᵉ⋅(x̂Ik,X̂t_tᵉ) .+dG.xX_⋅(x̂It,X̂k_t_) .+ dG.xX_⋅(x̂Ik,X̂t_t_)
                .+ dG.X_X_⋅(X̂t_t_,X̂k_t_) .+ dG.X_X⋅(X̂t_t_,X̂k_t) .+ dG.X_X⋅(X̂k_t_,X̂t_t) .+ dG.X_Xᵉ⋅(X̂t_t_,X̂k_tᵉ)
                .+ dG.X_Xᵉ⋅(X̂k_t_,X̂t_tᵉ) .+ dG.X_Θ⋅(X̂t_t_,θk) .+ dG.X_Θ⋅(X̂k_t_,θt) .+ dG.XX⋅(X̂t_t,X̂k_t)
                .+ dG.XXᵉ⋅(X̂t_t,X̂k_tᵉ) .+ dG.XXᵉ⋅(X̂k_t,X̂t_tᵉ) .+ dG.XΘ⋅(X̂t_t,θk) .+ dG.XΘ⋅(X̂k_t,θt)
                .+ dG.XᵉXᵉ⋅(X̂t_tᵉ,X̂k_tᵉ) .+ dG.XᵉΘ⋅(X̂t_tᵉ,θk) .+ dG.XᵉΘ⋅(X̂k_tᵉ,θt) .+ dG.ΘΘ⋅(θt,θk))

        GΘΘtk[:,t] .+= dG.x*(x̂t[t]*IntΦ̃Ω_Z .+ x̂k[t]*IntΦ̃Ω_Z2)
    end
    
end


"""
    compute_E_components!(SO)  
    
Computes the AF_Zx,AF_ZX,AF_Zx′ matrices that are components of the 
computation of the E matrices.
"""
function compute_lemma3_components!(SO::SecondOrderApproximation)
    @unpack FO,T = SO
    @unpack ZO,X̂t,x̂t = FO
    @unpack x̄,Φ̃ᵉₐ,Φ̃,Φ̃ᵉ,p,Q,dF,n = ZO

    Fxk = SO.Fxk = Matrix{Matrix{Float64}}(undef,n.sp,T)
    FXk = SO.FXk = Matrix{Matrix{Float64}}(undef,n.sp,T)
    Fx′k= SO.Fx′k = Matrix{Matrix{Float64}}(undef,n.sp,T)

    Ex̄_a = x̄*Φ̃ᵉₐ
    for k in 1:T
        X̂t_k = Q*view(X̂t,:,k)
        x̂t_k = x̂t[k]*Φ̃
        ât_k = (p*x̂t_k)[:]
        if k < T
            Ex̂t_k = x̂t[k+1]*Φ̃ᵉ 
        else
            Ex̂t_k = zeros(n.x,n.sp)
        end
        for j in 1:n.sp
            Ex̄′t_k = @views Ex̄_a[:,j].* ât_k[j] .+ Ex̂t_k[:,j]#t==T ? (x̄_a*Φ̃ᵉj)*ât_t : (x̄_a*Φ̃ᵉj)*ât_t + x̄_Z[t+1]*Φ̃ᵉj
            x̂t_kj = @views x̂t_k[:,j]
            @tensor Fxk[j,k][i,l] := dF.xx[j][i,l,o]*x̂t_kj[o] + dF.xX[j][i,l,o]*X̂t_k[o] + dF.xx′[j][i,l,o]*Ex̄′t_k[o]
            @tensor FXk[j,k][i,l] := dF.Xx[j][i,l,o]*x̂t_kj[o] + dF.XX[j][i,l,o]*X̂t_k[o] + dF.Xx′[j][i,l,o]*Ex̄′t_k[o]
            @tensor Fx′k[j,k][i,l] := dF.x′x[j][i,l,o]*x̂t_kj[o] + dF.x′X[j][i,l,o]*X̂t_k[o] + dF.x′x′[j][i,l,o]*Ex̄′t_k[o]
        end
    end
end

"""
    compute_lemma3_ZZ!(SO)

Computes the xtk terms for a given k
"""
function compute_lemma3_ZZ!(SO::SecondOrderApproximation)
    @unpack FO,Fxk,FXk,Fx′k,X̂k,x̂k,x̄_aa,Ixaaδ = SO
    @unpack ZO,f,X̂t,x̂t,x̄_a = FO
    @unpack x̄,Φ̃ᵉₐ,Φ̃,Φ̃ᵉ,Φ̃ᵉₐ,p,Q,n,dF = ZO
    T = length(x̂k) #allow for x̄_Z2 to have a shorter truncation
    luΦ̃ = lu(Φ̃)

    xtk_temp = zeros(n.x,n.sp)
    xtk = SO.xtk = zeros(n.x,n.sp,T)
    Ex̄_aa = x̄_a*Φ̃ᵉₐ
    Ex̄_a = x̄*Φ̃ᵉₐ
    for s in reverse(1:T)
        X̂t_s = @views Q*X̂t[:,s]
        X̂k_s = @views Q*X̂k[:,s]
        x̂t_s = x̂t[s]*Φ̃
        ât_s = (p*x̂t_s)[:]
        x̂k_s = x̂k[s]*Φ̃
        âk_s = (p*x̂k_s)[:]
        if s < T
            Ex̂t_a_s = x̂t[s+1]*Φ̃ᵉₐ
            Ex̂k_a_s = x̂k[s+1]*Φ̃ᵉₐ
            Ex̂k_s  =  x̂k[s+1]*Φ̃ᵉ
            Extk = xtk[:,:,s+1]*Φ̃ᵉ
        else
            Ex̂t_a_s = zeros(n.x,n.sp)
            Ex̂k_a_s = zeros(n.x,n.sp)
            Ex̂k_s  =  zeros(n.x,n.sp)
            Extk = zeros(n.x,n.sp)
        end
        
        for j in 1:n.sp
            Ex̄′k_s = view(Ex̄_a,:,j)*âk_s[j] .+ view(Ex̂k_s,:,j)

            xtk_temp[:,j]  .= dF.x′[j]*(view(Ex̄_aa,:,j)*ât_s[j]*âk_s[j].+ view(Ex̂k_a_s,:,j)*ât_s[j] .+ view(Ex̂t_a_s,:,j)*âk_s[j])
            xtk_temp[:,j] .+= Fxk[j,s]*view(x̂k_s,:,j) .+ FXk[j,s]*X̂k_s .+ Fx′k[j,s]*Ex̄′k_s
            xtk_temp[:,j] .+= dF.x′[j]*view(Extk,:,j)
            xtk_temp[:,j] = f[j]*view(xtk_temp,:,j)
        end
        xtk[:,:,s] = (luΦ̃'\xtk_temp')'  
    end
end


function compute_lemma3_ZZ_kink!(SO)
    @unpack FO,x̂k = SO
    @unpack ZO,x̂t,x̄_a = FO
    @unpack x̄,Φ̃,p,Q,n,ℵ,aθ_sp = ZO
    nℵ = length(ℵ)
    T = length(x̂k)
    
    Φ̃m = Φ̃[:,ℵ]
    Φ̃p = Φ̃[:,ℵ.+1]
    x̄Δ_a = x̄_a*Φ̃[:,ℵ.+1].-x̄_a*Φ̃[:,ℵ]
    āΔ_a = p*x̄Δ_a
    xtkδ  = zeros(n.x,nℵ,T)
    for t in 1:T
        x̄Δ_Z1 = x̂t[t]*Φ̃p .- x̂t[t]*Φ̃m
        x̄Δ_Z2 = x̂k[t]*Φ̃p .- x̂k[t]*Φ̃m
        astar_Z1 = -(āΔ_a).^(-1).*(p*x̄Δ_Z1)
        astar_Z2 = -(āΔ_a).^(-1).*(p*x̄Δ_Z2)
        xtkδ[:,:,t] .=  x̄Δ_a.*astar_Z1.*astar_Z2
    end
    #now construct Ixtkδ
    Ixtkδ = zeros(n.x,n.sp,T)
    for (iℵ,j) in enumerate(ℵ)
        #mask_stepfunction = filter(i->ẑ[i][1]>ẑ[j][1] && ẑ[i][2:end]==ẑ[j][2:end],1:n.ẑ) #mask for step function
        mask_stepfunction = filter(i->aθ_sp[i,1]>aθ_sp[j,1] && aθ_sp[i,2:end]==aθ_sp[j,2:end],1:n.sp) #mask for step function
        Ixtkδ[:,mask_stepfunction,:] .+= reshape(xtkδ[:,iℵ,:],n.x,1,T)
    end
    SO.Ixtkδ =  zeros(n.x,n.sp,T)
    luΦ̃ = lu(Φ̃)
    for t in 1:T
        @views SO.Ixtkδ[:,:,t] .= (luΦ̃'\Ixtkδ[:,:,t]')'
    end
end


"""
    compute_Lemma4_ZZ!(SO)

Computes the a and b elements from Lemma 4
"""
function compute_Lemma4_ZZ!(SO)
    @unpack FO,Ω̂k,x̂k,X̂k = SO
    @unpack ZO,Ω̂t,x̂t,X̂t,L = FO
    @unpack n,Φ,Φₐ,p,ω̄,Λ,Q,x̄ = ZO
    #Construct kink terms

    T = length(x̂k)
    btk= SO.btk = zeros(n.Ω,T)
    ctk= SO.ctk = zeros(n.Ω,T)
    for t in 2:T 
        ât = ((p*x̂t[t-1])*Φ)[:]
        âk = ((p*x̂k[t-1])*Φ)[:]
        ât_a = ((p*x̂t[t-1])*Φₐ)[:]
        âk_a = ((p*x̂k[t-1])*Φₐ)[:]
        #âk = ((p*x̂k[t-1])*Φ̃ā)[:]
        
        @views btk[:,t] .= Λ*(ât_a.*Ω̂k[:,t-1] .+ âk_a.*Ω̂t[:,t-1] )
        @views ctk[:,t] .= Λ*(ât.*âk.*ω̄) .+ L*(ât.*Ω̂k[:,t-1] .+ âk.*Ω̂t[:,t-1] ) 
    end
end

function construct_Laa!(SO)
    @unpack FO = SO
    @unpack ZO,Ω̂t,x̂t,X̂t,x̄_a = FO
    @unpack Φ,Φₐ,p,ω̄,Λ,Q,x̄,n = ZO

    ā_a = ((p*x̄)*Φₐ)[:]  #1xI array should work with broadcasting
    ā_aa = ((p*x̄_a)*Φₐ)[:]#((p*x̄_zz)*Φ)[:]#  #1xI array should work with broadcasting
    #This code stops the creation of a large dense matrix
    Laa1 = SO.Laa1 = copy(Λ)
    Laa2 = SO.Laa2 = copy(Λ)
    for j in 1:n.Ω
        for i in nzrange(Laa1,j)
            @inbounds Laa1.nzval[i] *= ā_aa[j]
        end
        for i in nzrange(Laa2,j)
            @inbounds Laa2.nzval[i] *= ā_a[j]*ā_a[j]
        end
    end
end


function compute_Corollary2_ZZ!(SO)
    @unpack FO,xtk,Ixtkδ,Laa1,Laa2,btk,ctk = SO
    @unpack ZO,L,M = FO
    @unpack ω̄,p,n,Φₐ,Λ = ZO
    Mda =  Λ*(Φₐ'.*ω̄)
    atk = (p*xtk)[1,:,:]
    Iatkδ = (p*Ixtkδ)[1,:,:]
    #apply iteration
    #Iz = length(ω̄)
    T = size(btk)[2] #use possibly shorter horizon
    Btk = SO.Btk = zeros(n.Ω,T)
    Ctk = SO.Ctk = zeros(n.Ω,T)
    for t in 2:T
        @views Btk[:,t] .= L*Btk[:,t-1] .+ Laa1*Ctk[:,t-1] .+ btk[:,t] .+ M*atk[:,t-1] .+ Mda*Iatkδ[:,t-1]
        @views Ctk[:,t] .= Laa2*Ctk[:,t-1] .+ ctk[:,t]
    end
end

"""
    compute_XZZ!(SO)

Computes the path of the second order derivatives X_ZZ
"""
function compute_XZZ!(SO::SecondOrderApproximation)
    @unpack FO,x̂k,GΘΘtk,Btk,Ctk,xtk,Ixtkδ = SO
    @unpack ZO,X̂t,luBB,x̄_a = FO
    @unpack x̄,aθ_Ω,Φ,Φₐ,ω̄,dG,P,Q,p,n,ρ_Θ = ZO
    T = size(FO.X̂t,2)
    T2 = length(x̂k)

    #Compute RFOT
    AA = zeros(n.X,T)
    for t in 1:T2
        IB = @views x̄*(Φₐ*Btk[:,t])
        IaaC = @views x̄_a*(Φₐ*Ctk[:,t])#x̄_zz*(Φ*Ctk[:,t])#
        Intxtk = @views xtk[:,:,t]*(Φ*ω̄) .+ Ixtkδ[:,:,t]*(Φₐ*ω̄)
        @views AA[:,t] .= dG.x*( IB .+ IaaC .+ Intxtk) .+ GΘΘtk[:,t] #Htk+GΘΘtk
    end

    #SO.X̂tk = reshape(-BB[1:n.X*T,1:n.X*T]\AA[:],n.X,T)
    SO.X̂tk = reshape(-(luBB\AA[:])[1:n.X*T2],n.X,T2)
end

"""
    compute_xZZ!(SO)

Computes the path of the second order derivatives x_ZZ
"""
function compute_xZZ!(SO::SecondOrderApproximation)
    @unpack FO,xtk,X̂tk = SO
    @unpack ZO,T = FO
    @unpack p,Q,n = ZO

    x̂tk = SO.x̂tk # =Vector{Matrix{Float64}}(undef,T)
#    x = permutedims(FO.x,[1,3,2])
    flipdims = xs->permutedims(xs,[1,3,2])
    x = flipdims.(FO.x)
    for t in 1:T
        x̂tk[t] = xtk[:,:,t]
        for s in 1:T-t+1
            @views x̂tk[t] += x[s]*(Q*X̂tk[:,t+s-1])
        end
    end
end

"""
    compute_xZZ!(SO)

Computes the path of the second order derivatives x_ZZ
"""
function compute_xZZ0!(SO::SecondOrderApproximation)
    @unpack FO,xtk,X̂tk = SO
    @unpack ZO,T = FO
    @unpack p,Q,n = ZO

    x̂tk = SO.x̂tk = Vector{Matrix{Float64}}(undef,T)
    flipdims = xs->permutedims(xs,[1,3,2])
    x = flipdims.(FO.x)
    x̂tk[1] = xtk[:,:,1]
    for s in 1:T
        @views x̂tk[1] += x[s]*(Q*X̂tk[:,s])
    end
end


"""
    compute_lemma3_σσ!(SO)

Computes the x_σσ terms from Lemma 3, as well as x∞ terms.
"""
function compute_lemma3_σσ!(SO::SecondOrderApproximation)
    @unpack FO,T,x̂_ΘΘ,x∞,Ixtkδ_ΘΘ = SO
    @unpack ZO,f,x̂_Θt = FO
    @unpack x̄,Φ̃,Φ̃ᵉ,Φ̃ᵉₐ,p,dF,n,Σ_Θ = ZO
    B = spzeros(n.x*n.sp,n.x*n.sp)
    F̂_σσ = zeros(n.x,n.sp)
    fF_X = zeros(n.x,n.sp,n.Q)
    fF_x′ = Vector{Matrix{Float64}}(undef,n.sp)

    Ex̂_ΘΘ′ = sum((x̂_ΘΘ[i][1]*Φ̃ᵉ.+Ixtkδ_ΘΘ[i][:,:,1]*Φ̃ᵉₐ)*Σ_Θ[i] for i in 1:length(Σ_Θ))
    #Ex̄_σσ′ = sum(x̄_Θ[i][1]*Φ̃ᵉ*μ_Θσσ[i] for i in 1:length(μ_Θσσ)) #account for change in mean
    for j in 1:n.sp

        fF_x′[j] = f[j]*dF.x′[j]
        B[1+(j-1)*n.x:j*n.x,:] = kron(sparse(Φ̃[:,j]'),Matrix(I,n.x,n.x)) - kron(sparse(Φ̃ᵉ[:,j]'),fF_x′[j])

        F̂_σσ[:,j] = fF_x′[j]*(Ex̂_ΘΘ′[:,j])#+Ex̂_σσ′[:,j])
        fF_X[:,j,:] = f[j] * dF.X[j]
    end
    luB = lu(B)
    xσσ = SO.xσσ = reshape(luB\F̂_σσ[:],n.x,:)
    x̂_σσX = reshape(luB\reshape(fF_X,:,n.Q),n.x,n.sp,n.Q)
    x∞ = SO.x∞ = Vector{Array{Float64,3}}(undef,T+1)
    x∞[1] = permutedims(x̂_σσX,[1,3,2])
    x∞temp = zeros(n.x,n.Q,n.sp)
    luΦ̃ = lu(Φ̃)
    for s in 2:T+1
        Ex∞ = x∞[s-1]*Φ̃ᵉ
        for j in 1:n.sp
            @views x∞temp[:,:,j] .= fF_x′[j]*Ex∞[:,:,j]
        end
        x∞[s] = x∞temp/luΦ̃
    end
end



function compute_corollary2_σσ!(SO::SecondOrderApproximation)
    @unpack FO,xσσ = SO
    @unpack ZO,L,M,T,ILM = FO
    @unpack p,n = ZO
    aσσ = (p*xσσ)[:]
    SO.IBσσ = cumsum([zeros(n.x) ILM*aσσ][:,1:T],dims=2)
end


function compute_proposition1_σσ!(SO::SecondOrderApproximation) 
    @unpack FO,x∞,xσσ,X̂_ΘΘ = SO
    @unpack ZO,J,T,L,M = FO
    @unpack p,ω̄,Φ,x̄,Φₐ,dG,n,Q,P  = ZO
    Iop = x̄*Φₐ
    IntΦ̃ = Φ * ω̄
    J∞ = SO.J∞ = copy(J)
    A∞ = zeros(n.Ω,n.Q)
    for t in 1:T
        a∞ = (p*x∞[T-t+1])[1,:,:]'
        @views J∞[:,t,:,T] .= x∞[T-t+1]*IntΦ̃+Iop*A∞#ILM[:,T-t+1,:]*a∞
        A∞ = L*A∞+M*a∞
    end

    ITT = sparse(I,T,T)
    ITT_ = spdiagm(-1=>ones(T-1))
    ITTᵉ = spdiagm(1=>ones(T-1))
    ITTᵉ[T,T] = 1 #last

    BBσσ = kron(ITT,dG.x)*reshape(J∞,n.x*T,:)*kron(ITT,Q) .+ kron(ITT,dG.X) .+ kron(ITT_,dG.X_*P) .+ kron(ITTᵉ,dG.Xᵉ) 
    
    EX_ΘΘᵉ = sum(X̂_ΘΘ[i][:,1]*Σ_Θ[i] for i in 1:length(Σ_Θ)) 

    AAσσ = -dG.x*(xσσ*IntΦ̃ .+ SO.IBσσ) .+ (dG.Xᵉ*EX_ΘΘᵉ)[:]

    SO.X̂_σσ = reshape(BBσσ\AAσσ[:],n.X,:)
end


function compute_x̄σσ!(SO::SecondOrderApproximation)
    @unpack FO,xσσ,X̂_σσ = SO
    @unpack ZO,T = FO
    @unpack p,Q,n = ZO
    x̂_σσ = SO.x̂_σσ =Vector{Matrix{Float64}}(undef,T)
    #    x = permutedims(FO.x,[1,3,2])
    flipdims = xs->permutedims(xs,[1,3,2])
    x = flipdims.(FO.x)
    x∞ = flipdims.(SO.x∞)
    QX_σσ = Q*X̂_σσ
    for t in 1:T
        x̂_σσ[t] = copy(xσσ)
        for s in 1:T-t
            @views x̂_σσ[t] += x[s]*QX_σσ[:,t+s-1]
        end
        @views x̂_σσ[t] += x∞[T-t+1]*QX_σσ[:,T]
    end
end


"""
    compute_lemma3_SV!(SO)

Computes the xσσΥ terms from Lemma 3 SV 
"""
function compute_lemma3_SV!(SO::SecondOrderApproximation)
    @unpack FO,T,x̂_ΘΘ,x∞ = SO
    @unpack ZO,f,x̂_Θt = FO
    @unpack Φ̃,Φ̃ᵉ,Σ_ΘΥ,ρΥ,dF,n = ZO
    B = spzeros(n.x*n.sp,n.x*n.sp)
    F̂_σσ = zeros(n.x,n.sp)
    fF_x′ = Vector{Matrix{Float64}}(undef,n.sp)

    Ex̄_ΘΘ′Υ = sum((x̂_ΘΘ[i][1]*Φ̃ᵉ)*Σ_ΘΥ[i] for i in 1:length(Σ_ΘΥ))
    for j in 1:n.sp

        fF_x′[j] = f[j]*dF.x′[j]
        B[1+(j-1)*n.x:j*n.x,:] = kron(sparse(Φ̃[:,j]'),Matrix(I,n.x,n.x)) - ρΥ*kron(sparse(Φ̃ᵉ[:,j]'),fF_x′[j])

        F̂_σσ[:,j] = fF_x′[j]*Ex̄_ΘΘ′Υ[:,j]
    end
    luB = lu(B)
    SO.xσσΥ = reshape(luB\F̂_σσ[:],n.x,:)
end


function compute_corollary2_SV(SO::SecondOrderApproximation)
    @unpack FO,xσσΥ = SO
    @unpack ZO,L,T,ILM = FO
    @unpack p,ω̄,Φ,Λ,ρΥ,n = ZO
    IntΦ̃ = Φ*ω̄
    
    ILMpxσσΥ = ILM*(p*xσσΥ)[:] #n.x×T
    IBΥ =  zeros(n.x,T)
    IBΥ[:,1] = xσσΥ*IntΦ̃
    for t in 2:T
        @views IBΥ[:,t] = ρΥ.*IBΥ[:,t-1] .+ ILMpxσσΥ[:,t-1] 
    end
    SO.IBΥ = IBΥ
end


function compute_proposition1_SV(SO::SecondOrderApproximation)
    @unpack FO,IBΥ,xσσΥ,X̂_ΘΘ = SO
    @unpack ZO,BB,T = FO 
    @unpack dG, n,Σ_ΘΥ,ρΥ = ZO
    ρΥt = ρΥ.^(0:T-1)
    EX̂_ΘΘᵉΥ = sum(X̂_ΘΘ[i][:,1]*Σ_ΘΥ[i] for i in 1:length(Σ_ΘΥ))
    AA = dG.x*IBΥ .+ (dG.Xᵉ*EX̂_ΘΘᵉΥ).*ρΥt'

    SO.X̂Υ = reshape(-BB\AA[:],n.X,T)
end


function compute_x̄σσΥ!(SO::SecondOrderApproximation)
    @unpack FO,xσσΥ,X̂Υ = SO
    @unpack ZO,T = FO
    @unpack p,Q,n = ZO
    x̂_σσΥ = SO.x̂_σσΥ =Vector{Matrix{Float64}}(undef,T)
    flipdims = xs->permutedims(xs,[1,3,2])
    x = flipdims.(FO.x)
    QX_σσΥ = Q*X̂Υ
    for t in 1:T
        x̂_σσΥ[t] = copy(xσσΥ)
        for s in 1:T-t+1
            @views x̂_σσΥ[t] += x[s]*QX_σσΥ[:,t+s-1]
        end
    end
end


"""
    computeSecondOrder!(SO)

Computes the second order derivatives with 
"""
function computeSecondOrder!(SO::SecondOrderApproximation)
    @unpack FO = SO
    @unpack ZO = FO
    @unpack n = ZO

    SO.X̂_ΘΘ = Matrix{Matrix{Float64}}(undef,n.Θ,n.Θ)
    SO.x̂_ΘΘ =  Matrix{Vector{Matrix{Float64}}}(undef,n.Θ,n.Θ)
    SO.Ixtkδ_ΘΘ = Matrix{Array{Float64,3}}(undef,n.Θ,n.Θ)
    ## initialization of aggregate state at 0
    #SO.X_02 = FO.X_0
    SO.X_02 = zeros(n.A) #initial predetermined for second derivative

    # initialization for compute_XZZjk
    SO.FxkΘ = Vector{Matrix{Matrix{Float64}}}(undef,n.Θ)
    SO.FXkΘ = Vector{Matrix{Matrix{Float64}}}(undef,n.Θ)
    SO.Fx′kΘ   = Vector{Matrix{Matrix{Float64}}}(undef,n.Θ)


    for i in 1:n.Θ
        # initialization direction of the first order shock
        FO.Θ_0 = I[1:n.Θ,i]
        FO.x̂t = FO.x̂_Θt[i]
        FO.X̂t = FO.X̂_Θt[i]
        FO.Ω̂t = FO.Ω̂_Θt[i]
        compute_lemma3_components!(SO)
        SO.FxkΘ[i]= SO.Fxk
        SO.FXkΘ[i]= SO.FXk
        SO.Fx′kΘ[i]= SO.Fx′k

        for j in 1:n.Θ
            #setup second direction
            SO.Θ_02 = I[1:n.Θ,j] #initial exogenous states
            SO.x̂k = FO.x̂_Θt[j]
            SO.X̂k = FO.X̂_Θt[j]
            SO.Ω̂k = FO.Ω̂_Θt[j]
            compute_lemma3_ZZ_kink!(SO)
            compute_Lemma2_ZZ!(SO)
            compute_lemma3_ZZ!(SO)
            compute_Lemma4_ZZ!(SO)
            construct_Laa!(SO)
            compute_Corollary2_ZZ!(SO)
            compute_XZZ!(SO)
            compute_xZZ0!(SO)
            #compute_xZZ!(SO) only need the initial period x_ZZ for σσ
            SO.X̂_ΘΘ[i,j] = SO.X̂tk
            SO.X̂_ΘΘ[j,i] = SO.X̂tk
            SO.x̂_ΘΘ[i,j] = SO.x̂tk
            SO.x̂_ΘΘ[j,i] = SO.x̂tk
            SO.Ixtkδ_ΘΘ[i,j] = SO.Ixtkδ
            SO.Ixtkδ_ΘΘ[j,i] = SO.Ixtkδ
        end 
    end 
    compute_lemma3_σσ!(SO)
    compute_corollary2_σσ!(SO)
    compute_proposition1_σσ!(SO)
end 


#TODO: rename based on k
function compute_XZZjk(SO::SecondOrderApproximation,iΘ,jΘ,Tjs)  #i.e. ks
    @unpack FO = SO
    @unpack ZO, T = FO
    @unpack P, n = ZO
    

    SO.Fxk = SO.FxkΘ[iΘ]
    SO.FXk = SO.FXkΘ[iΘ] 
    SO.Fx′k= SO.Fx′kΘ[iΘ]
    X̂tk_ijtemp = Vector{Matrix{Float64}}(undef,maximum(Tjs))
    Threads.@threads for j in Tjs
        SOtemp = copy(SO)
        SOtemp.x̂k = FO.x̂_Θt[jΘ][j+1:end]
        SOtemp.X̂k = FO.X̂_Θt[jΘ][:,j+1:end]
        SOtemp.X_02 = P*FO.X̂_Θt[jΘ][:,j]
        SOtemp.Θ_02 = ρ_Θ^j*I[1:n.Θ,jΘ]
        SOtemp.Ω̂k = FO.Ω̂_Θt[jΘ][:,j+1:end]
        compute_Lemma2_ZZ!(SOtemp)
        compute_lemma3_ZZ_kink!(SOtemp)
        compute_lemma3_ZZ!(SOtemp)
        compute_Lemma4_ZZ!(SOtemp)
        compute_Corollary2_ZZ!(SOtemp)
        compute_XZZ!(SOtemp)
        X̂tk_ijtemp[j] = SOtemp.X̂tk
    end 
    X̂tk_ij = Dict()
    for j in Tjs
        X̂tk_ij[j] = X̂tk_ijtemp[j]
    end
    return X̂tk_ij
end



"""
    getPath(SO,path,iX,X_ΘΘjk)

Computes the response to a path of shocks for both a first and
second order approximation.  Uses the interaction derivatives
X_ΘΘjk.
"""
function getPath(SO::SecondOrderApproximation,path,iX,X_ΘΘjk)
    FO,ZO = SO.FO,SO.FO.ZO
    Σ_Θ = ZO.Σ_Θ

    Tsim = length(path)
    Xt=zeros(Tsim)
    X_σσt=zeros(Tsim)
    X_σσt= SO.X̂_σσ[iX,:]
    Xzerotht=ZO.X̄[iX]*ones(Tsim)
    Xfirstordert=zeros(Tsim)
    Xinteractiont=zeros(Tsim)
    Xsigmasigmat=X_σσt[1:Tsim]

    for t in 1:Tsim
        fo=0
        so=0
        # Compute the first order response 
        for s=1:t
            fo += path[s]*FO.X̂t[iX,t-s+1]
        end

        # Compute the second order response 
        for s=1:t
            for m=1:t
                lag=max(s,m)-min(s,m)
                so += path[s]*path[m]*X_ΘΘjk[lag][iX,t-max(s,m)+1]
            end
        end

        Xfirstordert[t] = fo*Σ_Θ[1,1]^0.5
        Xinteractiont[t] = so*Σ_Θ[1,1]
    end

    # Save the IRF decomposition in a DataFrame 
    Xt=DataFrame()
    Xt.zeroth=Xzerotht
    Xt.firstorder=Xfirstordert
    Xt.interaction=Xinteractiont
    Xt.risk=Xsigmasigmat
    Xt.path=path
    Xt.X=Xzerotht+Xfirstordert+0.5*Xinteractiont+0.5*Xsigmasigmat
    Xt.FO=Xzerotht+Xfirstordert
    return Xt
end
