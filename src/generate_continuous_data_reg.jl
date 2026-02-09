export ContinuousDataGeneratorreg, generatereg

struct ContinuousDataGeneratorreg

    params::DataParameters
    binsYA::Vector{Float64}
    binsZA::Vector{Float64}
    binsYB::Vector{Float64}
    binsZB::Vector{Float64}

    function ContinuousDataGeneratorreg(params; scenario = 1, n = 10000)

        XA = rand(MvNormal(params.mA, params.covA), params.nA)
        XB = rand(MvNormal(params.mB, params.covB), params.nB)
        XB2 = rand(MvNormal(params.mB, params.covB), params.nA)

        X1 = XA
        X2 = XB
        X22=XB2

        qA = size(XA, 1)
        qB = size(XB, 1)

        aA = params.aA[1:qA]
        aB = params.aB[1:qB]

        cr2 = 1 / params.r2 - 1
        mA = params.mA
        mB = params.mB
        covA = params.covA
        covB = params.covB

        varerrorA =
            cr2 *
            sum([aA[i] * aA[j] * covA[i, j] for i in axes(covA, 1), j in axes(covA, 2)])
        varerrorB =
            cr2 *
            sum([aB[i] * aB[j] * covB[i, j] for i in axes(covB, 1), j in axes(covB, 2)])



        if varerrorA == 0
            U1 = X1' * aA
            V1 = X1' * aB
        else
            U1 = X1' * aA .+ rand(Normal(0.0, sqrt(varerrorA)), params.nA)
            V1 = X1' * aB .+ rand(Normal(0.0, sqrt(varerrorA)), params.nA)
        end
        if params.T == "homo"
            M = x -> sqrt(covA) * inv(sqrt(covB)) * (x .- mB) .+ mA

        elseif params.T == "Rotation"
            R = [
                cos(params.theta)  -sin(params.theta)   0.0
                sin(params.theta)   cos(params.theta)   0.0
                0.0      0.0      1.0
                ]
             M = x -> R * x
        elseif params.T=="Id"
             M = x -> x
        end
        X2M = hcat([M(X22[:, i]) for i in 1:size(X22, 2)]...)

        if varerrorB == 0
            U2 = X2' * aA
            U2T = X2M' * aA
            V2 = X2' * aB
            V2T = X2M' * aB
        else
            U2 = X2' * aA .+ rand(Normal(0.0, sqrt(varerrorB)), params.nB)
            U2T = X2M' * aA .+ rand(Normal(0.0, sqrt(varerrorB)), params.nB)
            V2 = X2' * aB .+ rand(Normal(0.0, sqrt(varerrorB)), params.nB)
            V2T = X2M' * aB .+ rand(Normal(0.0, sqrt(varerrorB)), params.nB)
        end


        bYA = quantile(U1, [0.25, 0.5, 0.75])
        bYB = quantile(U2, [0.25, 0.5, 0.75])
        bYB2 = quantile(U2T, [0.25, 0.5, 0.75])
        bZA = quantile(V1, [1 / 3, 2 / 3])
        bZB = quantile(V2, [1 / 3, 2 / 3])
        bZB2 = quantile(V2T, [1 / 3, 2 / 3])

        if scenario == 1
            binsYA = vcat(-Inf, bYA, Inf)
            binsZA = vcat(-Inf, bZA, Inf)
            binsYB = copy(binsYA)
            binsZB = copy(binsZA)
        else
            binsYA = vcat(-Inf, bYA, Inf)
            binsZA = vcat(-Inf, bZA, Inf)
            binsYB = vcat(-Inf, bYB2, Inf)
            binsZB = vcat(-Inf, bZB2, Inf)
        end

        return new(
            params,
            binsYA,
            binsZA,
            binsYB,
            binsZB,
        )


    end

end


"""
$(SIGNATURES)

Function to generate data where covariable X is **continuous** and (Y,Z) are categoricals

the function return a Dataframe with X1, X2, X3, Y, Z and the database id.

r2 is the coefficient of determination 

"""
function generatereg(generator::ContinuousDataGeneratorreg; eps = 0.0)

    params = generator.params

    XA = rand(MvNormal(params.mA, params.covA), params.nA)
    XB = rand(MvNormal(params.mB, params.covB), params.nB)
    XB2 = rand(MvNormal(params.mB, params.covB), params.nA)

    X1 = XA
    X2 = XB
    X22=XB2

    cr2 = 1.0 / params.r2 - 1

    qA = size(XA, 1)
    qB = size(XB, 1)

    aA = params.aA[1:qA]
    aB = params.aB[1:qB]

    covA = params.covA
    covB = params.covB

    varerrorA =
        cr2 *
        sum([aA[i] * aA[j] * covA[i, j] for i in axes(covA, 1), j in axes(covA, 2)])
    varerrorB =
        cr2 *
        sum([aB[i] * aB[j] * covB[i, j] for i in axes(covB, 1), j in axes(covB, 2)])


    if varerrorA == 0
        U1 = X1' * aA
        V1 = X1' * aB
    else
        U1 = X1' * aA .+ rand(Normal(0.0, sqrt(varerrorA)), params.nA)
        V1 = X1' * aB .+ rand(Normal(0.0, sqrt(varerrorA)), params.nA)
    end

    if params.hW=="ln"
        Y1hW = log(U1)
    elseif params.hW=="Id"
        Y1hW = U1
    end

    if params.hS=="ln"
        Y1hS = log(V1)
    elseif params.hS=="Id"
        Y1hS = V1
    end

    if params.hS=="ln"
        V1hS = log(V1)
    elseif params.hS=="Id"
        V1hS = V1
    end
                
    if params.T == "homo"
            M = x -> sqrt(covA) * inv(sqrt(covB)) * (x .- mB) .+ mA

    elseif params.T == "Rotation"
            R = [
                cos(params.theta)  -sin(params.theta)   0.0
                sin(params.theta)   cos(params.theta)   0.0
                0.0      0.0      1.0
                ]
             M = x -> R * x
    elseif params.T=="Id"
             M = x -> x
    end
    
    X2M = hcat([M(X22[:, i]) for i in 1:size(X22, 2)]...)

    if varerrorB == 0
        U2 = X2' * aA
        U2T = X2M' * aA
        V2 = X2' * aB
        V2T = X2M' * aB
    else
        U2 = X2' * aA .+ rand(Normal(0.0, sqrt(varerrorB)), params.nB)
        U2T = X2M' * aA .+ rand(Normal(0.0, sqrt(varerrorB)), params.nB)
        V2 = X2' * aB .+ rand(Normal(0.0, sqrt(varerrorB)), params.nB)
        V2T = X2M' * aB .+ rand(Normal(0.0, sqrt(varerrorB)), params.nB)
    end

    if params.hW=="ln"
        Y2hW = log(U2T)
    elseif params.hW=="Id"
        Y2hW = U2T
    end

    if params.hS=="ln"
        Y2hS = log(V2T)
    elseif params.hS=="Id"
        Y2hS = V2T
    end

    YA = digitize(U1, generator.binsYA)
    ZA = digitize(V1, generator.binsZA)
    VA = Y1hW
    WA = Y1hS
    YB = digitize(U2T, generator.binsYB .+ eps)
    ZB = digitize(V2T, generator.binsZB .+ eps)
    VB = Y2hW
    WB = Y2hS
    p = length(aA)
    colnames = Symbol.("X" .* string.(1:p))
    df = DataFrame(vcat(X1', X2'), colnames)
    #df.Y = vcat(YA, YB)
    #df.Z = vcat(ZA, ZB)
    df.V = vcat(VA, VB)
    df.W = vcat(WA, WB)
    df.database = vcat(fill(1, params.nA), fill(2, params.nB))

    return df

end


