export ContinuousDataGenerator

struct ContinuousDataGenerator

    params::DataParameters
    binsYA::Vector{Float64}
    binsZA::Vector{Float64}
    binsYB::Vector{Float64}
    binsZB::Vector{Float64}

    function ContinuousDataGenerator(params; scenario = 1, n = 10000,T="homo",hW="ln",theta=π)

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
            Base1 = X1' * aA
        else
            Base1 = X1' * aA .+ rand(Normal(0.0, sqrt(varerrorA)), params.nA)
        end

        if T == "homo"
            M = x -> sqrt(covA) * inv(sqrt(covB)) * (x .- mB) .+ mA

        elseif T == "Rotation"
            R = [
                cos(θ)  -sin(θ)   0.0
                sin(θ)   cos(θ)   0.0
                0.0      0.0      1.0
                ]
             M = x -> R * x
        end
        X2M = hcat([M(X22[:, i]) for i in 1:size(X22, 2)]...)

        if varerrorB == 0
            Base2 = X2' * aB
            Base2T = X2M' * aB 
        else
            Base2 = X2' * aB .+ rand(Normal(0.0, sqrt(varerrorB)), params.nB)
            Base2T = X2M' * aB .+ rand(Normal(0.0, sqrt(varerrorB)), params.nB)
        end

        bYA = quantile(Base1, [0.25, 0.5, 0.75])
        bYB = quantile(Base2, [0.25, 0.5, 0.75])
        bYB2 = quantile(Base2T, [0.25, 0.5, 0.75])
        bZA = quantile(Base1, [1 / 3, 2 / 3])
        bZB = quantile(Base2, [1 / 3, 2 / 3])
        bZB2 = quantile(Base2T, [1 / 3, 2 / 3])

        if scenario == 1
            binsYA = vcat(-Inf, bYA, Inf)
            binsZA = vcat(-Inf, bZA, Inf)
            binsYB = copy(binsYA)
            binsZB = copy(binsZA)
        else
            binsYA = vcat(-Inf, bYA, Inf)
            binsZA = vcat(-Inf, bZA2, Inf)
            binsYB = vcat(-Inf, bYB, Inf)
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
function generate(generator::ContinuousDataGenerator; eps = 0.0)

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
        Y1 = X1' * aA
    else
        Y1 = X1' * aA .+ rand(Normal(0.0, sqrt(varerrorA)), params.nA)
    end

    if hW="ln" 
            Y1hW = log(Y1)
                
    if T == "homo"
            M = x -> sqrt(covA) * inv(sqrt(covB)) * (x .- mB) .+ mA

    elseif T == "Rotation"
            R = [
                cos(θ)  -sin(θ)   0.0
                sin(θ)   cos(θ)   0.0
                0.0      0.0      1.0
                ]
             M = x -> R * x
    elsif T=="Id"
             M = x -> x
    end
        X2M = hcat([M(X22[:, i]) for i in 1:size(X22, 2)]...)

        X2M = np.array([M(x) for x in X2])

    if varerrorB == 0
        Y2 = X2' * aB
        Y2T = X2M' * aB
    else
        Y2 = X2' * aB .+ rand(Normal(0.0, sqrt(varerrorB)), params.nB)
        Y2T = X2M' * aB .+ rand(Normal(0.0, sqrt(varerrorB)), params.nB)
    end

    if hW="ln" 
            Y2ThW = log(Y2T)
            Y2hW = log(Y2)

    YA = digitize(Y1, generator.binsYA)
    ZA = digitize(Y1, generator.binsZA)
    VA = Y1
    WA = Y1hW
    YB = digitize(Y2, generator.binsYB .+ eps)
    ZB = digitize(Y2, generator.binsZB .+ eps)
    VB = Y2
    WB = Y2hW
    p = length(aA)
    colnames = Symbol.("X" .* string.(1:p))
    df = DataFrame(vcat(X1', X2'), colnames)
    df.Y = vcat(YA, YB)
    df.Z = vcat(ZA, ZB)
    df.V = vcat(VA, VB)
    df.W = vcat(WA, WB)
    df.database = vcat(fill(1, params.nA), fill(2, params.nB))

    @info "Categories in YA $(sort!(OrderedDict(countmap(YA))))"
    @info "Categories in ZA $(sort!(OrderedDict(countmap(ZA))))"
    @info "Categories in YB $(sort!(OrderedDict(countmap(YB))))"
    @info "Categories in ZB $(sort!(OrderedDict(countmap(ZB))))"

    return df

end
