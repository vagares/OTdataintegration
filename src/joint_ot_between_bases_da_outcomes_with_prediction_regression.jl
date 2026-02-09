function joint_ot_between_bases_da_outcomes_with_predictors_regression(
        data;
        iterations = 10,
        learning_rate = 0.01,
        batchsize = 512,
        epochs = 500,
        hidden_layer_size = 10,
        reg = 0.0,
        reg_m1 = 0.0,
        reg_m2 = 0.0
    )

    T = Int32

    dba = subset(data, :database => ByRow(==(1)))
    dbb = subset(data, :database => ByRow(==(2)))

    cols = names(dba, r"^X")

    XA = transpose(Matrix{Float32}(dba[:, cols]))
    XB = transpose(Matrix{Float32}(dbb[:, cols]))
    YA = dba.V
    ZB = dbb.W
    XYA = vcat(XA, YA)
    XZB = vcat(XB, ZB)

    nA = size(dba, 1)
    nB = size(dbb, 1)

    wa = ones(Float32, nA) ./ nA
    wb = ones(Float32, nB) ./ nB

    C0 = Float32.(pairwise(SqEuclidean(), XA, XB, dims = 2))
    C = C0 ./ maximum(C0)

    dimXA = size(XA, 1)
    dimXB = size(XB, 1)
    dimYA = 1
    dimZB = 1

    modelXYA = Chain(Dense(dimXA, hidden_layer_size), Dense(hidden_layer_size, dimYA))
    modelXZB = Chain(Dense(dimXB, hidden_layer_size), Dense(hidden_layer_size, dimZB))

    function train!(model, x, y)

        loader = Flux.DataLoader((x, y), batchsize = batchsize, shuffle = true)
        optim = Flux.setup(Flux.Adam(learning_rate), model)

        for epoch in 1:epochs
            for (x, y) in loader
                grads = Flux.gradient(model) do m
                    y_hat = m(x)
                    Flux.mse(y_hat, y)
                end
                Flux.update!(optim, model, grads[1])
            end
        end

        return
    end

    function loss_mse(Y, F)
        return mean((Y .- F).^2)
    end

    YBpred = modelXYA(XB)
    ZApred = modelXZB(XA)

    G = ones(Float32, length(wa), length(wb))

    if reg > 0
        G .= PythonOT.mm_unbalanced(wa, wb, C, (reg_m1, reg_m2); reg = reg, div = "kl")
    else
        G .= PythonOT.emd(wa, wb, C)
    end

    YBt = similar(YBpred)
    ZAt = similar(ZApred)

    YBt .= nB .* YA * G
    ZAt .= nA .* ZB * G'

    train!(modelXYA, XB, YBt)
    train!(modelXZB, XA, ZAt)

    YBpred .= modelXYA(XB)
    ZApred .= modelXZB(XA)

    return YBpred, ZApred

end
