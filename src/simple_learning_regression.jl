import Flux
import Flux: Chain, Dense



function simple_learning_regression(
        data;
        hidden_layer_size = 10,
        learning_rate = 0.01,
        batchsize = 512,
        epochs = 1000
         )

    dba = subset(data, :database => ByRow(==(1)))
    dbb = subset(data, :database => ByRow(==(2)))

    colnames = names(data, r"^X")
    XA = onehot(Matrix(dba[!, colnames]))
    XB = onehot(Matrix(dbb[!, colnames]))
    YA = dba.V
    ZB = dbb.W
    dimXA = size(XA, 1)
    dimXB = size(XB, 1)
    dimYA = 1
    dimZB = 1

    modelA = Chain(Dense(dimXA, hidden_layer_size), Dense(hidden_layer_size, dimYA))
    modelB = Chain(Dense(dimXB, hidden_layer_size), Dense(hidden_layer_size, dimZB))

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

    train!(modelA, XA, YA)
    train!(modelB, XB, ZB)

    YB = modelA(XB)
    ZA = modelB(XA)

    return YB, ZA

end
