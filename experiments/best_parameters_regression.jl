using OptimalTransportDataIntegration
using DelimitedFiles

function best_parameters_continuous(start, stop)

    alpha = collect(0:0.1:2)
    lambda = collect(0:0.1:1)
    reg = [0.001, 0.01, 0.1, 1.0]
    reg_m = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75]
    params = DataParameters()
    discrete = false

    outfile = "best_parameters_continuous.csv"
    header = [
        "id", "alpha", "lambda", "est_v", "est_wa", "est", "method", "scenario", "discrete",
        "reg", "reg_m",
    ]

    return open(outfile, "w") do io

        seekstart(io)
        writedlm(io, hcat(header...))

        for scenario in 1:2

            rng = ContinuousDataGenerator(params, scenario = scenario)

            for i in start:stop

                data = generate(rng)


                for r in reg, r_m in reg_m

                    result = otrecod(data, JointOTBetweenBasesWithPredictors_regression(reg = r, reg_m1 = r_m, reg_m2 = r_m))
                    est_vb, est_wa, est = accuracy(result)
                    writedlm(io, [i missing missing est_vb est_wa est "between" scenario discrete r r_m])

                end

            end

        end

    end

end

best_parameters_continuous(1, 100)

#import Statistics: mean
#data = CSV.read("best_parameters.csv", DataFrame)
#sort(
#    combine(groupby(data, ["alpha", "lambda"]), :estimation => mean),
#    order(:estimation_mean, rev = true),
#)
# equivalent with pandas
# import pandas as pd
# data = pd.read_csv("results_otjoint.csv", sep="\t")
# data.groupby(["alpha", "lambda"]).estimation.mean().sort_values(ascending=False)
