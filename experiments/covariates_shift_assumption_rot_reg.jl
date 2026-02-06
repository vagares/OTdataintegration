using DelimitedFiles
using OptimalTransportDataIntegration

function covariates_shift_assumption_continuous(nsimulations::Int)

    outfile = "covariates_shift_assumption_continuous.csv"
    header = ["id", "mB", "estvb", "estwa", "est", "method", "scenario"]
    theta = (pi/4, pi/3, pi/2, pi)

    return open(outfile, "w") do io

        writedlm(io, hcat(header...))
        methods = Dict{String, AbstractMethod}(
            "sl" => SimpleLearning(),
            "wi-r" => JointOTWithinBase(),
            "be" => JointOTBetweenBasesWithPredictors(reg = 0.0),
            "be-x" => JointOTBetweenBasesWithoutOutcomes(reg = 0.0),
            "jdot" => JointOTBetweenBasesJDOT(reg = 0.0),
            "otda-x" => JointOTDABetweenBasesCovariables(reg = 0.0),
            "otda-yz" => JointOTDABetweenBasesOutcomes(reg = 0.0),
            "otda-yz-f" => JointOTDABetweenBasesOutcomesWithPredictors(reg = 0.0)
        )
        for theta in shift

            params = DataParameters(theta = theta,T="rot")

            for scenario in 1:2

                rng = ContinuousDataGenerator(params, scenario = scenario)

                for i in 1:nsimulations

                    data = generate(rng)

                    alpha, lambda = 0.0, 0.0
                    result = otrecod(data, JointOTWithinBase(alpha = alpha, lambda = lambda))
                    estvb, estwa, est = accuracy(result)
                    writedlm(io, [i repr(theta) estvb estwa est "wi" scenario])

                    alpha, lambda = best_parameters(:within, :continuous, scenario)
                    result = otrecod(data, JointOTWithinBase(alpha = alpha, lambda = lambda))
                    estvb, estwa, est = accuracy(result)
                    writedlm(io, [i repr(theta) estvb estwa est "wi-r" scenario])

                    result = otrecod(data, JointOTBetweenBasesWithPredictors_regression(reg = 0.0))
                    estvb, estwa, est = accuracy(result)
                    writedlm(io, [i repr(theta) estvb estwa est "be" scenario])

                    reg, reg_m = best_parameters(:between, :continuous, scenario)
                    result = otrecod(data, JointOTBetweenBasesWithPredictors_regression(reg = reg, reg_m1 = reg_m, reg_m2 = reg_m))
                    estvb, estwa, est = accuracy(result)
                    writedlm(io, [i repr(theta) estvb estwa est "be-un-r" scenario])

                    result = otrecod(data, JointOTBetweenBasesJDOT_regression(reg = 0.0))
                    estvb, estwa, est = accuracy(result)
                    writedlm(io, [i repr(theta) estvb estwa est "jdot" scenario])
      

                    result = otrecod(data,  JointOTDABetweenBasesOutcomesWithPredictors_regression(reg = 0.0))
                    estvb, estwa, est = accuracy(result)
                    writedlm(io, [i repr(theta) estvb estwa est "otda-yz-f" scenario])

                    result = otrecod(data, SimpleLearning_regression())
                    estvb, estwa, est = accuracy(result)
                    writedlm(io, [i repr(theta) estvb estwa est "sl" scenario])

                end

            end

        end

    end

end

