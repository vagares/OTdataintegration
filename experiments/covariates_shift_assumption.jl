using DelimitedFiles
using OptimalTransportDataIntegration

function covariates_shift_assumption_continuous(nsimulations::Int)

    outfile = "covariates_shift_assumption_continuous.csv"
    header = ["id", "mB", "estyb", "estza", "est", "method", "scenario"]
    shift = ([0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1])

    return open(outfile, "w") do io

        writedlm(io, hcat(header...))

        for mB in shift

            params = DataParameters(mB = mB)

            for scenario in 1:2

                rng = ContinuousDataGenerator(params, scenario = scenario)

                for i in 1:nsimulations

                    data = generate(rng)

                    alpha, lambda = 0.0, 0.0
                    result = otrecod(data, JointOTWithinBase(alpha = alpha, lambda = lambda))
                    estyb, estza, est = accuracy(result)
                    writedlm(io, [i repr(mB) estyb estza est "wi" scenario])

                    alpha, lambda = best_parameters(:within, :continuous, scenario)
                    result = otrecod(data, JointOTWithinBase(alpha = alpha, lambda = lambda))
                    estyb, estza, est = accuracy(result)
                    writedlm(io, [i repr(mB) estyb estza est "wi-r" scenario])

                    result = otrecod(data, JointOTBetweenBasesWithPredictors(reg = 0.0))
                    estyb, estza, est = accuracy(result)
                    writedlm(io, [i repr(mB) estyb estza est "be" scenario])

                    reg, reg_m = best_parameters(:between, :continuous, scenario)
                    result = otrecod(data, JointOTBetweenBasesWithPredictors(reg = reg, reg_m1 = reg_m, reg_m2 = reg_m))
                    estyb, estza, est = accuracy(result)
                    writedlm(io, [i repr(mB) estyb estza est "be-un-r" scenario])

                    result = otrecod(data, JointOTBetweenBasesWithoutOutcomes(reg = 0.0))
                    estyb, estza, est = accuracy(result)
                    writedlm(io, [i repr(mB) estyb estza est "be-x" scenario])

                    result = otrecod(data, JointOTBetweenBasesJDOT(reg = 0.0))
                    estyb, estza, est = accuracy(result)
                    writedlm(io, [i repr(mB) estyb estza est "jdot" scenario])

                    result = otrecod(data, JointOTDABetweenBasesCovariables(reg = 0.0))
                    estyb, estza, est = accuracy(result)
                    writedlm(io, [i repr(mB) estyb estza est "otda-x" scenario])

                    result = otrecod(data,  JointOTDABetweenBasesOutcomes(reg = 0.0))
                    estyb, estza, est = accuracy(result)
                    writedlm(io, [i repr(mB) estyb estza est "otda-yz" scenario])

                    result = otrecod(data,  JointOTDABetweenBasesOutcomesWithPredictors(reg = 0.0))
                    estyb, estza, est = accuracy(result)
                    writedlm(io, [i repr(mB) estyb estza est "otda-yz-f" scenario])

                    result = otrecod(data, SimpleLearning())
                    estyb, estza, est = accuracy(result)
                    writedlm(io, [i repr(mB) estyb estza est "sl" scenario])

                end

            end

        end

    end

end

