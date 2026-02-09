using DelimitedFiles
using OptimalTransportDataIntegration

function covariates_shift_assumption_continuousreg(nsimulations::Int)

    outfile = "covariates_shift_assumption_continuousreg.csv"
    header = ["id", "mB", "estvb", "estwa", "est", "method", "scenario"]
    shift = ([1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0], [0.5 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 2.0], [0.5 0.2 0.2; 0.2 1.0 0.2; 0.2 0.2 2.0], 
                [0.5 0.1 0.2; 0.1 1.0 0.4; 0.2 0.4 2.0])

    return open(outfile, "w") do io

        writedlm(io, hcat(header...))

        for covB in shift

            params = DataParameters(covB = covB)

            for scenario in 1:2

                rng = ContinuousDataGeneratorreg(params, scenario = scenario)

                for i in 1:nsimulations

                    data = generate(rng)


                    result = otrecod(data, JointOTBetweenBasesWithPredictors_regression(reg = 0.0))
                    estvb, estwa, est = accuracy(result)
                    writedlm(io, [i repr(mB) estvb estwa est "be" scenario])

                    reg, reg_m = best_parameters(:between, :continuous, scenario)
                    result = otrecod(data, JointOTBetweenBasesWithPredictors_regression(reg = reg, reg_m1 = reg_m, reg_m2 = reg_m))
                    estvb, estwa, est = accuracy(result)
                    writedlm(io, [i repr(covB) estvb estwa est "be-un-r" scenario])


                    result = otrecod(data, JointOTBetweenBasesJDOT_regression(reg = 0.0))
                    estvb, estwa, est = accuracy(result)
                    writedlm(io, [i repr(covB) estvb estwa est "jdot" scenario])

                    result = otrecod(data,  JointOTDABetweenBasesOutcomesWithPredictors_regression(reg = 0.0))
                    estvb, estwa, est = accuracy(result)
                    writedlm(io, [i repr(covB) estvb estwa est "otda-yz-f" scenario])

                    result = otrecod(data, SimpleLearning_regression())
                    estvb, estwa, est = accuracy(result)
                    writedlm(io, [i repr(covB) estvb estwa est "sl" scenario])

                end

            end

        end

    end

end

