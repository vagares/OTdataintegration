import JSON
export DataParameters
export read_params
export save_params

@with_kw struct DataParameters

    nA::Int = 1000
    nB::Int = 1000
    mA::Vector{Float64} = [0.0, 0.0, 0.0]
    mB::Vector{Float64} = [0.0, 1.0, 0.0]
    covA::Matrix{Float64} = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    covB::Matrix{Float64} = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    aA::Vector{Float64} = [1.0, 1.0, 1.0]
    aB::Vector{Float64} = [1.0, 1.0, 1.0]
    r2::Float64 = 0.6
    T::String = "Id"
    hw::String = "ln"
    theta::Float64 = pi
end

"""
$(SIGNATURES)

Read the data generation scenario from a JSON file
"""
function read(jsonfile::AbstractString)

    data = JSON.parsefile(jsonfile)

    nA = Int(data["nA"])
    nB = Int(data["nB"])

    aA = Int.(data["aA"])
    aB = Int.(data["aB"])

    mA = vec(Int.(data["mA"]))
    mB = vec(Int.(data["mB"]))

    covA = stack([Float64.(x) for x in data["covA"]])
    covB = stack([Float64.(x) for x in data["covB"]])

    r2 = Float64(data["r2"])

    return DataParameters(nA, nB, mA, mB, covA, covB, aA, aB, r2,T,hw,theta)

end

"""
$(SIGNATURES)

Write the data generation scenario to a JSON file
"""
function save(jsonfile::AbstractString, params::DataParameters)

    data = Dict(
        fieldnames(DataParameters) .=>
            getfield.(Ref(params), fieldnames(DataParameters)),
    )

    return open(jsonfile, "w") do io
        JSON.print(io, data)
    end

end
