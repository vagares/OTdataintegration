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
    hW::String = "Id"
    hS::String = "Id"
    theta::Float64 = 3.14
end

"""
$(SIGNATURES)

Read the data generation scenario from a JSON file
"""
function read(jsonfile::AbstractString)

    data = JSON.parsefile(jsonfile)

    nA = Int(data["nA"])
    nB = Int(data["nB"])

    aA = Float64.(data["aA"])
    aB = Float64.(data["aB"])

    mA = vec(Float64.(data["mA"]))
    mB = vec(Float64.(data["mB"]))
    covA = stack([Float64.(x) for x in data["covA"]])
    covB = stack([Float64.(x) for x in data["covB"]])

    r2 = Float64(data["r2"])

    T = String(data["T"])
    hW = String(data["hW"])
    hS = String(data["hS"])
    theta = Float64(data["theta"])
    
    return DataParameters(nA, nB, mA, mB, covA, covB, aA, aB, r2,T,hW,hS,theta)

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
