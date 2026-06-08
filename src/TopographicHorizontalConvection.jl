module TopographicHorizontalConvection
    include("simulation.jl")
    #include("simulation_test.jl")
    include("SIM_GPU_TEST.jl")
    include("analysis.jl")
    include("grid_testing.jl")
    include("RANS_energies.jl")
end

#print the julia version in the scripts so we know which version is loaded

#start from the one that is working and make it more complex?

#ask for an interactive gpu node line by line do everything in the batch script (except module load julia)
#then line by line run your code 