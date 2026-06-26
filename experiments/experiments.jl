# Environment setup
using Pkg
Pkg.activate(".")
Pkg.instantiate()

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

using OpenML
using MLDatasets
using DataFrames
using CSV
using Statistics
using MLUtils
using ProgressMeter
using StatsBase
using Tables
using CategoricalArrays
using LinearAlgebra
using Random
using SparseArrays
using TOML
using ArgParse
using Logging

# Load the local package (declared as a path dependency in this environment's
# Project.toml). Using it as a package resolves its own dependencies (e.g.
# ChunkSplitters); a bare `include` of the source would not.
using FlyClassifiers

const K_FOLDS = 5

# Struct to hold all hyperparameters
struct Hyperparameters
    exp_values::Vector{Int}
    nnz_values::Vector{Float64}
    gamma::Float64
    s_ratio::Float64
    normalize::Bool
end

# Define configurations for each experiment type
# const OPENML_CONFIG = Hyperparameters([2^i for i in 8:8], [2.0^-i for i in 3:8], 0.9, 0.1)
# const VISION_CONFIG = Hyperparameters([2^i for i in 0:7], [2.0^i for i in 0:12], 0.9, 0.1)

println("Julia environment for experiments configured successfully.")

"""
Performs a single cross-validation run for a given model and configuration.
"""
function cross_validation(model_type, proj_type, X, y, m, k, s, γ; normalize=false)
    d = size(X, 1)
    accuracies = Float64[]
    
    folds = kfolds(shuffleobs((X, y)), k=K_FOLDS)

    for (i, (train_data, test_data)) in enumerate(folds)
        X_train, y_train = train_data
        X_test, y_test = test_data
        
        if normalize
            dt = StatsBase.fit(ZScoreTransform, X_train; dims=2)
            dt.scale .+= eps()

            X_train = StatsBase.transform(dt, X_train)
            replace!(X_train, NaN => 0)

            X_test = StatsBase.transform(dt, X_test)
            replace!(X_test, NaN => 0)
            println("norm ok")
        end
        

        P = if proj_type == RandomBinaryProjectionMatrix
            RandomBinaryProjectionMatrix(m, d, s; seed=42 + i)
        else # RandomUniformProjectionMatrix
            RandomUniformProjectionMatrix(m, d; seed=42 + i)
        end
        println("proj ok")
        println(model_type, proj_type)
        
        model = if model_type == FlyNNM
            FlyClassifiers.fit(FlyNNM, X_train, y_train, P, k, γ)
        else # EaS
            FlyClassifiers.fit(FlyNNA, X_train, y_train, P, k)
        end
        println("training ok")

        y_pred = FlyClassifiers.predict(model, X_test)
        accuracy = Statistics.mean(y_pred .== y_test)
        println("pred ok")
        push!(accuracies, accuracy)
        println("fold ok")
    end

    return round(Statistics.mean(accuracies); digits=4)
end

"""
Runs the suite of four model configurations for a given dataset.
"""
function run_model_suite(X, y, m, k, s, gamma; normalize=false)
    acc_flynn_binary = cross_validation(FlyNNM, RandomBinaryProjectionMatrix, X, y, m, k, s, gamma; normalize=normalize)
    println("ok FMB")
    acc_flynn_uniform = cross_validation(FlyNNM, RandomUniformProjectionMatrix, X, y, m, k, s, gamma; normalize=normalize)
    println("ok FMU")
    acc_eas_binary = cross_validation(FlyNNA, RandomBinaryProjectionMatrix, X, y, m, k, s, 0.0; normalize=normalize)
    println("ok FAB")
    acc_eas_uniform = cross_validation(FlyNNA, RandomUniformProjectionMatrix, X, y, m, k, s, 0.0; normalize=normalize)
    println("ok FAU")
    return (acc_flynn_binary, acc_flynn_uniform, acc_eas_binary, acc_eas_uniform)
end


"""
Runs experiments on a collection of datasets from OpenML.
"""
function openml_datasets_experiments(results_dir::String, config::Hyperparameters)
    #dataset_ids = [14, 16, 22, 28, 44, 60, 182, 715, 718, 722, 723, 734, 752, 761,
    #               797, 806, 833, 837, 849, 866, 903, 904, 917, 971, 979, 980, 995, 1020,
    #               1049, 1050, 1067, 1068, 1444, 1453, 1466, 1475, 1479, 1487, 1494, 1496,
    #               1497, 1504, 1507, 1547, 1560, 1566, 4538, 40497, 40499, 40900, 40982,
    #               41146, 41156, 41671, 41946, 12, 300, 978, 1038, 1041, 1042, 1468, 1485,
    #               1501, 40666, 40910, 40979, 41082, 41144, 41145, 41158]
    dataset_ids = [40497, 41156, 41946, 1041, 1042, 1468]
    println("Found $(length(dataset_ids)) datasets to process from the OpenML list.")

    generate_showvalues(name, nr, n, m, k) = () -> [("Dataset", "$(nr) - $(name)"), ("n", n), ("m", m), ("k",k)]

    for exp in config.exp_values
        results_for_exp = DataFrame(
            DatasetName=String[],
            FlyNNM_Binary=Float64[], FlyNNM_Uniform=Float64[],
            FlyNNA_Binary=Float64[], FlyNNA_Uniform=Float64[]
        )

        println("\nStarting processing for m = $(exp)d. Number of datasets: $(length(dataset_ids)).")
        progress = Progress(length(dataset_ids)*length(config.nnz_values); desc="Processing datasets for m=$(exp)d: ", showspeed=true)

        for (i,dataset_id) in enumerate(dataset_ids)
            dataset_name = "ID_$dataset_id"
            #try
                dataset_info = OpenML.list_datasets(output_format=DataFrame, filter="data_id/$dataset_id")
                dataset_name = isempty(dataset_info) ? dataset_name : first(dataset_info.name)
                df = OpenML.load(dataset_id) |> DataFrame
                
                target_col_index = findfirst(col -> eltype(col) <: CategoricalValue, eachcol(df))
                y = df[!, target_col_index]
                X = transpose(Matrix(select(df, Not(target_col_index))))

                d, n = size(X)
                s = max(1, round(Int, config.s_ratio * d))
                m = exp * d

                accs_list = []
                for nnz in config.nnz_values
                    local k::Int
                    if nnz >= 1
                        k = round(Int, nnz) # Absolute k value
                    else
                        k = ceil(Int, nnz * m) # Ratio-based k value
                    end
                    
                    println("k $k")

                    next!(progress; showvalues = generate_showvalues(dataset_name, i, n, m, k))

                    # Safety checks for k
                    if k >= m || k <= 0
                        @warn "For dataset '$dataset_name', invalid k=$k (m=$m) from nnz=$nnz. Skipping."
                        continue
                    end
                    push!(accs_list, run_model_suite(X, y, m, k, s, config.gamma; normalize=config.normalize))
                end

                if isempty(accs_list)
                    @warn "No valid k configurations found for dataset '$dataset_name'. Skipping."
                    continue
                end

                # Average the accuracies across all nnz values
                avg_accs = (
                    Statistics.mean(getindex.(accs_list, 1)), Statistics.mean(getindex.(accs_list, 2)),
                    Statistics.mean(getindex.(accs_list, 3)), Statistics.mean(getindex.(accs_list, 4))
                )

                push!(results_for_exp, (dataset_name, avg_accs...))

            # catch e
            #     for _ in eachindex(config.nnz_values) next!(progress) end
            #     @error "Error while processing dataset '$dataset_name' (ID: $dataset_id)" exception=(e, catch_backtrace())
            # end
        end

        if nrow(results_for_exp) > 0
            !isdir(results_dir) && mkpath(results_dir)
            output_path = joinpath(results_dir, "accuracies_m_$(exp)d_openml.csv")
            CSV.write(output_path, results_for_exp)
            println("\nResults for m = $(exp)d saved to '$output_path'")
        else
            println("\nNo results obtained for m = $(exp)d.")
        end
    end
    println("\n\n--- OpenML processing complete! ---")
end


"""
Runs experiments on vision datasets (FashionMNIST, CIFAR10).
"""
function vision_datasets_experiments(results_dir::String, config::Hyperparameters)
    datasets_to_process = [
        (name="CIFAR10", loader=CIFAR10)
        (name="FashionMNIST", loader=FashionMNIST)
    ]

    for exp in config.exp_values
        results_for_exp = DataFrame(
            DatasetName=String[], nnz=Float64[], k=Int[],
            FlyNNM_Binary=Float64[], FlyNNM_Uniform=Float64[],
            FlyNNA_Binary=Float64[], FlyNNA_Uniform=Float64[]
        )

        println("\nStarting processing for m = $(exp)d. Number of datasets: $(length(datasets_to_process)).")
        
        for d_config in datasets_to_process
            try
                X_all, y_all = d_config.loader()[:]
                X = MLUtils.flatten(X_all)
                y = y_all
                
                d, n = size(X)
                s = max(1, round(Int, config.s_ratio * d))
                m = round(Int, exp * d)

                println("Processing $(d_config.name) with m=$m...")
                progress = Progress(length(config.nnz_values); desc="$(d_config.name): ")
                
                for nnz in config.nnz_values
                    local k::Int
                    if nnz >= 1
                        k = round(Int, nnz) # Absolute k value
                    else
                        k = ceil(Int, nnz * m) # Ratio-based k value
                    end
                    
                    # Safety checks for k
                    if k >= m || k <= 0
                        @warn "For dataset '$(d_config.name)', invalid k=$k (m=$m) from nnz=$nnz. Skipping."
                        next!(progress); continue
                    end

                    accuracies = run_model_suite(X, y, m, k, s, config.gamma)
                    push!(results_for_exp, (d_config.name, nnz, k, accuracies...))
                    next!(progress; showvalues = [("k", k)])
                end
            catch e
                @warn "Error while processing dataset: $e"
            end
        end

        if nrow(results_for_exp) > 0
            !isdir(results_dir) && mkpath(results_dir)
            output_path = joinpath(results_dir, "accuracies_m_$(exp)d_vision.csv")
            CSV.write(output_path, results_for_exp)
            println("\nResults for m = $(exp)d saved to '$output_path'")
        else
            println("\nNo results for m = $(exp)d.")
        end
    end
    println("\n\n--- Vision processing complete! ---")
end


function load_config(config_path::String, experiment_type::String)::Hyperparameters
    config_data = TOML.parsefile(config_path)
    if !haskey(config_data, experiment_type)
        error("Config file '$config_path' does not contain a section for '$experiment_type'")
    end
    
    exp_config = config_data[experiment_type]
    exp_values = convert(Vector{Int}, exp_config["exp_values"])

    return Hyperparameters(
        exp_values,
        exp_config["nnz_values"],
        exp_config["gamma"],
        exp_config["s_ratio"],
        exp_config["normalize"]
    )
end

"""
Writes a provenance file next to the result CSVs, recording everything needed
to reproduce the run: Julia version, thread count, base seed, number of
cross-validation folds, the resolved hyperparameters and the versions of all
packages in the active environment.
"""
function write_provenance(results_dir::String, experiment::String, config::Hyperparameters)
    !isdir(results_dir) && mkpath(results_dir)

    # Resolved version of every package in the active environment.
    pkg_versions = Dict(
        info.name => string(info.version)
        for (_, info) in Pkg.dependencies() if info.version !== nothing
    )

    # Current git commit, if the script is run from inside the repository.
    git_commit = try
        strip(read(`git rev-parse HEAD`, String))
    catch
        "unknown"
    end

    metadata = Dict(
        "experiment" => experiment,
        "timestamp" => Libc.strftime("%Y-%m-%dT%H:%M:%S", time()),
        "julia_version" => string(VERSION),
        "nthreads" => Threads.nthreads(),
        "git_commit" => git_commit,
        "base_seed" => 42,
        "k_folds" => K_FOLDS,
        "config" => Dict(
            "exp_values" => config.exp_values,
            "nnz_values" => config.nnz_values,
            "gamma" => config.gamma,
            "s_ratio" => config.s_ratio,
            "normalize" => config.normalize,
        ),
        "packages" => pkg_versions,
    )

    output_path = joinpath(results_dir, "run_metadata_$(experiment).toml")
    open(output_path, "w") do io
        TOML.print(io, metadata)
    end
    println("Provenance written to '$output_path'")
end

function parse_commandline()
    s = ArgParseSettings(description = "Run classification experiments.")
    
    @add_arg_table! s begin
        "experiment"
            help = "The type of experiment to run: 'vision' or 'openml'."
            required = true
            range_tester = (x -> x in ["vision", "openml"])
        "--config", "-c"
            help = "Path to the TOML configuration file."
            default = "config.toml"
        "--output", "-o"
            help = "Directory to save CSV results."
            default = joinpath("..", "results")
    end
    
    return parse_args(s)
end

# Script entry point
function main()
    global_logger(SimpleLogger(open("experiment_log.txt", "w+")))
    parsed_args = parse_commandline()
    
    experiment = parsed_args["experiment"]
    config_path = parsed_args["config"]
    results_dir = parsed_args["output"]

    println("Starting experiment: $experiment.")
    println("Loading configuration from: $config_path")
    println("Results will be saved to: $results_dir")

    try
        config = load_config(config_path, experiment)

        # Record run provenance before the experiment, so it exists even if a
        # later dataset fails partway through.
        write_provenance(results_dir, experiment, config)

        if experiment == "vision"
            vision_datasets_experiments(results_dir, config)
        elseif experiment == "openml"
            openml_datasets_experiments(results_dir, config)
        end
    catch e
        println("An error occurred: ", e)
    end
end

main()
