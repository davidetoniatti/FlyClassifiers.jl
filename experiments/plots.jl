using DataFrames
using CSV
using Plots
using Statistics
using LaTeXStrings

println("Julia environment for plotting configured successfully.")

function generate_bar_plots(df::DataFrame, exp::Int, plots_dir::String)
    !isdir(plots_dir) && mkpath(plots_dir)

    model_names = names(df)[2:end]
    
    mean_accuracy = [mean(df[!, col]) for col in model_names]

    label_values = round.(mean_accuracy, digits=4)

    @info "Generating model comparison plot for m=$(exp)d"

    plot_obj = bar(
        replace.(model_names,"_" => " "),
        mean_accuracy,
        legend = false,
        xlabel = "model",
        ylabel = "accuracy (mean)",
        xrotation = 15,
        size = (800, 600),
        series_annotations = text.(label_values, :top, :center, 10)
    )

    ylims!(plot_obj, (0, maximum(mean_accuracy) * 1.15))

    filename = joinpath(plots_dir, "comparison_models_m_$(exp)d.png")
    savefig(plot_obj, filename)

    println("Comparison bar plot for m = $(exp)d saved to '$(plots_dir)/' directory.")
end

"""
Generates plots to compare model performance (for Vision experiments).
"""
function generate_vision_plots(df::DataFrame, exp::Int, plots_dir::String)
    model_names = names(df)[end-3:end]
    dataset_names = unique(df.DatasetName)
    
    !isdir(plots_dir) && mkpath(plots_dir)

    for name in dataset_names
        println("Generating plot for dataset: $name (m=$(exp)d)")

        subset_df = filter(row -> row.DatasetName == name, df)
        sort!(subset_df, :nnz, rev=false)
        x_positions = 1:nrow(subset_df)
        exponents = Int.(round.(log2.(subset_df.nnz)))
        x_labels = [L"2^{%$e}" for e in exponents]

        p = plot(
            xlabel = "Density (k)",
            ylabel = "Accuracy (%)",
            legend = :bottomright,
            xticks = (x_positions, x_labels),
            minorgrid = true,
            titlefontsize=10,
            tickfontsize=7,
            ylims = (20, 100),
            size = (800,600)
        )

        for model in model_names
            plot!(p,
                x_positions,
                subset_df[!, model] .* 100,
                label = replace(model, "_" => " "),
                marker = :circle,
                linewidth = 2
            )
        end
        
        filename = joinpath(plots_dir, "accuracies_m_$(exp)d_$(name).png")
        savefig(p, filename)
    end
end


"""
Generates scatter plots to compare model performance (for OpenML experiments).
"""
function generate_scatter_plots(results_df::DataFrame, exp::Int, plots_dir::String)
    println("\nCreating scatter plots for m = $(exp)d...")
    !isdir(plots_dir) && mkpath(plots_dir)

    if nrow(results_df) < 1
        @warn "Results DataFrame is empty for m=$(exp)d. Skipping plot generation."
        return
    end

    # Column names must match those written by experiments.jl:
    # FlyNNM_Binary, FlyNNM_Uniform, FlyNNA_Binary, FlyNNA_Uniform.
    comparisons = [
        (Symbol("FlyNNM_Binary"), Symbol("FlyNNM_Uniform"), "Multiplicative filter: Binary vs Uniform projection"),
        (Symbol("FlyNNA_Binary"), Symbol("FlyNNA_Uniform"), "Additive filter: Binary vs Uniform projection"),
        (Symbol("FlyNNM_Binary"), Symbol("FlyNNA_Binary"), "Binary Projection: Multiplicative vs Additive filter"),
        (Symbol("FlyNNM_Uniform"), Symbol("FlyNNA_Uniform"), "Uniform Projection: Multiplicative vs Additive filter"),
        (Symbol("FlyNNM_Binary"), Symbol("FlyNNA_Uniform"), "FlyNN-M original vs FlyNN-A original"),
        (Symbol("FlyNNM_Uniform"), Symbol("FlyNNA_Binary"), "FlyNN-M Uniform vs FlyNN-A Binary"),
    ]

    # model_cols = select(results_df, Not(:DatasetName))
    # min_acc, max_acc = extrema(Matrix(model_cols))
    # 
    # plot_margin = (max_acc - min_acc) * 0.05
    # lims = (min_acc - plot_margin, max_acc + plot_margin)
    #
    # AGGIUNTO: Definizione di limiti e tick fissi come richiesto.
    # Crea un range da 0.4 a 1.0 con step di 0.1
    fixed_ticks = 0.4:0.1:1.0
    # I limiti del grafico corrisponderanno al minimo e massimo dei tick
    fixed_lims = (minimum(fixed_ticks)+0.02, maximum(fixed_ticks)+0.02)

    for (x_col, y_col, title_str) in comparisons
        plt = scatter(
            results_df[!, x_col],
            results_df[!, y_col],
            xlabel=replace(string(x_col), "_" => " "),
            ylabel=replace(string(y_col), "_" => " "),
            legend=:topleft,
            aspect_ratio=1,
            xlims=fixed_lims,
            ylims=fixed_lims,
            xticks=fixed_ticks,
            yticks=fixed_ticks,
            markersize=5,
            markerstrokewidth=0,
            alpha=0.6,
            label="",
            size=(700, 700)
        )
        plot!(plt, [fixed_lims[1], fixed_lims[2]], [fixed_lims[1], fixed_lims[2]], style=:dash, color=:red, label="")

        centroid_x = mean(results_df[!, x_col])
        centroid_y = mean(results_df[!, y_col])

        scatter!(
            plt,
            [centroid_x],
            [centroid_y],
            label="Centroid",
            markercolor=:black,
            markershape=:star5,
            markersize=10,
            markerstrokewidth=1,
            markerstrokecolor=:yellow
        )

        filename = joinpath(plots_dir, "comparison_$(x_col)_vs_$(y_col)_m_$(exp)d.png")
        savefig(plt, filename)
    end
    println("Scatter plots for m = $(exp)d saved to '$(plots_dir)/' directory.")
end


"""
Generic function to find and process result files.
"""
function process_result_files(plot_type::String, file_pattern::Regex, plot_function::Function, results_dir::String, plots_dir::String)
    println("\n--- Generating $plot_type Plots ---")
    
    if !isdir(results_dir)
        println("'$results_dir' directory not found. Run experiments first.")
        return
    end

    result_files = filter(f -> occursin(file_pattern, f), readdir(results_dir))
    
    if isempty(result_files)
        println("No $plot_type result files found in '$results_dir'.")
        return
    end

    for file in result_files
        match_exp = match(r"m_(\d+)d", file)
        if isnothing(match_exp) continue end
        
        exp = parse(Int, match_exp.captures[1])
        filepath = joinpath(results_dir, file)
        df = CSV.read(filepath, DataFrame)
        plot_function(df, exp, plots_dir)
    end
    println("\n$plot_type plots generation complete.")
end


# Script entry point
function main()
    if !(1 <= length(ARGS) <= 3) || !(ARGS[1] in ["vision", "openml", "comparison"])
        println("Usage: julia $(@__FILE__) [vision|openml] [results_dir] [plots_dir]")
        println("\nArguments:")
        println("  [vision|openml|comparison]: (Required) The type of plots to generate.")
        println("  [results_dir]:   (Optional) Directory to read CSV results from. Defaults to '../results'.")
        println("  [plots_dir]:     (Optional) Directory to save generated plots. Defaults to '../plots'.")
        return
    end

    plot_type = ARGS[1]
    results_dir = length(ARGS) >= 2 ? ARGS[2] : joinpath("..", "results")
    plots_dir = length(ARGS) >= 3 ? ARGS[3] : joinpath("..", "plots")

    println("Generating $plot_type plots. Reading from: $results_dir, Saving to: $plots_dir")

    if plot_type == "vision"
        process_result_files("Vision", r"accuracies_m_\d+d_vision\.csv", generate_vision_plots, results_dir, plots_dir)
    elseif plot_type == "openml"
        process_result_files("OpenML", r"accuracies_m_\d+d_openml\.csv", generate_scatter_plots, results_dir, plots_dir)
    elseif plot_type == "comparison"
        process_result_files("comparison", r"accuracies_m_\d+d_openml\.csv", generate_bar_plots, results_dir, plots_dir)
    end
end

main()
