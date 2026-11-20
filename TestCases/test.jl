#!/usr/bin/env julia

# =============================================================================
# Lowe2005.jl
#
# Oceananigans GPU Boussinesq two-liquid model.
#
# PACANOWSKI-PHILANDER TURBULENCE CLOSURE
#
# PHASE-DEPENDENT VERTICAL VISCOSITY:
#   Molecular viscosity:
#       nu_molecular = 9.0e-7 m2/s
#
#   Background eddy viscosity:
#       Water  = 6.0e-6 m2/s
#       Sludge = 0.0 m2/s
#
# SALT INLET:
#   Lower fluid region extends to east boundary.
#
# SALT-WEDGE TRACKING:
#   - Leading edge (5% concentration)
#   - Trailing edge (5% concentration)
#   - Dense core (50% concentration)
#   - Concentration-weighted centroid
#   - Wedge length
#   - Sludge volume
#   - Toe propagation velocity
#   - Centroid propagation velocity
#   - CSV output
#   - Hotstart-aware diagnostic history
#
# GRID:
#   805 x 32 = 25,760 cells
#
# =============================================================================

using Oceananigans
using Oceananigans.ImmersedBoundaries: GridFittedBoundary
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using Oceananigans.Grids: Center, Face
using Oceananigans.Fields: Field, interior
using Oceananigans.BuoyancyFormulations: ∂z_b
using Oceananigans.Operators: ℑxᶜᵃᵃ, ∂zᶠᶜᶠ
using Oceananigans.Utils: launch!
using Oceananigans.TurbulenceClosures:
    AbstractScalarDiffusivity,
    VerticalFormulation,
    VerticallyImplicitTimeDiscretization,
    buoyancy_tracers,
    buoyancy_force

import Oceananigans.TurbulenceClosures

using KernelAbstractions: @kernel, @index
using CUDA
using JLD2
using WriteVTK
using Printf


# =============================================================================
# PACANOWSKI-PHILANDER TURBULENCE CLOSURE
# =============================================================================

struct PacanowskiPhilanderVerticalDiffusivity{TD, FT} <:
    AbstractScalarDiffusivity{TD, VerticalFormulation, 1}

    ν_molecular::FT

    ν_background_water::FT
    ν_background_sludge::FT

    ν₁::FT
    κ₀::FT

    c::FT
    n::FT

    gprime::FT

    turbulence_exponent::FT

    maximum_diffusivity::FT
    maximum_viscosity::FT
end


function PacanowskiPhilanderVerticalDiffusivity(
    time_discretization = VerticallyImplicitTimeDiscretization(),
    FT = Float64;

    ν_molecular = 9.0e-7,

    ν_background_water = 6.0e-6,
    ν_background_sludge = 0.0,

    ν₁ = 2.0e-7,
    κ₀ = 1.5e-9,

    c = 5.0,
    n = 2.0,

    gprime = 0.016,

    turbulence_exponent = 2.0,

    maximum_diffusivity = 2.0e-6,
    maximum_viscosity = 8.0e-6,
)

    TD = typeof(time_discretization)

    return PacanowskiPhilanderVerticalDiffusivity{TD, FT}(
        convert(FT, ν_molecular),
        convert(FT, ν_background_water),
        convert(FT, ν_background_sludge),
        convert(FT, ν₁),
        convert(FT, κ₀),
        convert(FT, c),
        convert(FT, n),
        convert(FT, gprime),
        convert(FT, turbulence_exponent),
        convert(FT, maximum_diffusivity),
        convert(FT, maximum_viscosity),
    )
end


const PPVD = PacanowskiPhilanderVerticalDiffusivity

@inline pp_fields(model) = model.closure_fields[2]


# =============================================================================
# CLOSURE LOCATIONS AND ACCESSORS
# =============================================================================

@inline TurbulenceClosures.viscosity_location(::PPVD) =
    (Center(), Center(), Face())

@inline TurbulenceClosures.diffusivity_location(::PPVD) =
    (Center(), Center(), Face())

@inline TurbulenceClosures.viscosity(::PPVD, diffusivities) =
    diffusivities.νz

@inline TurbulenceClosures.diffusivity(::PPVD, diffusivities, id) =
    diffusivities.κz


# =============================================================================
# CLOSURE FIELDS
# =============================================================================

function TurbulenceClosures.build_closure_fields(
    grid,
    clock,
    tracer_names,
    bcs,
    closure::PPVD
)

    κz = Field{Center, Center, Face}(grid)
    νz = Field{Center, Center, Face}(grid)
    νt = Field{Center, Center, Face}(grid)
    νbackground = Field{Center, Center, Face}(grid)
    alpha_sludge = Field{Center, Center, Face}(grid)
    Ri = Field{Center, Center, Face}(grid)

    return (;
        κz,
        νz,
        νt,
        νbackground,
        alpha_sludge,
        Ri,
    )
end


# =============================================================================
# RICHARDSON NUMBER
# =============================================================================

@inline ϕ²(i, j, k, grid, ϕ, args...) =
    ϕ(i, j, k, grid, args...)^2


@inline function shear_squaredᶜᶜᶠ(
    i, j, k,
    grid,
    velocities
)

    ∂z_u² = ℑxᶜᵃᵃ(
        i, j, k,
        grid,
        ϕ²,
        ∂zᶠᶜᶠ,
        velocities.u,
    )

    return ∂z_u²
end


@inline function Riᶜᶜᶠ(
    i, j, k,
    grid,
    velocities,
    buoyancy,
    tracers
)

    S² = shear_squaredᶜᶜᶠ(
        i, j, k,
        grid,
        velocities,
    )

    N² = ∂z_b(
        i, j, k,
        grid,
        buoyancy,
        tracers,
    )

    S²_min = eps(eltype(grid))

    Ri = max(zero(grid), N²) / max(S², S²_min)

    return Ri
end


# =============================================================================
# GPU KERNEL: PHASE-DEPENDENT PP DIFFUSIVITIES
# =============================================================================

@kernel function compute_pp_diffusivities!(
    diffusivities,
    grid,
    closure,
    velocities,
    tracers,
    buoyancy
)

    i, j, k = @index(Global, NTuple)

    Ri = Riᶜᶜᶠ(
        i, j, k,
        grid,
        velocities,
        buoyancy,
        tracers,
    )

    b_lower = tracers.b[i, j, k-1]
    b_upper = tracers.b[i, j, k]

    b_face = 0.5 * (b_lower + b_upper)

    alpha = clamp(
        -b_face / closure.gprime,
        0.0,
        1.0,
    )

    ν_background =
        (1 - alpha) * closure.ν_background_water +
        alpha * closure.ν_background_sludge

    turbulence_factor =
        (1 - alpha)^closure.turbulence_exponent

    denominator = 1 + closure.c * Ri

    ν_t = turbulence_factor *
          closure.ν₁ / denominator^closure.n

    κ_t = turbulence_factor *
          closure.ν₁ / denominator^(closure.n + 1)

    νz = closure.ν_molecular +
         ν_background +
         ν_t

    κz = closure.κ₀ + κ_t

    νz = min(νz, closure.maximum_viscosity)
    κz = min(κz, closure.maximum_diffusivity)

    @inbounds diffusivities.νz[i, j, k] = νz
    @inbounds diffusivities.κz[i, j, k] = κz
    @inbounds diffusivities.νt[i, j, k] = ν_t
    @inbounds diffusivities.νbackground[i, j, k] = ν_background
    @inbounds diffusivities.alpha_sludge[i, j, k] = alpha
    @inbounds diffusivities.Ri[i, j, k] = Ri

end


# =============================================================================
# UPDATE CLOSURE FIELDS
# =============================================================================

function TurbulenceClosures.compute_closure_fields!(
    diffusivities,
    closure::PPVD,
    model;
    parameters = :xyz
)

    arch = model.architecture
    grid = model.grid

    tracers = buoyancy_tracers(model)
    buoyancy = buoyancy_force(model)
    velocities = model.velocities

    pp_diffusivities = diffusivities isa Tuple ?
                       diffusivities[2] : diffusivities

    launch!(
        arch,
        grid,
        parameters,
        compute_pp_diffusivities!,
        pp_diffusivities,
        grid,
        closure,
        velocities,
        tracers,
        buoyancy,
    )

    return nothing
end


# =============================================================================
# MAIN
# =============================================================================

function main()

    # =========================================================================
    # GPU CHECK
    # =========================================================================

    CUDA.functional() ||
        error("CUDA is not functional. Check NVIDIA driver and CUDA.jl.")

    println()
    println("============================================================")
    println("CUDA GPU INFORMATION")
    println("============================================================")

    CUDA.versioninfo()


    # =========================================================================
    # USER SETTINGS
    # =========================================================================

    OUTPUT_DIR = get(
        ENV,
        "OUTPUT_DIR",
        "two_liquid_oceananigans_results",
    )

    OUTPUT_PREFIX = get(
        ENV,
        "OUTPUT_PREFIX",
        "two_liquid_oceananigans",
    )

    WRITE_JLD2 =
        lowercase(get(ENV, "WRITE_JLD2", "false")) == "true"

    WRITE_PARAVIEW =
        lowercase(get(ENV, "WRITE_PARAVIEW", "true")) == "true"

    STOP_TIME = parse(
        Float64,
        get(ENV, "STOP_TIME", "100.0"),
    )

    OUTPUT_INTERVAL = parse(
        Float64,
        get(ENV, "OUTPUT_INTERVAL", "10.0"),
    )

    INITIAL_DT = parse(
        Float64,
        get(ENV, "INITIAL_DT", "1.0e-4"),
    )

    MAX_DT = parse(
        Float64,
        get(ENV, "MAX_DT", "1.0"),
    )

    CFL_ENABLED =
        lowercase(get(ENV, "CFL_ENABLED", "true")) == "true"

    CFL = parse(
        Float64,
        get(ENV, "CFL", "0.15"),
    )

    STATIC_DT_ENABLED =
        lowercase(get(ENV, "STATIC_DT_ENABLED", "false")) == "true"

    STATIC_DT = parse(
        Float64,
        get(ENV, "STATIC_DT", "1.0e-3"),
    )

    DIAG_EVERY = parse(
        Int,
        get(ENV, "DIAG_EVERY", "10"),
    )

    PICKUP_FILE = get(ENV, "PICKUP_FILE", "")

    CHECKPOINT_DIR = get(
        ENV,
        "CHECKPOINT_DIR",
        joinpath(OUTPUT_DIR, "checkpoints"),
    )

    CHECKPOINT_INTERVAL = parse(
        Float64,
        get(ENV, "CHECKPOINT_INTERVAL", "30.0"),
    )

    WEDGE_DIAGNOSTIC_INTERVAL = parse(
        Float64,
        get(ENV, "WEDGE_DIAGNOSTIC_INTERVAL", string(OUTPUT_INTERVAL)),
    )

    @assert WEDGE_DIAGNOSTIC_INTERVAL > 0.0
    @assert OUTPUT_INTERVAL > 0.0


    # =========================================================================
    # SALT-WEDGE TRACKING SETTINGS
    # =========================================================================

    WEDGE_ALPHA_THRESHOLD = parse(
        Float64,
        get(ENV, "WEDGE_ALPHA_THRESHOLD", "0.05"),
    )

    WEDGE_CORE_THRESHOLD = parse(
        Float64,
        get(ENV, "WEDGE_CORE_THRESHOLD", "0.50"),
    )

    @assert 0.0 < WEDGE_ALPHA_THRESHOLD < 1.0
    @assert 0.0 < WEDGE_CORE_THRESHOLD < 1.0
    @assert WEDGE_CORE_THRESHOLD > WEDGE_ALPHA_THRESHOLD

    WEDGE_CSV = joinpath(
        OUTPUT_DIR,
        "$(OUTPUT_PREFIX)_wedge_position.csv",
    )


    # =========================================================================
    # PHYSICAL PROPERTIES
    # =========================================================================

    rho0 = 1000.0
    rho_sludge = 1001.6718
    delta_rho = rho_sludge - rho0

    g = 9.81
    gprime = g * delta_rho / rho0

    nu = 9.0e-7
    kappa_b = 1.5e-9


    # =========================================================================
    # PACANOWSKI-PHILANDER SETTINGS
    # =========================================================================

    PP_ENABLED =
        lowercase(get(ENV, "PP_ENABLED", "true")) == "true"

    PP_NU_MOLECULAR = parse(
        Float64,
        get(ENV, "PP_NU_MOLECULAR", string(nu)),
    )

    PP_NU_BACKGROUND_WATER = parse(
        Float64,
        get(ENV, "PP_NU_BACKGROUND_WATER", "6.0e-6"),
    )

    PP_NU_BACKGROUND_SLUDGE = parse(
        Float64,
        get(ENV, "PP_NU_BACKGROUND_SLUDGE", "0.0"),
    )

    PP_NU1 = parse(
        Float64,
        get(ENV, "PP_NU1", "2.0e-7"),
    )

    PP_KAPPA0 = parse(
        Float64,
        get(ENV, "PP_KAPPA0", string(kappa_b)),
    )

    PP_C = parse(
        Float64,
        get(ENV, "PP_C", "5.0"),
    )

    PP_N = parse(
        Float64,
        get(ENV, "PP_N", "2.0"),
    )

    PP_TURBULENCE_EXPONENT = parse(
        Float64,
        get(ENV, "PP_TURBULENCE_EXPONENT", "2.0"),
    )

    PP_MAX_NU = parse(
        Float64,
        get(ENV, "PP_MAX_NU", "8.0e-6"),
    )

    PP_MAX_KAPPA = parse(
        Float64,
        get(ENV, "PP_MAX_KAPPA", "2.0e-6"),
    )


    # =========================================================================
    # GEOMETRY
    # =========================================================================

    xL = -0.00785207748413086
    xA = 7.33836030960083
    xB = 7.3868021965026855
    xR = 8.000005722045898

    z0 = 0.0
    z1 = 0.023437976837158203
    z2 = 0.031247854232788086
    z3 = 0.15500009059906006

    Ly = 0.12070107460021973

    Nx = 805


    # =========================================================================
    # VERTICAL GRID GENERATION
    # =========================================================================

    N_bottom = 16
    N_step = 2
    N_upper = 14

    N_bottom_stretched = 16
    N_step_stretched = 2
    N_upper_stretched = 3

    bottom_ratio = 1.08
    step_ratio = 1.12
    upper_ratio = 1.08

    function graded_faces(a, b, N, N_stretched, ratio)

        @assert N >= 1
        @assert 1 <= N_stretched <= N
        @assert ratio >= 1.0

        growth = ratio .^ (0:N_stretched-1)

        weights = vcat(
            growth,
            fill(growth[end], N - N_stretched),
        )

        base_dz = (b - a) / sum(weights)

        widths = base_dz .* weights

        return vcat(a, a .+ cumsum(widths))
    end

    z_bottom = graded_faces(
        z0, z1, N_bottom, N_bottom_stretched, bottom_ratio,
    )

    z_step = graded_faces(
        z1, z2, N_step, N_step_stretched, step_ratio,
    )

    z_upper = graded_faces(
        z2, z3, N_upper, N_upper_stretched, upper_ratio,
    )

    z_faces = vcat(
        z_bottom,
        z_step[2:end],
        z_upper[2:end],
    )

    Nz = length(z_faces) - 1

    @assert Nz == 32
    @assert length(z_faces) == Nz + 1
    @assert issorted(z_faces)

    x_faces = collect(range(xL, xR; length=Nx+1))

    dx = (xR - xL) / Nx

    x_centers = 0.5 .* (
        x_faces[1:end-1] .+
        x_faces[2:end]
    )

    z_centers = 0.5 .* (
        z_faces[1:end-1] .+
        z_faces[2:end]
    )

    dz = diff(z_faces)


    # =========================================================================
    # GRID DIAGNOSTICS
    # =========================================================================

    println()
    println("============================================================")
    println("REDUCED GRID INFORMATION")
    println("============================================================")

    println("Horizontal cells:       ", Nx)
    println("Vertical cells:         ", Nz)
    println("Total cells:            ", Nx * Nz)
    println("Horizontal dx:          ", dx, " m")

    println()
    println("GLOBAL BOTTOM")
    println("Cells:                  ", N_bottom)
    println("Expansion ratio:        ", bottom_ratio)
    println("First dz:               ", dz[1], " m")
    println("Last dz:                ", dz[N_bottom], " m")

    println()
    println("INTERMEDIATE STEP")
    println("Cells:                  ", N_step)
    println("Expansion ratio:        ", step_ratio)
    println("First dz:               ", dz[N_bottom+1], " m")
    println("Last dz:                ", dz[N_bottom+N_step], " m")

    println()
    println("UPPER REGION")
    println("Cells:                  ", N_upper)
    println("Expansion cells:        ", N_upper_stretched)
    println("Expansion ratio:        ", upper_ratio)
    println("First dz:               ", dz[N_bottom+N_step+1], " m")
    println("Last dz:                ", dz[end], " m")

    println()
    println("Minimum dz:             ", minimum(dz), " m")
    println("Maximum dz:             ", maximum(dz), " m")
    println("============================================================")


    # =========================================================================
    # IMMERSED STEP GEOMETRY
    # =========================================================================

    @inline is_fluid(x, z) =
        (z < z1) ||
        ((z >= z1) && (z < z2) && (x < xA)) ||
        (z >= z2)

    active_cells = Array{Bool}(undef, Nx, 1, Nz)

    for k in 1:Nz, i in 1:Nx
        active_cells[i, 1, k] =
            is_fluid(x_centers[i], z_centers[k])
    end

    immersed_cells = .!active_cells

    n_fluid = count(active_cells)
    n_solid = count(immersed_cells)

    println()
    println("============================================================")
    println("IMMERSED BOUNDARY MASK DIAGNOSTICS")
    println("============================================================")

    println("Total cells:          ", length(active_cells))
    println("Fluid cells:          ", n_fluid)
    println("Solid cells:          ", n_solid)

    println(
        "Fluid fraction:       ",
        round(100 * n_fluid / length(active_cells), digits=2),
        "%",
    )

    println("============================================================")


    # =========================================================================
    # GPU GRID
    # =========================================================================

    arch = GPU()

    underlying_grid = RectilinearGrid(
        arch;
        size = (Nx, Nz),
        x = (xL, xR),
        z = z_faces,
        topology = (Open, Flat, Bounded),
        halo = (5, 5),
    )

    immersed_cells_gpu = CuArray(immersed_cells)

    grid = ImmersedBoundaryGrid(
        underlying_grid,
        GridFittedBoundary(immersed_cells_gpu);
        active_cells_map = true,
    )

    println()
    println("Grid created:")
    println(grid)


    # =========================================================================
    # PRESCRIBED VELOCITIES AND FLOW RATES
    # =========================================================================

    U_water = 0.02000000
    U_sludge = -0.00348584

    water_inlet_height = z3 - z0
    sludge_inlet_height = z1 - z0
    outlet_height = z3 - z2

    Q_water = U_water * Ly * water_inlet_height
    Q_sludge = abs(U_sludge) * Ly * sludge_inlet_height

    U_outlet = (Q_water + Q_sludge) / (Ly * outlet_height)

    Q_salt_model = abs(U_sludge) * Ly * sludge_inlet_height
    Q_outlet_model = U_outlet * Ly * outlet_height
    Q_east_net = Q_outlet_model - Q_salt_model

    @assert isapprox(
        Q_salt_model,
        Q_sludge;
        rtol = 1e-13,
        atol = 1e-15,
    )

    @assert isapprox(
        Q_east_net,
        Q_water;
        rtol = 1e-13,
        atol = 1e-15,
    )

    println()
    println("============================================================")
    println("PRESCRIBED VELOCITIES AND FLOW RATES")
    println("============================================================")

    @printf("Water inlet velocity:  %+12.8f m/s\n", U_water)
    @printf("Sludge inlet velocity: %+12.8f m/s\n", U_sludge)
    @printf("East outlet velocity:  %+12.8f m/s\n", U_outlet)

    @printf("Water flow rate:       %12.8e m³/s\n", Q_water)
    @printf("Sludge flow rate:      %12.8e m³/s\n", Q_sludge)
    @printf("East outlet flow rate: %12.8e m³/s\n", Q_outlet_model)
    @printf("East net flow rate:    %12.8e m³/s\n", Q_east_net)

    println("============================================================")


    # =========================================================================
    # BOUNDARY CONDITIONS
    # =========================================================================

    west_scheme = PerturbationAdvection(
        inflow_timescale = 0.10,
        outflow_timescale = 1.0,
    )

    west_u = NormalFlowBoundaryCondition(
        U_water;
        scheme = west_scheme,
    )

    @inline east_velocity(z, t, p) =
        ifelse(z < p.z1, p.U_sludge,
            ifelse(z >= p.z2, p.U_outlet, 0.0))

    east_u = NormalFlowBoundaryCondition(
        east_velocity;
        parameters = (
            z1 = z1,
            z2 = z2,
            U_sludge = U_sludge,
            U_outlet = U_outlet,
        ),
    )

    no_slip = ValueBoundaryCondition(0.0)
    free_slip = FluxBoundaryCondition(nothing)

    u_bcs = FieldBoundaryConditions(
        west = west_u,
        east = east_u,
        bottom = no_slip,
        top = free_slip,
        immersed = free_slip,
    )

    w_bcs = FieldBoundaryConditions()

    east_tracer_scheme = PerturbationAdvection(
        inflow_timescale = 1.0,
        outflow_timescale = 10.0,
    )

    @inline east_buoyancy(z, t, p) =
        ifelse(z < p.z1, -p.gprime, 0.0)

    b_bcs = FieldBoundaryConditions(
        west = ValueBoundaryCondition(0.0),

        east = ValueBoundaryCondition(
            east_buoyancy;
            parameters = (
                z1 = z1,
                gprime = gprime,
            ),
            scheme = east_tracer_scheme,
        ),

        bottom = FluxBoundaryCondition(0.0),
        top = FluxBoundaryCondition(0.0),
        immersed = FluxBoundaryCondition(0.0),
    )


    # =========================================================================
    # PRESSURE SOLVER
    # =========================================================================

    pressure_solver = ConjugateGradientPoissonSolver(grid)

    @info "Pressure solver" typeof(pressure_solver)
    @info "Pressure solver properties" propertynames(pressure_solver)

    if hasproperty(pressure_solver, :preconditioner)
        @info "Pressure preconditioner" typeof(
            pressure_solver.preconditioner,
        )
    end


    # =========================================================================
    # TURBULENCE CLOSURE
    # =========================================================================

    if PP_ENABLED

        pp_closure = PacanowskiPhilanderVerticalDiffusivity(
            VerticallyImplicitTimeDiscretization();

            ν_molecular = PP_NU_MOLECULAR,

            ν_background_water = PP_NU_BACKGROUND_WATER,
            ν_background_sludge = PP_NU_BACKGROUND_SLUDGE,

            ν₁ = PP_NU1,
            κ₀ = PP_KAPPA0,

            c = PP_C,
            n = PP_N,

            gprime = gprime,

            turbulence_exponent = PP_TURBULENCE_EXPONENT,

            maximum_viscosity = PP_MAX_NU,
            maximum_diffusivity = PP_MAX_KAPPA,
        )

        horizontal_closure = HorizontalScalarDiffusivity(
            ν = nu,
            κ = kappa_b,
        )

        closure = (
            horizontal_closure,
            pp_closure,
        )

    else

        closure = ScalarDiffusivity(
            ν = nu,
            κ = kappa_b,
        )

    end


    # =========================================================================
    # MODEL
    # =========================================================================

    model = NonhydrostaticModel(
        grid;
        tracers = :b,
        buoyancy = BuoyancyTracer(),
        advection = WENO(order=5),
        closure = closure,
        boundary_conditions = (
            u = u_bcs,
            w = w_bcs,
            b = b_bcs,
        ),
        pressure_solver,
    )

    @info "Time stepper" typeof(model.timestepper)
    @info "Turbulence closure" closure


    # =========================================================================
    # INITIAL CONDITIONS
    # =========================================================================

    x_patch_lo = 7.142
    x_patch_hi = 7.5703

    z_patch_lo = -0.066406
    z_patch_hi = 0.050781

    @inline initial_b(x, z) = ifelse(
        (
            (x >= xB) &&
            (z >= z0) &&
            (z <= z1)
        ) ||
        (
            (x >= x_patch_lo) &&
            (x <= x_patch_hi) &&
            (z >= z_patch_lo) &&
            (z <= z_patch_hi)
        ),
        -gprime,
        0.0,
    )

    set!(
        model;
        u = 0.0,
        w = 0.0,
        b = initial_b,
    )


    # =========================================================================
    # SIMULATION
    # =========================================================================

    mkpath(OUTPUT_DIR)
    mkpath(CHECKPOINT_DIR)

    if STATIC_DT_ENABLED

        @assert STATIC_DT > 0.0 "STATIC_DT must be positive."

        simulation = Simulation(
            model;
            Δt = STATIC_DT,
            stop_time = STOP_TIME,
        )

        @info "TIMESTEP MODE: STATIC" STATIC_DT

    elseif CFL_ENABLED

        @assert CFL > 0.0 "CFL must be positive."

        simulation = Simulation(
            model;
            Δt = INITIAL_DT,
            stop_time = STOP_TIME,
        )

        conjure_time_step_wizard!(
            simulation,
            IterationInterval(10);
            cfl = CFL,
            max_change = 1.10,
            max_Δt = MAX_DT,
        )

        @info "TIMESTEP MODE: ADAPTIVE CFL" CFL MAX_DT

    else

        @assert INITIAL_DT > 0.0 "INITIAL_DT must be positive."

        simulation = Simulation(
            model;
            Δt = INITIAL_DT,
            stop_time = STOP_TIME,
        )

        @info "TIMESTEP MODE: FIXED INITIAL_DT" INITIAL_DT

    end


    # =========================================================================
    # CHECKPOINT WRITER
    # =========================================================================

    simulation.output_writers[:checkpointer] = Checkpointer(
        model;
        schedule = TimeInterval(CHECKPOINT_INTERVAL),
        dir = CHECKPOINT_DIR,
        prefix = "Lowe2005_checkpoint",
        cleanup = false,
        verbose = true,
    )


    # =========================================================================
    # OUTPUT FIELDS
    # =========================================================================

    output_fields = if PP_ENABLED

        fields = pp_fields(model)

        (
            u = model.velocities.u,
            w = model.velocities.w,
            b = model.tracers.b,

            Ri = fields.Ri,

            alpha_sludge = fields.alpha_sludge,

            nu_background_vertical = fields.νbackground,

            nu_t_vertical = fields.νt,

            nu_effective_vertical = fields.νz,

            kappa_t_vertical = fields.κz,
        )

    else

        (
            u = model.velocities.u,
            w = model.velocities.w,
            b = model.tracers.b,
        )

    end


    if WRITE_JLD2

        simulation.output_writers[:jld2_fields] = JLD2Writer(
            model,
            output_fields;
            schedule = TimeInterval(OUTPUT_INTERVAL),
            filename = joinpath(
                OUTPUT_DIR,
                "$(OUTPUT_PREFIX).jld2",
            ),
            overwrite_existing = true,
        )

    end


    # =========================================================================
    # PARAVIEW OUTPUT
    # =========================================================================

    vtk_y_faces = [0.0, Ly]

    pvd_path = joinpath(OUTPUT_DIR, OUTPUT_PREFIX)

    pvd = WRITE_PARAVIEW ? paraview_collection(pvd_path) : nothing


    function xz_array(field)

        A = Array(interior(field))

        if ndims(A) == 3
            return dropdims(A; dims=2)
        else
            return A
        end

    end


    cell_scalar(Axz) = reshape(Axz, Nx, 1, Nz)


    function vertical_face_to_center(A)

        @assert size(A, 2) == Nz + 1

        return 0.5 .* (
            A[:, 1:end-1] .+
            A[:, 2:end]
        )

    end


    function write_paraview(sim)

        WRITE_PARAVIEW || return nothing

        m = sim.model
        t = m.clock.time

        u_face = xz_array(m.velocities.u)
        w_face = xz_array(m.velocities.w)
        b = xz_array(m.tracers.b)

        u_center = 0.5 .* (
            u_face[1:end-1, :] .+
            u_face[2:end, :]
        )

        w_center = 0.5 .* (
            w_face[:, 1:end-1] .+
            w_face[:, 2:end]
        )

        alpha_sludge = clamp.(
            -b ./ gprime,
            0.0,
            1.0,
        )

        density = rho0 .+ delta_rho .* alpha_sludge

        velocity_magnitude = sqrt.(
            u_center.^2 .+
            w_center.^2,
        )

        fluid_mask = Float64.(
            dropdims(active_cells; dims=2),
        )

        output_number = round(Int, t / OUTPUT_INTERVAL)

        vtk_filename = joinpath(
            OUTPUT_DIR,
            "$(OUTPUT_PREFIX)_t$(lpad(output_number, 6, '0'))",
        )

        vtk = vtk_grid(
            vtk_filename,
            x_faces,
            vtk_y_faces,
            z_faces,
        )

        vtk["alpha_sludge", VTKCellData()] =
            cell_scalar(alpha_sludge)

        vtk["density", VTKCellData()] =
            cell_scalar(density)

        vtk["buoyancy", VTKCellData()] =
            cell_scalar(b)

        vtk["velocity_x", VTKCellData()] =
            cell_scalar(u_center)

        vtk["velocity_z", VTKCellData()] =
            cell_scalar(w_center)

        vtk["velocity_magnitude", VTKCellData()] =
            cell_scalar(velocity_magnitude)

        vtk["fluid_mask", VTKCellData()] =
            cell_scalar(fluid_mask)


        if PP_ENABLED

            fields = pp_fields(m)

            Ri_faces = xz_array(fields.Ri)
            alpha_faces = xz_array(fields.alpha_sludge)
            nu_t_faces = xz_array(fields.νt)
            nu_background_faces = xz_array(fields.νbackground)
            nu_effective_faces = xz_array(fields.νz)
            kappa_faces = xz_array(fields.κz)

            Ri_center = vertical_face_to_center(Ri_faces)
            alpha_center = vertical_face_to_center(alpha_faces)
            nu_t_center = vertical_face_to_center(nu_t_faces)

            nu_background_center =
                vertical_face_to_center(nu_background_faces)

            nu_effective_center =
                vertical_face_to_center(nu_effective_faces)

            kappa_center = vertical_face_to_center(kappa_faces)

            vtk["Richardson_number", VTKCellData()] =
                cell_scalar(Ri_center)

            vtk["alpha_sludge_closure", VTKCellData()] =
                cell_scalar(alpha_center)

            vtk["nu_t_vertical", VTKCellData()] =
                cell_scalar(nu_t_center)

            vtk["nu_background_vertical", VTKCellData()] =
                cell_scalar(nu_background_center)

            vtk["nu_effective_vertical", VTKCellData()] =
                cell_scalar(nu_effective_center)

            vtk["kappa_t_vertical", VTKCellData()] =
                cell_scalar(kappa_center)

        end


        pvd[t] = vtk

        @info @sprintf(
            "Wrote ParaView snapshot at t = %.3f s",
            t,
        )

        return nothing

    end


    if WRITE_PARAVIEW

        simulation.callbacks[:paraview_output] = Callback(
            write_paraview,
            TimeInterval(OUTPUT_INTERVAL),
        )

    end


    # =========================================================================
    # TRANSPORT DIAGNOSTICS
    # =========================================================================

    function transport_diagnostics(sim)

        u_face = xz_array(sim.model.velocities.u)

        east_u_values = u_face[end, :]

        salt_indices = findall(z -> z < z1, z_centers)
        outlet_indices = findall(z -> z >= z2, z_centers)

        Q_salt_report = -sum(
            east_u_values[k] * Ly * dz[k]
            for k in salt_indices
        )

        Q_out_report = sum(
            east_u_values[k] * Ly * dz[k]
            for k in outlet_indices
        )

        Q_east_net_report = Q_out_report - Q_salt_report

        @info @sprintf(
            "EAST salt inflow = %.12e m^3/s (target %.12e) | outlet = %.12e | net east = %.12e",
            Q_salt_report,
            Q_sludge,
            Q_out_report,
            Q_east_net_report,
        )

        @info @sprintf(
            "EAST velocity: salt min/max = %.6e / %.6e | outlet min/max = %.6e / %.6e",
            minimum(east_u_values[salt_indices]),
            maximum(east_u_values[salt_indices]),
            minimum(east_u_values[outlet_indices]),
            maximum(east_u_values[outlet_indices]),
        )

        return nothing

    end


    # =========================================================================
    # SALT-WEDGE POSITION TRACKER
    # =========================================================================

    function read_previous_wedge_record(filename)

        if !isfile(filename)
            return nothing
        end

        lines = readlines(filename)

        for line in Iterators.reverse(lines)

            isempty(strip(line)) && continue

            startswith(strip(line), "time_s") && continue

            values = split(strip(line), ',')

            length(values) >= 9 || continue

            try

                return (
                    time = parse(Float64, values[1]),
                    toe = parse(Float64, values[2]),
                    centroid = parse(Float64, values[6]),
                )

            catch

                continue

            end

        end

        return nothing

    end


    # =========================================================================
    # INITIALIZE WEDGE DIAGNOSTIC HISTORY
    # =========================================================================

    # IMPORTANT:
    # previous_wedge must be a Ref because the callback reads and updates
    # its contents using previous_wedge[].
    #
    # A direct assignment of nothing or a tuple would cause:
    # MethodError: no method matching getindex(::Nothing)
    #
    # Fresh simulation:
    #   Start with no previous measurement and overwrite CSV.
    #
    # Hotstart:
    #   Recover the last measurement from the existing CSV and append.

previous_wedge = Ref{Any}(
    if !isempty(PICKUP_FILE)
        read_previous_wedge_record(WEDGE_CSV)
    else
        nothing
    end
)

    if !isempty(PICKUP_FILE) && isfile(WEDGE_CSV)

        println("Preserving existing wedge diagnostic history:")
        println(WEDGE_CSV)

    else

        open(WEDGE_CSV, "w") do io

            println(
                io,
                "time_s,toe_x_m,tail_x_m,wedge_length_m,core_x_m,centroid_x_m,sludge_volume_m3,toe_velocity_m_s,centroid_velocity_m_s"
            )

        end

    end


    # =========================================================================
    # WEDGE DIAGNOSTIC FUNCTION
    # =========================================================================

    function salt_wedge_diagnostics(sim)

        global_previous = previous_wedge[]

        t = time(sim)

        # Retrieve buoyancy field.
        b = xz_array(sim.model.tracers.b)

        # Convert buoyancy to sludge concentration.
        alpha = clamp.(
            -b ./ gprime,
            0.0,
            1.0,
        )

        fluid = dropdims(
            active_cells;
            dims=2,
        )

        # ---------------------------------------------------------------------
        # Maximum concentration along each vertical column.
        # ---------------------------------------------------------------------

        alpha_x = zeros(Float64, Nx)

        for i in 1:Nx

            maximum_alpha = 0.0

            for k in 1:Nz

                fluid[i, k] || continue

                maximum_alpha = max(
                    maximum_alpha,
                    alpha[i, k],
                )

            end

            alpha_x[i] = maximum_alpha

        end


        # ---------------------------------------------------------------------
        # Threshold-based wedge boundaries.
        # ---------------------------------------------------------------------

        wedge_indices = findall(
            x -> x >= WEDGE_ALPHA_THRESHOLD,
            alpha_x,
        )

        core_indices = findall(
            x -> x >= WEDGE_CORE_THRESHOLD,
            alpha_x,
        )


        # ---------------------------------------------------------------------
        # Sludge volume and concentration-weighted centroid.
        # ---------------------------------------------------------------------

        sludge_volume = 0.0
        sludge_moment = 0.0

        for i in 1:Nx, k in 1:Nz

            fluid[i, k] || continue

            cell_volume = dx * Ly * dz[k]

            sludge_amount = alpha[i, k] * cell_volume

            sludge_volume += sludge_amount

            sludge_moment += x_centers[i] * sludge_amount

        end

        x_centroid = sludge_volume > 0.0 ?
            sludge_moment / sludge_volume : NaN


        # ---------------------------------------------------------------------
        # Determine toe, tail, length, and core.
        # ---------------------------------------------------------------------

        if isempty(wedge_indices)

            x_toe = NaN
            x_tail = NaN
            wedge_length = NaN

        else

            i_toe = first(wedge_indices)
            i_tail = last(wedge_indices)

            x_toe = x_centers[i_toe]
            x_tail = x_centers[i_tail]

            wedge_length = x_tail - x_toe

        end


        if isempty(core_indices)

            x_core = NaN

        else

            i_core_left = first(core_indices)
            i_core_right = last(core_indices)

            x_core = 0.5 * (
                x_centers[i_core_left] +
                x_centers[i_core_right]
            )

        end


        # ---------------------------------------------------------------------
        # Position propagation velocities.
        # ---------------------------------------------------------------------

        toe_velocity = NaN
        centroid_velocity = NaN

        if !isnothing(global_previous)

            previous_time = global_previous.time

            Δt_wedge = t - previous_time

            if Δt_wedge > 0.0

                if isfinite(x_toe) &&
                   isfinite(global_previous.toe)

                    toe_velocity = (
                        x_toe - global_previous.toe
                    ) / Δt_wedge

                end

                if isfinite(x_centroid) &&
                   isfinite(global_previous.centroid)

                    centroid_velocity = (
                        x_centroid - global_previous.centroid
                    ) / Δt_wedge

                end

            end

        end


        # ---------------------------------------------------------------------
        # Write CSV.
        # ---------------------------------------------------------------------

        open(WEDGE_CSV, "a") do io

            @printf(
                io,
                "%.10f,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e\n",
                t,
                x_toe,
                x_tail,
                wedge_length,
                x_core,
                x_centroid,
                sludge_volume,
                toe_velocity,
                centroid_velocity,
            )

        end


        # ---------------------------------------------------------------------
        # Update previous measurement for next diagnostic.
        # ---------------------------------------------------------------------

        previous_wedge[] = (
            time = t,
            toe = x_toe,
            centroid = x_centroid,
        )


        # ---------------------------------------------------------------------
        # Console diagnostics.
        # ---------------------------------------------------------------------

        @info @sprintf(
            "SALT WEDGE | t=%9.3f s | toe=%10.6f m | tail=%10.6f m | length=%9.6f m | core=%10.6f m | centroid=%10.6f m | volume=%.6e m3",
            t,
            x_toe,
            x_tail,
            wedge_length,
            x_core,
            x_centroid,
            sludge_volume,
        )

        @info @sprintf(
            "WEDGE VELOCITY | toe=%+.6e m/s | centroid=%+.6e m/s",
            toe_velocity,
            centroid_velocity,
        )

        return nothing

    end


    # =========================================================================
    # STABILITY DIAGNOSTICS
    # =========================================================================

    function progress_message(sim)

        m = sim.model

        u = interior(m.velocities.u)
        w = interior(m.velocities.w)
        b = interior(m.tracers.b)

        u_max = maximum(abs, u)
        w_max = maximum(abs, w)

        b_min = minimum(b)
        b_max = maximum(b)

        u_finite = all(isfinite, u)
        w_finite = all(isfinite, w)
        b_finite = all(isfinite, b)


        if PP_ENABLED

            fields = pp_fields(m)

            νt_field = interior(fields.νt)
            νbackground_field = interior(fields.νbackground)
            νeffective_field = interior(fields.νz)
            κ_field = interior(fields.κz)
            Ri_field = interior(fields.Ri)
            alpha_field = interior(fields.alpha_sludge)

            nu_t_max = maximum(νt_field)
            nu_background_max = maximum(νbackground_field)
            nu_effective_max = maximum(νeffective_field)
            kappa_t_max = maximum(κ_field)
            Ri_max = maximum(Ri_field)
            alpha_max = maximum(alpha_field)

            pp_finite =
                all(isfinite, νt_field) &&
                all(isfinite, νbackground_field) &&
                all(isfinite, νeffective_field) &&
                all(isfinite, κ_field) &&
                all(isfinite, Ri_field) &&
                all(isfinite, alpha_field)

            @info @sprintf(
                "iter=%7d | t=%10.6f | dt=%.3e | umax=%.3e | wmax=%.3e | b=[%.3e, %.3e] | alpha_max=%.3f | nut_max=%.3e | nu_bg_max=%.3e | nu_eff_max=%.3e | kappa_max=%.3e | Ri_max=%.3e | finite=(%s,%s,%s,%s)",
                iteration(sim),
                time(sim),
                sim.Δt,
                u_max,
                w_max,
                b_min,
                b_max,
                alpha_max,
                nu_t_max,
                nu_background_max,
                nu_effective_max,
                kappa_t_max,
                Ri_max,
                string(u_finite),
                string(w_finite),
                string(b_finite),
                string(pp_finite),
            )

        else

            @info @sprintf(
                "iter=%7d | t=%10.6f | dt=%.3e | umax=%.3e | wmax=%.3e | b=[%.3e, %.3e] | finite(u,w,b)=(%s,%s,%s)",
                iteration(sim),
                time(sim),
                sim.Δt,
                u_max,
                w_max,
                b_min,
                b_max,
                string(u_finite),
                string(w_finite),
                string(b_finite),
            )

        end

        return nothing

    end


    # =========================================================================
    # REGISTER CALLBACKS
    # =========================================================================

    add_callback!(
        simulation,
        progress_message,
        IterationInterval(DIAG_EVERY),
    )

    add_callback!(
        simulation,
        transport_diagnostics,
        IterationInterval(DIAG_EVERY),
    )

    add_callback!(
        simulation,
        salt_wedge_diagnostics,
        TimeInterval(WEDGE_DIAGNOSTIC_INTERVAL),
    )


    # =========================================================================
    # RUN CONFIGURATION SUMMARY
    # =========================================================================

    println()
    println("============================================================")
    println("Oceananigans GPU two-liquid Boussinesq approximation")
    println("============================================================")

    println("Architecture:          NVIDIA CUDA GPU")
    println("Grid:                  $(Nx) x $(Nz)")
    println("Total cells:           $(Nx * Nz)")
    println("Original cells:        103040")
    println("Cell reduction:        75%")
    println("Horizontal spacing:    $(dx) m")
    println("Vertical spacing:      smoothly graded")

    println()
    println("Bottom cells:          $(N_bottom)")
    println("Bottom expansion:      $(bottom_ratio)")
    println("First cell height:     $(dz[1]) m")

    println("Step cells:            $(N_step)")
    println("Step expansion:        $(step_ratio)")

    println("Upper cells:           $(N_upper)")
    println("Upper expansion:       $(upper_ratio)")

    println()
    println("Fluid cells:           $(n_fluid)")
    println("Solid cells:           $(n_solid)")
    println("Reduced gravity:       $(gprime) m/s²")

    println()
    println("WATER AND SLUDGE")

    println("Water inlet:           $(U_water) m/s")
    println("Sludge inlet:          $(U_sludge) m/s")
    println("East outlet:           $(U_outlet) m/s")

    println("Water flow rate:       $(Q_water) m³/s")
    println("Sludge flow rate:      $(Q_sludge) m³/s")

    println()
    println("TURBULENCE")

    println("PP enabled:            $(PP_ENABLED)")

    if PP_ENABLED

        println("PP time discretization: Vertically implicit")
        println("Molecular viscosity:   $(PP_NU_MOLECULAR) m²/s")
        println("Water background nut:  $(PP_NU_BACKGROUND_WATER) m²/s")
        println("Sludge background nut: $(PP_NU_BACKGROUND_SLUDGE) m²/s")
        println("PP nu1:                $(PP_NU1) m²/s")
        println("PP kappa0:             $(PP_KAPPA0) m²/s")
        println("PP coefficient c:      $(PP_C)")
        println("PP exponent n:          $(PP_N)")
        println("Turbulence exponent:   $(PP_TURBULENCE_EXPONENT)")
        println("Maximum total nu:      $(PP_MAX_NU) m²/s")
        println("Maximum kappa:         $(PP_MAX_KAPPA) m²/s")

    else

        println("Molecular-only diffusion")

    end


    println()
    println("SALT-WEDGE TRACKING")

    println("Toe threshold:         $(WEDGE_ALPHA_THRESHOLD)")
    println("Core threshold:        $(WEDGE_CORE_THRESHOLD)")
    println("Diagnostic interval:   $(WEDGE_DIAGNOSTIC_INTERVAL) s")
    println("CSV output:             $(WEDGE_CSV)")

    println()
    println("TIMESTEP")

    if STATIC_DT_ENABLED

        println("Mode:                  Static timestep")
        println("Static Δt:             $(STATIC_DT) s")

    elseif CFL_ENABLED

        println("Mode:                  Adaptive CFL")
        println("CFL enabled:           true")
        println("CFL target:            $(CFL)")
        println("Initial Δt:            $(INITIAL_DT) s")
        println("Maximum Δt:            $(MAX_DT) s")

    else

        println("Mode:                  Fixed INITIAL_DT")
        println("Fixed Δt:              $(INITIAL_DT) s")

    end

    println()
    println("OUTPUT")

    println("Stop time:             $(STOP_TIME) s")
    println("Output interval:       $(OUTPUT_INTERVAL) s")
    println("Diagnostics every:     $(DIAG_EVERY) iterations")
    println("ParaView output:       $(WRITE_PARAVIEW)")
    println("JLD2 output:           $(WRITE_JLD2)")
    println("Output directory:      $(OUTPUT_DIR)")

    println()
    println("CHECKPOINTING")

    println(
        "Pickup file:           ",
        isempty(PICKUP_FILE) ? "None (fresh run)" : PICKUP_FILE,
    )

    println("Checkpoint interval:   $(CHECKPOINT_INTERVAL) s")
    println("Checkpoint directory:  $(CHECKPOINT_DIR)")
    println("Final checkpoint:      enabled")

    println("============================================================")
    println()


    # =========================================================================
    # RUN
    # =========================================================================

    if !isempty(PICKUP_FILE)

        isfile(PICKUP_FILE) ||
            error("Checkpoint file does not exist: $PICKUP_FILE")

        println("Loading checkpoint:")
        println(PICKUP_FILE)

    else

        println("Starting fresh simulation from t = 0.")

    end


    wall_start = time_ns()

    iteration_start = iteration(simulation)
    time_start = time(simulation)

    try

        if isempty(PICKUP_FILE)

            run!(
                simulation;
                checkpoint_at_end = true,
            )

        else

            run!(
                simulation;
                pickup = PICKUP_FILE,
                checkpoint_at_end = true,
            )

        end

    finally

        elapsed = (time_ns() - wall_start) / 1e9

        completed_iterations =
            iteration(simulation) - iteration_start

        simulated_time =
            time(simulation) - time_start

        @info "Simulation performance" (
            wall_seconds = elapsed,

            iterations = completed_iterations,

            simulated_seconds = simulated_time,

            milliseconds_per_iteration =
                completed_iterations > 0 ?
                1000 * elapsed / completed_iterations : NaN,

            simulated_seconds_per_wall_second =
                elapsed > 0 ?
                simulated_time / elapsed : NaN
        )

        if WRITE_PARAVIEW
            close(pvd)
        end

    end


    println()
    println("============================================================")
    println("SIMULATION COMPLETE")
    println("============================================================")

    println("Final simulation time: ", time(simulation))
    println("Final iteration:       ", iteration(simulation))

    if WRITE_JLD2
        println(
            "JLD2: ",
            joinpath(OUTPUT_DIR, "$(OUTPUT_PREFIX).jld2"),
        )
    end

    if WRITE_PARAVIEW
        println(
            "ParaView: ",
            joinpath(OUTPUT_DIR, "$(OUTPUT_PREFIX).pvd"),
        )
    end

    println("Wedge CSV:             ", WEDGE_CSV)
    println("Checkpoints:           ", CHECKPOINT_DIR)
    println("============================================================")

end


main()
