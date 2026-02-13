@testsnippet LocalScaleSetup begin
    using EnvironmentalTransport
    using ModelingToolkit
    using ModelingToolkit: t, mtkcompile
    using OrdinaryDiffEqDefault
    using DynamicQuantities
    using Test
end

# =============================================================================
# Structural Tests
# =============================================================================

@testitem "AtmosphericStability structure" setup = [LocalScaleSetup] tags = [:local_met] begin
    sys = AtmosphericStability()

    # Check number of equations
    @test length(equations(sys)) == 4

    # Check expected variables exist
    var_names = Symbol.(unknowns(sys))
    @test Symbol("θ(t)") in var_names
    @test Symbol("dT_dz(t)") in var_names
    @test Symbol("dθ_dz(t)") in var_names
    @test Symbol("S(t)") in var_names

    # Check expected parameters exist
    param_names = Symbol.(parameters(sys))
    @test :T in param_names
    @test :T_below in param_names
    @test :p in param_names
    @test :Δz in param_names
end

@testitem "SurfaceLayerProfile structure" setup = [LocalScaleSetup] tags = [:local_met] begin
    sys = SurfaceLayerProfile()

    # Check number of equations
    @test length(equations(sys)) == 7

    # Check expected variables exist
    var_names = Symbol.(unknowns(sys))
    @test Symbol("L(t)") in var_names
    @test Symbol("ζ(t)") in var_names
    @test Symbol("φ_m(t)") in var_names
    @test Symbol("φ_h(t)") in var_names
    @test Symbol("ψ_m(t)") in var_names
    @test Symbol("ψ_h(t)") in var_names
    @test Symbol("ū(t)") in var_names

    # Check expected parameters exist
    param_names = Symbol.(parameters(sys))
    @test :z in param_names
    @test :z₀ in param_names
    @test :u_star in param_names
    @test :q_z in param_names
end

@testitem "LocalScaleMeteorology structure" setup = [LocalScaleSetup] tags = [:local_met] begin
    sys = LocalScaleMeteorology()

    # Check that subsystems are composed (not duplicated)
    subsys_names = Symbol.(nameof.(ModelingToolkit.get_systems(sys)))
    @test :stability in subsys_names
    @test :surface in subsys_names

    # Check total equations: 4 (AtmosphericStability) + 7 (SurfaceLayerProfile) + 2 (own) = 13
    @test length(equations(sys)) == 13

    # Check own variables
    var_names = Symbol.(unknowns(sys))
    @test Symbol("L_inv(t)") in var_names
    @test Symbol("pasquill_class(t)") in var_names
end

# =============================================================================
# Equation Verification Tests - Dry Adiabatic Lapse Rate (Eq. 16.8)
# =============================================================================

@testitem "Dry adiabatic lapse rate" setup = [LocalScaleSetup] tags = [:local_met] begin
    # From Seinfeld & Pandis p.724: Γ = g/ĉ_p = 9.807/1005 = 9.76 K/km
    g = 9.807  # m/s²
    ĉ_p = 1005.0  # J/(kg·K)
    Γ_expected = g / ĉ_p  # K/m

    # Convert to K/km for comparison with textbook
    Γ_km = Γ_expected * 1000  # K/km
    @test isapprox(Γ_km, 9.76, rtol = 0.01)

    # Test in AtmosphericStability system
    sys = AtmosphericStability()
    csys = mtkcompile(sys)

    # When actual lapse rate equals dry adiabatic, stability should be ~0
    # dT/dz = -Γ_d means neutral atmosphere
    T_surface = 288.15  # K
    Δz = 100.0  # m
    T_above = T_surface - Γ_expected * Δz  # Temperature at height Δz above

    prob = ODEProblem(csys, Dict(), (0.0, 1.0), Dict(
        csys.T => T_above,
        csys.T_below => T_surface,
        csys.Δz => Δz,
        csys.p => 101325.0,
    ))
    sol = solve(prob)

    # Stability parameter S = dθ/dz should be approximately 0 for neutral atmosphere
    @test isapprox(sol[csys.S][end], 0.0, atol = 1e-5)
end

# =============================================================================
# Equation Verification Tests - Potential Temperature (Eq. 16.14)
# =============================================================================

@testitem "Potential temperature" setup = [LocalScaleSetup] tags = [:local_met] begin
    # From Seinfeld & Pandis Eq. 16.14: θ = T(p₀/p)^0.286
    # At sea level (p = p₀), θ = T

    sys = AtmosphericStability()
    csys = mtkcompile(sys)

    T = 288.15  # K
    p₀ = 101325.0  # Pa

    # Test 1: At sea level, θ = T
    prob = ODEProblem(csys, Dict(), (0.0, 1.0), Dict(
        csys.T => T,
        csys.p => p₀,
    ))
    sol = solve(prob)
    @test isapprox(sol[csys.θ][end], T, rtol = 1e-6)

    # Test 2: At lower pressure, θ > T
    # At 850 hPa, with T = 280 K
    p_850 = 85000.0  # Pa
    T_850 = 280.0  # K
    θ_expected = T_850 * (p₀ / p_850)^0.286

    prob2 = ODEProblem(csys, Dict(), (0.0, 1.0), Dict(
        csys.T => T_850,
        csys.p => p_850,
    ))
    sol2 = solve(prob2)
    @test isapprox(sol2[csys.θ][end], θ_expected, rtol = 0.01)

    # θ should be greater than T at lower pressure
    @test sol2[csys.θ][end] > T_850
end

# =============================================================================
# Stability Classification Tests
# =============================================================================

@testitem "Atmospheric stability classification" setup = [LocalScaleSetup] tags = [:local_met] begin
    sys = AtmosphericStability()
    csys = mtkcompile(sys)

    T_surface = 288.15  # K
    Δz = 100.0  # m

    # Test 1: Stable atmosphere - temperature decreases less than adiabatic
    # (or even increases with height - inversion)
    T_stable = T_surface - 0.005 * Δz  # 0.5 K/100m, less than 0.976 K/100m
    prob_stable = ODEProblem(csys, Dict(), (0.0, 1.0), Dict(
        csys.T => T_stable,
        csys.T_below => T_surface,
        csys.Δz => Δz,
        csys.p => 101325.0,
    ))
    sol_stable = solve(prob_stable)
    # S > 0 for stable (dθ/dz > 0)
    @test sol_stable[csys.S][end] > 0

    # Test 2: Unstable atmosphere - superadiabatic lapse rate
    T_unstable = T_surface - 0.015 * Δz  # 1.5 K/100m, more than 0.976 K/100m
    prob_unstable = ODEProblem(csys, Dict(), (0.0, 1.0), Dict(
        csys.T => T_unstable,
        csys.T_below => T_surface,
        csys.Δz => Δz,
        csys.p => 101325.0,
    ))
    sol_unstable = solve(prob_unstable)
    # S < 0 for unstable (dθ/dz < 0)
    @test sol_unstable[csys.S][end] < 0
end

# =============================================================================
# Monin-Obukhov Length Tests (Eq. 16.70)
# =============================================================================

@testitem "Monin-Obukhov length" setup = [LocalScaleSetup] tags = [:local_met] begin
    # L = -ρĉ_p T₀ u*³ / (κ g q̄_z) (Eq. 16.70)

    sys = SurfaceLayerProfile()
    csys = mtkcompile(sys)

    # Test 1: Neutral conditions (very small heat flux)
    # When q_z ≈ 0, |L| should be very large (Table 16.2)
    prob_neutral = ODEProblem(csys, Dict(), (0.0, 1.0), Dict(
        csys.u_star => 0.3,
        csys.q_z => 1e-6,
        csys.T₀ => 288.15,
        csys.ρ => 1.225,
    ))
    sol_neutral = solve(prob_neutral)
    @test abs(sol_neutral[csys.L][end]) > 1e6

    # Test 2: Unstable conditions (positive heat flux - surface heating)
    # L should be negative (Table 16.2)
    prob_unstable = ODEProblem(csys, Dict(), (0.0, 1.0), Dict(
        csys.u_star => 0.4,
        csys.q_z => 100.0,
        csys.T₀ => 288.15,
        csys.ρ => 1.225,
    ))
    sol_unstable = solve(prob_unstable)
    @test sol_unstable[csys.L][end] < 0

    # Test 3: Stable conditions (negative heat flux - surface cooling)
    # L should be positive (Table 16.2)
    prob_stable = ODEProblem(csys, Dict(), (0.0, 1.0), Dict(
        csys.u_star => 0.3,
        csys.q_z => -50.0,
        csys.T₀ => 288.15,
        csys.ρ => 1.225,
    ))
    sol_stable = solve(prob_stable)
    @test sol_stable[csys.L][end] > 0

    # Test 4: Verify numerical value against hand calculation (p. 751 example)
    # Agricultural area, z₀ = 0.1 m, stability class B
    # Using (1/L) = -0.037 + 0.029*log10(0.1) = -0.037 - 0.029 = -0.066
    # So L ≈ -15.15 m
    # With u* = 0.3 m/s, T₀ = 288.15 K, ρ = 1.225 kg/m³:
    # L = -1.225 * 1005 * 288.15 * 0.3^3 / (0.4 * 9.807 * q_z)
    # For L = -15 m: q_z = -1.225*1005*288.15*0.027 / (0.4*9.807*(-15)) ≈ 163 W/m²
    ρ = 1.225
    ĉ_p = 1005.0
    T₀ = 288.15
    u_star = 0.3
    κ = 0.4
    g = 9.807
    q_z_test = ρ * ĉ_p * T₀ * u_star^3 / (κ * g * 15.0)  # q_z for L = -15 m
    prob_ex = ODEProblem(csys, Dict(), (0.0, 1.0), Dict(
        csys.u_star => u_star,
        csys.q_z => q_z_test,
        csys.T₀ => T₀,
        csys.ρ => ρ,
    ))
    sol_ex = solve(prob_ex)
    @test isapprox(sol_ex[csys.L][end], -15.0, rtol = 0.01)
end

# =============================================================================
# Businger-Dyer Stability Functions Tests (Eq. 16.75)
# =============================================================================

@testitem "Businger-Dyer stability functions" setup = [LocalScaleSetup] tags = [:local_met] begin
    sys = SurfaceLayerProfile()
    csys = mtkcompile(sys)

    # Test 1: Neutral conditions (ζ ≈ 0)
    # φ_m = φ_h = 1 (Eq. 16.74)
    prob_neutral = ODEProblem(csys, Dict(), (0.0, 1.0), Dict(
        csys.u_star => 0.3,
        csys.q_z => 1e-8,
        csys.z => 10.0,
        csys.T₀ => 288.15,
        csys.ρ => 1.225,
    ))
    sol_neutral = solve(prob_neutral)
    @test isapprox(sol_neutral[csys.ζ][end], 0.0, atol = 1e-4)
    @test isapprox(sol_neutral[csys.φ_m][end], 1.0, rtol = 0.1)
    @test isapprox(sol_neutral[csys.φ_h][end], 1.0, rtol = 0.1)

    # Test 2: Stable conditions (ζ > 0)
    # φ_m = φ_h = 1 + 4.7ζ (Eq. 16.75)
    prob_stable = ODEProblem(csys, Dict(), (0.0, 1.0), Dict(
        csys.u_star => 0.2,
        csys.q_z => -50.0,
        csys.z => 10.0,
        csys.T₀ => 288.15,
        csys.ρ => 1.225,
    ))
    sol_stable = solve(prob_stable)
    ζ_stable = sol_stable[csys.ζ][end]
    @test ζ_stable > 0
    expected_φ = 1 + 4.7 * ζ_stable
    @test isapprox(sol_stable[csys.φ_m][end], expected_φ, rtol = 0.01)
    @test isapprox(sol_stable[csys.φ_h][end], expected_φ, rtol = 0.01)

    # Test 3: Unstable conditions (ζ < 0)
    # φ_m = (1 - 15ζ)^(-1/4) (Eq. 16.75)
    # φ_h = (1 - 15ζ)^(-1/2) (Businger et al. 1971)
    prob_unstable = ODEProblem(csys, Dict(), (0.0, 1.0), Dict(
        csys.u_star => 0.3,
        csys.q_z => 100.0,
        csys.z => 10.0,
        csys.T₀ => 288.15,
        csys.ρ => 1.225,
    ))
    sol_unstable = solve(prob_unstable)
    ζ_unstable = sol_unstable[csys.ζ][end]
    @test ζ_unstable < 0
    expected_φm = (1 - 15 * ζ_unstable)^(-0.25)
    expected_φh = (1 - 15 * ζ_unstable)^(-0.5)
    @test isapprox(sol_unstable[csys.φ_m][end], expected_φm, rtol = 0.01)
    @test isapprox(sol_unstable[csys.φ_h][end], expected_φh, rtol = 0.01)
end

# =============================================================================
# Logarithmic Wind Profile Tests (Eq. 16.66)
# =============================================================================

@testitem "Logarithmic wind profile" setup = [LocalScaleSetup] tags = [:local_met] begin
    # Eq. 16.66: ū(z) = (u*/κ)ln(z/z₀) for neutral (adiabatic) conditions

    sys = SurfaceLayerProfile()
    csys = mtkcompile(sys)

    κ = 0.4  # von Karman constant
    u_star = 0.4  # m/s
    z₀ = 0.1  # m
    z = 10.0  # m

    # Expected neutral wind speed (Eq. 16.66)
    ū_expected = (u_star / κ) * log(z / z₀)

    prob = ODEProblem(csys, Dict(), (0.0, 1.0), Dict(
        csys.u_star => u_star,
        csys.z₀ => z₀,
        csys.z => z,
        csys.q_z => 1e-8,
        csys.T₀ => 288.15,
        csys.ρ => 1.225,
    ))
    sol = solve(prob)
    @test isapprox(sol[csys.ū][end], ū_expected, rtol = 0.05)

    # Test with Wangara experiment values (p.745)
    # z₀ = 0.0015 m, u* = 0.4 m/s
    # ū(0.5m) = 5.8 m/s, ū(16m) = 9.3 m/s (Deardorff 1978)
    z₀_wangara = 0.0015  # m
    z_low = 0.5  # m
    z_high = 16.0  # m
    ū_low_expected = (u_star / κ) * log(z_low / z₀_wangara)  # ≈ 5.8 m/s
    ū_high_expected = (u_star / κ) * log(z_high / z₀_wangara)  # ≈ 9.3 m/s

    prob_low = ODEProblem(csys, Dict(), (0.0, 1.0), Dict(
        csys.u_star => u_star,
        csys.z₀ => z₀_wangara,
        csys.z => z_low,
        csys.q_z => 1e-8,
        csys.T₀ => 288.15,
        csys.ρ => 1.225,
    ))
    sol_low = solve(prob_low)
    @test isapprox(sol_low[csys.ū][end], ū_low_expected, rtol = 0.05)
    @test isapprox(sol_low[csys.ū][end], 5.8, rtol = 0.02)  # Within 2% of observation

    prob_high = ODEProblem(csys, Dict(), (0.0, 1.0), Dict(
        csys.u_star => u_star,
        csys.z₀ => z₀_wangara,
        csys.z => z_high,
        csys.q_z => 1e-8,
        csys.T₀ => 288.15,
        csys.ρ => 1.225,
    ))
    sol_high = solve(prob_high)
    @test isapprox(sol_high[csys.ū][end], ū_high_expected, rtol = 0.05)
    @test isapprox(sol_high[csys.ū][end], 9.3, rtol = 0.02)  # Within 2% of observation
end

# =============================================================================
# Pasquill Stability Class Tests (Table 16.4, Eq. 16.83)
# =============================================================================

@testitem "Pasquill stability classification" setup = [LocalScaleSetup] tags = [:local_met] begin
    sys = LocalScaleMeteorology()
    csys = mtkcompile(sys)

    # Test 1: Verify Golder (1972) example from p.751
    # Agricultural area, z₀ = 0.1 m, class B:
    # (1/L) = -0.037 + 0.029*log10(0.1) = -0.037 - 0.029 = -0.066 m⁻¹
    # So L ≈ -15.15 m
    # Need to set parameters to get L ≈ -15 m
    ρ = 1.225
    ĉ_p = 1005.0
    T₀ = 288.15
    κ = 0.4
    g = 9.807
    L_target = -15.0  # m (moderately unstable, class B)
    u_star = 0.3
    q_z_target = -ρ * ĉ_p * T₀ * u_star^3 / (κ * g * L_target)

    prob_B = ODEProblem(csys, Dict(), (0.0, 1.0), Dict(
        csys.surface.u_star => u_star,
        csys.surface.q_z => q_z_target,
        csys.surface.z₀ => 0.1,
        csys.surface.T₀ => T₀,
        csys.surface.ρ => ρ,
    ))
    sol_B = solve(prob_B)
    @test sol_B[csys.surface.L][end] < 0  # Unstable
    # Should be class B (2) or nearby
    @test sol_B[csys.pasquill_class][end] <= 3  # A, B, or C

    # Test 2: Very stable conditions should give class near 6 (F)
    prob_F = ODEProblem(csys, Dict(), (0.0, 1.0), Dict(
        csys.surface.u_star => 0.2,
        csys.surface.q_z => -100.0,
        csys.surface.z₀ => 0.1,
        csys.surface.T₀ => 280.0,
        csys.surface.ρ => 1.225,
    ))
    sol_F = solve(prob_F)
    @test sol_F[csys.surface.L][end] > 0  # Stable has positive L
    @test sol_F[csys.pasquill_class][end] >= 4  # D, E, or F
end

# =============================================================================
# Composed System Tests
# =============================================================================

@testitem "LocalScaleMeteorology composed system" setup = [LocalScaleSetup] tags = [:local_met] begin
    sys = LocalScaleMeteorology()
    csys = mtkcompile(sys)

    # Verify that subsystem variables are accessible via dot notation
    prob = ODEProblem(csys, Dict(), (0.0, 1.0), Dict(
        csys.stability.T => 298.15,
        csys.stability.T_below => 300.15,
        csys.stability.p => 100000.0,
        csys.stability.Δz => 10.0,
        csys.surface.z => 10.0,
        csys.surface.z₀ => 0.1,
        csys.surface.T₀ => 300.15,
        csys.surface.ρ => 1.18,
        csys.surface.u_star => 0.35,
        csys.surface.q_z => 150.0,
    ))
    sol = solve(prob)

    # Stability subsystem outputs
    @test sol[csys.stability.θ][end] > 0
    @test sol[csys.stability.dθ_dz][end] != 0

    # Surface layer subsystem outputs
    @test sol[csys.surface.L][end] < 0  # Unstable with positive heat flux
    @test sol[csys.surface.ū][end] > 0

    # Composite system outputs
    @test sol[csys.L_inv][end] < 0  # 1/L negative for unstable
    @test 1 <= sol[csys.pasquill_class][end] <= 6
end

# =============================================================================
# Limiting Behavior Tests
# =============================================================================

@testitem "Limiting behaviors" setup = [LocalScaleSetup] tags = [:local_met] begin
    sys = SurfaceLayerProfile()
    csys = mtkcompile(sys)

    # Test 1: As |L| → ∞ (neutral), ψ_m → 0
    prob_neutral = ODEProblem(csys, Dict(), (0.0, 1.0), Dict(
        csys.u_star => 0.3,
        csys.q_z => 1e-10,
        csys.z => 10.0,
        csys.z₀ => 0.1,
        csys.T₀ => 288.15,
        csys.ρ => 1.225,
    ))
    sol_neutral = solve(prob_neutral)
    @test isapprox(sol_neutral[csys.ψ_m][end], 0.0, atol = 0.1)

    # Test 2: Wind speed increases with height (Eq. 16.61)
    z_low = 5.0
    z_high = 20.0

    prob_low = ODEProblem(csys, Dict(), (0.0, 1.0), Dict(
        csys.u_star => 0.3,
        csys.q_z => 1e-8,
        csys.z => z_low,
        csys.z₀ => 0.1,
        csys.T₀ => 288.15,
        csys.ρ => 1.225,
    ))
    sol_low = solve(prob_low)

    prob_high = ODEProblem(csys, Dict(), (0.0, 1.0), Dict(
        csys.u_star => 0.3,
        csys.q_z => 1e-8,
        csys.z => z_high,
        csys.z₀ => 0.1,
        csys.T₀ => 288.15,
        csys.ρ => 1.225,
    ))
    sol_high = solve(prob_high)

    @test sol_high[csys.ū][end] > sol_low[csys.ū][end]
end

# =============================================================================
# Physical Constraints Tests
# =============================================================================

@testitem "Physical constraints" setup = [LocalScaleSetup] tags = [:local_met] begin
    # Test 1: Potential temperature is always positive
    sys_stab = AtmosphericStability()
    csys_stab = mtkcompile(sys_stab)

    prob = ODEProblem(csys_stab, Dict(), (0.0, 1.0), Dict(
        csys_stab.T => 200.0,
        csys_stab.p => 50000.0,
    ))
    sol = solve(prob)
    @test sol[csys_stab.θ][end] > 0

    # Test 2: Wind speed should be non-negative for typical conditions
    sys_surf = SurfaceLayerProfile()
    csys_surf = mtkcompile(sys_surf)

    prob2 = ODEProblem(csys_surf, Dict(), (0.0, 1.0), Dict(
        csys_surf.u_star => 0.5,
        csys_surf.z => 10.0,
        csys_surf.z₀ => 0.1,
        csys_surf.q_z => 50.0,
        csys_surf.T₀ => 288.15,
        csys_surf.ρ => 1.225,
    ))
    sol2 = solve(prob2)
    @test sol2[csys_surf.ū][end] > 0

    # Test 3: Stability functions φ should be positive
    prob3 = ODEProblem(csys_surf, Dict(), (0.0, 1.0), Dict(
        csys_surf.u_star => 0.3,
        csys_surf.z => 10.0,
        csys_surf.q_z => 100.0,
        csys_surf.T₀ => 288.15,
        csys_surf.ρ => 1.225,
    ))
    sol3 = solve(prob3)
    @test sol3[csys_surf.φ_m][end] > 0
    @test sol3[csys_surf.φ_h][end] > 0
end

# =============================================================================
# Unit Verification Tests
# =============================================================================

@testitem "Unit consistency" setup = [LocalScaleSetup] tags = [:local_met] begin
    using ModelingToolkit: get_unit

    # Check AtmosphericStability units
    sys_stab = AtmosphericStability()
    for v in unknowns(sys_stab)
        name = string(Symbol(v))
        unit = get_unit(v)
        if name == "θ(t)"
            @test unit == u"K"
        elseif name == "dT_dz(t)" || name == "dθ_dz(t)" || name == "S(t)"
            @test unit == u"K/m"
        end
    end

    # Check SurfaceLayerProfile units
    sys_surf = SurfaceLayerProfile()
    for v in unknowns(sys_surf)
        name = string(Symbol(v))
        unit = get_unit(v)
        if name == "L(t)"
            @test unit == u"m"
        elseif name == "ū(t)"
            @test unit == u"m/s"
        end
    end

    # Check LocalScaleMeteorology own variables
    sys = LocalScaleMeteorology()
    for v in unknowns(sys)
        name = string(Symbol(v))
        unit = get_unit(v)
        if name == "L_inv(t)"
            @test unit == u"m^-1"
        end
    end
end
