import os
# CRITICAL: Use ctypes bindings to avoid nvJitLink errors
# This MUST be set BEFORE importing numba!
os.environ['NUMBA_CUDA_USE_NVIDIA_BINDING'] = '0'

import numpy as np
from numba import cuda, float64
import math
import time

def _candidate_block_sizes():
    """Return a standard set of warp-aligned candidates (filtered by device limits)."""
    dev = cuda.get_current_device()
    warp = dev.WARP_SIZE
    max_tb = dev.MAX_THREADS_PER_BLOCK
    base = [warp, 2*warp, 4*warp, 8*warp, 16*warp, 32*warp]  # up to 1024 on most GPUs
    return [b for b in base if b <= max_tb]

def suggest_blocksize_by_occupancy(kernel, dynamic_smem=0):
    """
    Try to get an occupancy-driven block size from Numba. Falls back to None
    if the API is unavailable or the kernel isn't compiled yet.
    """
    try:
        # Numba returns (min_grid_size, block_size)
        min_grid, occ_block = cuda.occupancy_max_potential_block_size(kernel, dynamic_smem=dynamic_smem)
        return occ_block
    except Exception:
        return None

def autotune_blocksize(kernel, args_builder, n_items, trial_steps, repeats=2, candidates=None):
    """
    Micro-benchmark the kernel for a small sample to pick the fastest block size.

    Parameters
    ----------
    kernel : numba.cuda.cudadrv.driver.AutoJitCUDAKernel
        Your CUDA kernel object (already created).
    args_builder : callable(sample_n, sample_steps) -> tuple
        Function that builds the *device* arguments tuple for the kernel call
        using `sample_n` items and `sample_steps` steps. It should allocate or slice
        device buffers once and reuse them across calls if possible.
    n_items : int
        Total number of trajectories (batch size).
    trial_steps : int
        Number of steps to use in the micro-benchmark (e.g. min(steps, 200)).
    repeats : int
        How many timing repeats per candidate.
    candidates : list[int] or None
        Optional list of candidate block sizes. If None, use warp-aligned defaults.

    Returns
    -------
    best_block : int
        Chosen threads-per-block.
    est_grid  : int
        Corresponding grid size for `n_items`.
    """
    dev = cuda.get_current_device()
    warp = dev.WARP_SIZE
    max_tb = dev.MAX_THREADS_PER_BLOCK

    if candidates is None:
        candidates = _candidate_block_sizes()
    else:
        # Keep warp-aligned values within the hardware limit
        candidates = [b for b in candidates if (b % warp == 0) and (b <= max_tb)]
        if not candidates:
            candidates = _candidate_block_sizes()

    # Try to include the occupancy suggestion if available
    try:
        min_grid, occ_block = cuda.occupancy_max_potential_block_size(kernel, dynamic_smem=0)
        if occ_block is not None and (occ_block % warp == 0) and (occ_block <= max_tb):
            if occ_block not in candidates:
                candidates.append(occ_block)
    except Exception:
        pass

    # Use a moderate sample size to keep tuning quick but representative
    sample_n = min(n_items, 16384)

    # Build device arguments once (your builder should allocate/slice device buffers)
    args = args_builder(sample_n, trial_steps)

    # Warm up (JIT-compile) on the first candidate
    warm_b = candidates[0]
    warm_g = (sample_n + warm_b - 1) // warm_b
    kernel[warm_g, warm_b](*args)
    cuda.synchronize()

    # Time each candidate
    best_block = warm_b
    best_ms = float("inf")

    for b in candidates:
        g = (sample_n + b - 1) // b

        # One extra warmup for fairness
        kernel[g, b](*args)
        cuda.synchronize()

        # GPU event-based timing
        acc_ms = 0.0
        for _ in range(repeats):
            start = cuda.event()
            stop  = cuda.event()
            start.record()
            kernel[g, b](*args)
            stop.record()
            stop.synchronize()
            acc_ms += cuda.event_elapsed_time(start, stop)  # milliseconds

        avg_ms = acc_ms / repeats
        if avg_ms < best_ms:
            best_ms = avg_ms
            best_block = b

    est_grid = (n_items + best_block - 1) // best_block
    return best_block, est_grid

# --------------------------
# Device gradients (examples)
# --------------------------
@cuda.jit(device=True, inline=True)
def grad_T_unit_mass(p, dTdp, params_T, dof: int):
    """dT/dp for T = 1/2 * sum_i p_i^2 (unit masses)."""
    for i in range(dof):
        dTdp[i] = p[i]  # dT/dp_i = p_i

@cuda.jit(device=True, inline=True)
def grad_V_duffing(q, dVdq, params_V, dof: int):
    """dV/dq for V = 1/2 * alpha * sum_i q_i^2 + 1/4 * beta * sum_i q_i^4 (Duffing oscillator)."""
    # params_V = (alpha, beta)
    alpha = params_V[0]
    beta = params_V[1]
    for i in range(dof):
        dVdq[i] = alpha * q[i] + beta * q[i] * q[i] * q[i]  # dV/dq_i = alpha*q_i + beta*q_i^3

# Set this to your intended DoF (must match the kernel factory argument)
DOF = 1

# -------------------------------------------------------
# Kernel factory: Yoshida-6 with on-the-fly LD accumulation
# -------------------------------------------------------
def create_yoshida6_symplectic_kernel(
    grad_T_device,
    grad_V_device,
    dof: int,
    solution: str = "A",
    ld_alpha: float = 1.0,
):
    """
    Creates a CUDA kernel that:
      • Integrates trajectories with a 6th-order Yoshida scheme for H = T(p) + V(q)
      • Accumulates a Lagrangian Descriptor (LD) per trajectory on GPU

    LD integrand (per default): sum_i ( |dq_i/dt|^alpha + |dp_i/dt|^alpha )
    evaluated at the start of each DRIFT sub-stage, weighted by that sub-stage's dt.

    Notes:
      - For separable H: dq/dt = ∂T/∂p = dT, dp/dt = -∂V/∂q = -dV.
      - We reuse the gradient evaluations already needed by the integrator whenever possible.
    """
    # ---- Compile-time "constants" captured in the kernel closure ----
    _DOF = dof
    _NVAR = 2 * dof
    _ALPHA = float(ld_alpha)

    # Yoshida Table-1 (6th order, m=3) coefficients w1,w2,w3; w0=1-2*(w1+w2+w3)
    sol = solution.upper()
    if sol == "A":
        w1, w2, w3 = (-1.17767998417887,  0.235573213359357,  0.784513610477560)
    elif sol == "B":
        w1, w2, w3 = (-2.13228522200144,  0.0426068187079180, 1.43984816797678)
    elif sol == "C":
        w1, w2, w3 = ( 0.0152886228424922, -2.14403531630539,  1.44778256239930)
    else:
        raise ValueError("solution must be one of {'A','B','C'}")

    _w1, _w2, _w3 = float(w1), float(w2), float(w3)

    @cuda.jit
    def kernel(Y0, t0, dt, params_T, params_V, Y_out, LD_out, steps):
        """
        CUDA kernel: Fixed-step Yoshida-6 symplectic integrator with LD accumulation.
        Each thread integrates one trajectory independently.
        Outputs:
          - Y_out[idx, :] = final [q | p]
          - LD_out[idx]   = accumulated LD over the full integration window
        """
        idx = cuda.grid(1)
        if idx >= Y0.shape[0]:
            return

        # ---- Local state: q, p and temporary work arrays ----
        # NOTE: local array sizes must be compile-time constants for CUDA.
        q   = cuda.local.array(DOF, float64)
        p   = cuda.local.array(DOF, float64)
        dT  = cuda.local.array(DOF, float64)  # dT/dp
        dV  = cuda.local.array(DOF, float64)  # dV/dq

        # Load initial condition into q|p
        for i in range(_DOF):
            q[i] = Y0[idx, i]
            p[i] = Y0[idx, _DOF + i]

        # Precompute composition coefficients (as scalars) inside the kernel
        w1 = _w1
        w2 = _w2
        w3 = _w3
        w0 = 1.0 - 2.0 * (w1 + w2 + w3)

        # Kick coefficients (8 terms): d1..d8
        d1 = w3; d2 = w2; d3 = w1; d4 = w0; d5 = w0; d6 = w1; d7 = w2; d8 = w3

        # Drift coefficients (9 terms): c1..c9  (sum c_i = 1)
        c1 = 0.5 * w3
        c2 = 0.5 * (w3 + w2)
        c3 = 0.5 * (w2 + w1)
        c4 = 0.5 * (w1 + w0)
        c5 = w0
        c6 = c4
        c7 = c3
        c8 = c2
        c9 = c1

        # Accumulator for the Lagrangian Descriptor
        LD = 0.0

        # Helper: accumulate LD from current (q,p) for a drift slice 'c*dt'
        # Uses phase-space L^alpha with alpha=_ALPHA.
        def accumulate_LD(c_factor):
            # Compute both gradients at *current* state
            grad_T_device(p, dT, params_T, _DOF)
            grad_V_device(q, dV, params_V, _DOF)
            acc = 0.0
            for j in range(_DOF):
                # |dq/dt| = |dT|, |dp/dt| = | - dV | = |dV|
                vq = dT[j]
                vp = dV[j]
                # Accumulate |vq|^alpha + |vp|^alpha
                # (alpha=1 => L1; alpha=0.5 => common LD choice in literature)
                if vq < 0.0:
                    vq = -vq
                if vp < 0.0:
                    vp = -vp
                acc += vq ** _ALPHA + vp ** _ALPHA
            return acc * (c_factor * dt)

        # ---------------- Main integration loop ----------------
        # Important: we accumulate LD *at the start of each drift sub-stage*
        # and weight by that drift's time-slice (c_k * dt). This yields a
        # consistent Riemann-sum approximation that converges as dt -> 0.
        for _ in range(steps):
            # --- Drift 1 ---
            LD += accumulate_LD(c1)
            # Use the dT computed in accumulate_LD? We re-call for clarity/correctness:
            grad_T_device(p, dT, params_T, _DOF)
            for i in range(_DOF):
                q[i] += (c1 * dt) * dT[i]

            # Kick 1
            grad_V_device(q, dV, params_V, _DOF)
            for i in range(_DOF):
                p[i] -= (d1 * dt) * dV[i]

            # --- Drift 2 ---
            LD += accumulate_LD(c2)
            grad_T_device(p, dT, params_T, _DOF)
            for i in range(_DOF):
                q[i] += (c2 * dt) * dT[i]

            # Kick 2
            grad_V_device(q, dV, params_V, _DOF)
            for i in range(_DOF):
                p[i] -= (d2 * dt) * dV[i]

            # --- Drift 3 ---
            LD += accumulate_LD(c3)
            grad_T_device(p, dT, params_T, _DOF)
            for i in range(_DOF):
                q[i] += (c3 * dt) * dT[i]

            # Kick 3
            grad_V_device(q, dV, params_V, _DOF)
            for i in range(_DOF):
                p[i] -= (d3 * dt) * dV[i]

            # --- Drift 4 ---
            LD += accumulate_LD(c4)
            grad_T_device(p, dT, params_T, _DOF)
            for i in range(_DOF):
                q[i] += (c4 * dt) * dT[i]

            # Kick 4
            grad_V_device(q, dV, params_V, _DOF)
            for i in range(_DOF):
                p[i] -= (d4 * dt) * dV[i]

            # --- Drift 5 ---
            LD += accumulate_LD(c5)
            grad_T_device(p, dT, params_T, _DOF)
            for i in range(_DOF):
                q[i] += (c5 * dt) * dT[i]

            # Kick 5
            grad_V_device(q, dV, params_V, _DOF)
            for i in range(_DOF):
                p[i] -= (d5 * dt) * dV[i]

            # --- Drift 6 ---
            LD += accumulate_LD(c6)
            grad_T_device(p, dT, params_T, _DOF)
            for i in range(_DOF):
                q[i] += (c6 * dt) * dT[i]

            # Kick 6
            grad_V_device(q, dV, params_V, _DOF)
            for i in range(_DOF):
                p[i] -= (d6 * dt) * dV[i]

            # --- Drift 7 ---
            LD += accumulate_LD(c7)
            grad_T_device(p, dT, params_T, _DOF)
            for i in range(_DOF):
                q[i] += (c7 * dt) * dT[i]

            # Kick 7
            grad_V_device(q, dV, params_V, _DOF)
            for i in range(_DOF):
                p[i] -= (d7 * dt) * dV[i]

            # --- Drift 8 ---
            LD += accumulate_LD(c8)
            grad_T_device(p, dT, params_T, _DOF)
            for i in range(_DOF):
                q[i] += (c8 * dt) * dT[i]

            # Kick 8
            grad_V_device(q, dV, params_V, _DOF)
            for i in range(_DOF):
                p[i] -= (d8 * dt) * dV[i]

            # --- Drift 9 (final) ---
            LD += accumulate_LD(c9)
            grad_T_device(p, dT, params_T, _DOF)
            for i in range(_DOF):
                q[i] += (c9 * dt) * dT[i]

        # Store final state back to global memory as [q | p]
        for i in range(_DOF):
            Y_out[idx, i]       = q[i]
            Y_out[idx, _DOF+i]  = p[i]

        # Store LD
        LD_out[idx] = LD

    return kernel


# --------------------------
# Main function
# --------------------------
def main():
    dof = 1 
    assert dof == DOF, "For CUDA local arrays, DOF must match the factory 'dof'."

    kernel = create_yoshida6_symplectic_kernel(
        grad_T_device=grad_T_unit_mass,
        grad_V_device=grad_V_duffing,  # Changed from grad_V_iso_harmonic
        dof=dof,
        solution="A",   # Yoshida Table-1 set
        ld_alpha=1.0,   # 1.0 => phase-space L1; try 0.5 for a common LD choice
    )

    # Batch of ICs: [q0_x, q0_y, p0_x, p0_y]
    """
    num_ics = 1000
    Y0_h = np.zeros((num_ics, 2*dof), dtype=np.float64)
    # Example: random q in [-1,1], p in [-0.5,0.5]
    rng = np.random.default_rng(1)
    Y0_h[:, 0:dof] = rng.uniform(-1.0, 1.0, size=(num_ics, dof))
    Y0_h[:, dof: ] = rng.uniform(-0.5, 0.5, size=(num_ics, dof))
    """

    # regular grid of 100x100 points in the q-p plane
    q_grid = np.linspace(-1.5, 1.5, 100)
    p_grid = np.linspace(-1.0, 1.0, 100)
    Q_grid, P_grid = np.meshgrid(q_grid, p_grid)
    num_ics = Q_grid.size  # Add this line
    Y0_h = np.zeros((num_ics, 2*dof), dtype=np.float64)
    Y0_h[:, 0:dof] = Q_grid.flatten().reshape(-1, dof)  # Reshape to (N, dof)
    Y0_h[:, dof: ] = P_grid.flatten().reshape(-1, dof)  # Reshape to (N, dof)

    # GPU buffers
    print("  Transferring ICs to GPU..."); Y0_d = cuda.to_device(Y0_h)
    print(f"  Allocating outputs on GPU (state: {num_ics}x{2*dof}, LD: {num_ics})...")
    Yout_d = cuda.device_array_like(Y0_d)
    LD_d   = cuda.device_array((num_ics,), dtype=np.float64)

    # Integration params
    t0      = 0.0
    dt      = 0.01
    t_final = 20.0     # NOTE: shorter default to test quickly; raise to 1000.0 as needed
    steps   = int(t_final / dt)
    params_T = ()         # none for unit-mass kinetic
    # Duffing parameters: (alpha, beta)
    # Common choices: alpha = -1.0, beta = 1.0 (double-well potential)
    #                 alpha = 1.0, beta = 1.0 (hardening spring)
    params_V = (-1.0, 1.0)  # Changed from (1.0,) - now (alpha, beta)

    def args_builder(sample_n, sample_steps):
        """Build kernel args for a sample run (re-use big buffers by slicing)."""
        # IMPORTANT: keep shapes consistent with the kernel signature
        return (
            Y0_d[:sample_n],   # Y0
            t0,                # t0
            dt,                # dt
            params_T,          # params_T
            params_V,          # params_V
            Yout_d[:sample_n], # Y_out
            LD_d[:sample_n],   # LD_out
            sample_steps       # steps
        )

    # Choose trial steps for tuning (keep it small but not too small)
    trial_steps = min(steps, 200)

    # 1) Quick occupancy-only suggestion (no timing)
    occ_block = suggest_blocksize_by_occupancy(kernel)

    # 2) Micro-benchmark tuning (recommended). You can pass custom candidates if you like.
    best_block, best_grid = autotune_blocksize(
        kernel, args_builder, n_items=num_ics, trial_steps=trial_steps, repeats=2,
        candidates=[64, 128, 256, 512, 1024]
    )

    print(f"  Occupancy-suggested block: {occ_block}")
    print(f"  Autotuned threads_per_block: {best_block}, grid: {best_grid}")

    # Use the tuned config for the full run
    threads_per_block = best_block
    blocks = (num_ics + threads_per_block - 1) // threads_per_block
    print(f"  Using: Grid={blocks}, Block={threads_per_block} for production run")

    # Launch production kernel with timing
    print("  Launching kernel...")
    start_time = time.time()
    kernel[blocks, threads_per_block](Y0_d, t0, dt, params_T, params_V, Yout_d, LD_d, steps)
    cuda.synchronize()  # Wait for GPU to finish
    kernel_time = time.time() - start_time
    print(f"  CUDA Launch Config: Grid={blocks}, Block={threads_per_block}, Steps={steps}")
    print(f"  Kernel execution time: {kernel_time:.4f} seconds")
    print(f"  Time per trajectory: {kernel_time/num_ics:.6e} seconds")
    print(f"  Trajectories per second: {num_ics/kernel_time:.2f}")

    """
    print("  Launching kernel...")
    kernel[blocks, threads_per_block](Y0_d, t0, dt, params_T, params_V, Yout_d, LD_d, steps)
    cuda.synchronize()
    print("  Kernel execution finished.")
    """

    # Fetch results
    print("  Copying results back to host...")
    copy_start = time.time()
    Yf_h = Yout_d.copy_to_host()
    LD_h = LD_d.copy_to_host()
    copy_time = time.time() - copy_start
    print(f"  Data copy time: {copy_time:.4f} seconds")

    # ---------- Diagnostics: energy conservation ----------
    alpha = params_V[0]
    beta = params_V[1]
    q0_all = Y0_h[:, :dof]; p0_all = Y0_h[:, dof:]
    qf_all = Yf_h[:, :dof]; pf_all = Yf_h[:, dof:]
    # Energy: E = 1/2 * sum(p^2) + 1/2 * alpha * sum(q^2) + 1/4 * beta * sum(q^4)
    E0_all = 0.5 * np.sum(p0_all**2, axis=1) + 0.5 * alpha * np.sum(q0_all**2, axis=1) + 0.25 * beta * np.sum(q0_all**4, axis=1)
    Ef_all = 0.5 * np.sum(pf_all**2, axis=1) + 0.5 * alpha * np.sum(qf_all**2, axis=1) + 0.25 * beta * np.sum(qf_all**4, axis=1)
    energy_diffs = np.abs(Ef_all - E0_all)
    max_energy_diff = float(np.max(energy_diffs))
    print(f"\nMax energy error over {num_ics} trajectories: {max_energy_diff:.3e}")

    # ---------- Diagnostics: LD summary ----------
    print(f"LD stats  (alpha = {1.0:.3g}): "
          f"min={float(LD_h.min()):.6g}, "
          f"median={float(np.median(LD_h)):.6g}, "
          f"mean={float(LD_h.mean()):.6g}, "
          f"max={float(LD_h.max()):.6g}")

    # Quick energy for first trajectory
    def energy(q, p, alpha=alpha, beta=beta):
        return 0.5*np.sum(p*p) + 0.5*alpha*np.sum(q*q) + 0.25*beta*np.sum(q*q*q*q)
    qf = Yf_h[0, :dof]; pf = Yf_h[0, dof:]
    qi = Y0_h[0, :dof]; pi = Y0_h[0, dof:]
    print("E0 =", energy(qi, pi), " Ef =", energy(qf, pf),
          " Error =", np.abs(energy(qf, pf) - energy(qi, pi)))

    # Example: print first 5 LD values
    print("\nExample LD (first 5 trajectories):", LD_h[:5])


# --------------------------
# Entry point
# --------------------------
if __name__ == "__main__":
    if not cuda.is_available():
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! CUDA is not available or not detected! !!!")
        print("!!! This script requires a CUDA-enabled GPU!!!")
        print("!!! and correctly installed CUDA drivers   !!!")
        print("!!! and Numba.                              !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        raise SystemExit(1)
    else:
        try:
            dev_name = str(cuda.get_current_device().name)
        except Exception:
            dev_name = "<unknown device>"
        print(f"Found CUDA device: {dev_name}")
        main()
