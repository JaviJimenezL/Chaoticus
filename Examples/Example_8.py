import os
# CRITICAL: Use ctypes bindings to avoid nvJitLink errors
# This MUST be set BEFORE importing numba!
os.environ['NUMBA_CUDA_USE_NVIDIA_BINDING'] = '0'

import numpy as np
from numba import cuda, float64

# --------------------------
# Device gradients (examples)
# --------------------------
@cuda.jit(device=True, inline=True)
def grad_T_unit_mass(p, dTdp, params_T, dof: int):
    """dT/dp for T = 1/2 * sum_i p_i^2 (unit masses)."""
    for i in range(dof):  # Use array shape
        dTdp[i] = p[i]  # dT/dp_i = p_i

@cuda.jit(device=True, inline=True)
def grad_V_iso_harmonic(q, dVdq, params_V, dof: int):
    """dV/dq for V = 1/2 * omega^2 * sum_i q_i^2 (isotropic)."""
    # omega = params_V[0]  # scalar frequency
    omega = 1.0
    w2 = omega * omega
    for i in range(dof):  # Use array shape
        dVdq[i] = w2 * q[i]

DOF = 2

# Kernel
def create_yoshida6_symplectic_kernel(grad_T_device, grad_V_device, dof: int, solution: str = "A"):
    """
    Factory that creates a Numba CUDA kernel for a fixed-step
    6th-order Yoshida symplectic integrator (for separable H = T(p) + V(q)).
    """
    # ---- Compile-time "constants" captured in the kernel closure ----
    _DOF = dof
    _NVAR = 2 * dof

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

    # Capture floats for the kernel
    _w1, _w2, _w3 = float(w1), float(w2), float(w3)

    @cuda.jit
    def kernel(Y0, t0, dt, params_T, params_V, Y_out, steps):
        """
        CUDA kernel: Fixed-step Yoshida-6 symplectic integrator.
        Each thread integrates one trajectory independently.
        """
        idx = cuda.grid(1)
        if idx >= Y0.shape[0]:
            return

        # ---- Local state: q, p and temporary work arrays ----
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

        # Drift coefficients (9 terms): c1..c9
        c1 = 0.5 * w3
        c2 = 0.5 * (w3 + w2)
        c3 = 0.5 * (w2 + w1)
        c4 = 0.5 * (w1 + w0)
        c5 = w0
        c6 = c4  # symmetry
        c7 = c3
        c8 = c2
        c9 = c1

        t = t0  # kept for API compatibility; not used (autonomous Hamiltonian)

        # ---------------- Main integration loop ----------------
        for _ in range(steps):
            # Drift 1
            grad_T_device(p, dT, params_T, _DOF)
            for i in range(_DOF):
                q[i] += (c1 * dt) * dT[i]

            # (Kick 1) + Drift 2
            grad_V_device(q, dV, params_V, _DOF)
            for i in range(_DOF):
                p[i] -= (d1 * dt) * dV[i]
            grad_T_device(p, dT, params_T, _DOF)
            for i in range(_DOF):
                q[i] += (c2 * dt) * dT[i]

            # (Kick 2) + Drift 3
            grad_V_device(q, dV, params_V, _DOF)
            for i in range(_DOF):
                p[i] -= (d2 * dt) * dV[i]
            grad_T_device(p, dT, params_T, _DOF)
            for i in range(_DOF):
                q[i] += (c3 * dt) * dT[i]

            # (Kick 3) + Drift 4
            grad_V_device(q, dV, params_V, _DOF)
            for i in range(_DOF):
                p[i] -= (d3 * dt) * dV[i]
            grad_T_device(p, dT, params_T, _DOF)
            for i in range(_DOF):
                q[i] += (c4 * dt) * dT[i]

            # (Kick 4) + Drift 5
            grad_V_device(q, dV, params_V, _DOF)
            for i in range(_DOF):
                p[i] -= (d4 * dt) * dV[i]
            grad_T_device(p, dT, params_T, _DOF)
            for i in range(_DOF):
                q[i] += (c5 * dt) * dT[i]

            # (Kick 5) + Drift 6
            grad_V_device(q, dV, params_V, _DOF)
            for i in range(_DOF):
                p[i] -= (d5 * dt) * dV[i]
            grad_T_device(p, dT, params_T, _DOF)
            for i in range(_DOF):
                q[i] += (c6 * dt) * dT[i]

            # (Kick 6) + Drift 7
            grad_V_device(q, dV, params_V, _DOF)
            for i in range(_DOF):
                p[i] -= (d6 * dt) * dV[i]
            grad_T_device(p, dT, params_T, _DOF)
            for i in range(_DOF):
                q[i] += (c7 * dt) * dT[i]

            # (Kick 7) + Drift 8
            grad_V_device(q, dV, params_V, _DOF)
            for i in range(_DOF):
                p[i] -= (d7 * dt) * dV[i]
            grad_T_device(p, dT, params_T, _DOF)
            for i in range(_DOF):
                q[i] += (c8 * dt) * dT[i]

            # (Kick 8) + Drift 9 (final drift)
            grad_V_device(q, dV, params_V, _DOF)
            for i in range(_DOF):
                p[i] -= (d8 * dt) * dV[i]
            grad_T_device(p, dT, params_T, _DOF)
            for i in range(_DOF):
                q[i] += (c9 * dt) * dT[i]

        # Store final state back to global memory as [q | p]
        for i in range(_DOF):
            Y_out[idx, i]       = q[i]
            Y_out[idx, _DOF+i]  = p[i]

    return kernel


# --------------------------
# Main function (following Example_4 pattern)
# --------------------------
def main():
    dof = 2  # e.g., 2D oscillator -> state len = 4
    kernel = create_yoshida6_symplectic_kernel(
        grad_T_device=grad_T_unit_mass,
        grad_V_device=grad_V_iso_harmonic,
        dof=dof,
        solution="A",  # Yoshida Table-1 set
    )

    # Batch of ICs: [q0_x, q0_y, p0_x, p0_y]
    num_ics = 1000
    Y0_h = np.zeros((num_ics, 2*dof), dtype=np.float64)
    # Example: random q in [-1,1], p in [-0.5,0.5]
    rng = np.random.default_rng(1)
    Y0_h[:, 0:dof] = rng.uniform(-1.0, 1.0, size=(num_ics, dof))
    Y0_h[:, dof: ] = rng.uniform(-0.5, 0.5, size=(num_ics, dof))

    # GPU buffers - exactly like Example_4
    print("  Transferring ICs to GPU..."); Y0_d = cuda.to_device(Y0_h)
    print(f"  Allocating output on GPU ({num_ics}x{2*dof})...")
    Yout_d = cuda.device_array_like(Y0_d)

    # Integration params
    t0     = 0.0
    dt     = 0.001
    t_final = 1000.0
    steps = int(t_final / dt)
    params_T = ()         # none for unit-mass kinetic
    params_V = (1.0,)     # omega = 1

    # Launch
    threads_per_block = 256
    blocks = (num_ics + threads_per_block - 1) // threads_per_block
    print(f"  CUDA Launch Config: Grid={blocks}, Block={threads_per_block}")

    print("  Launching kernel...")
    kernel[blocks, threads_per_block](Y0_d, t0, dt, params_T, params_V, Yout_d, steps)
    cuda.synchronize()  # Wait for the kernel to complete
    print("  Kernel execution finished.")

    # Fetch results
    print("  Copying results back to host...")
    Yf_h = Yout_d.copy_to_host()

    # Compute energy conservation for all trajectories (vectorized)
    omega = 1.0
    omega2 = omega * omega
    
    # Extract q and p for all trajectories
    q0_all = Y0_h[:, :dof]
    p0_all = Y0_h[:, dof:]
    qf_all = Yf_h[:, :dof]
    pf_all = Yf_h[:, dof:]
    
    # Compute energies vectorized
    E0_all = 0.5 * np.sum(p0_all**2, axis=1) + 0.5 * omega2 * np.sum(q0_all**2, axis=1)
    Ef_all = 0.5 * np.sum(pf_all**2, axis=1) + 0.5 * omega2 * np.sum(qf_all**2, axis=1)
    
    # Calculate energy differences (absolute)
    energy_diffs = np.abs(Ef_all - E0_all)
    
    # Find and print maximum energy conservation difference
    max_energy_diff = np.max(energy_diffs)
    print(f"\nMaximum energy conservation difference among all {num_ics} trajectories: {max_energy_diff:.10e}")

    # (Optional) quick energy check on host for one trajectory
    def energy(q, p, omega=1.0):
        return 0.5*np.sum(p*p) + 0.5*(omega**2)*np.sum(q*q)
    qf = Yf_h[0, :dof]; pf = Yf_h[0, dof:]
    qi = Y0_h[0, :dof]; pi = Y0_h[0, dof:]
    print("E0 =", energy(qi, pi), " Ef =", energy(qf, pf), " Error =", np.abs(energy(qf, pf) - energy(qi, pi)))


# --------------------------
# Call main function (following Example_4 pattern exactly)
# --------------------------
if __name__ == "__main__":
    # Check for CUDA availability
    if not cuda.is_available():
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! CUDA is not available or not detected! !!!")
        print("!!! This script requires a CUDA-enabled GPU!!!")
        print("!!! and correctly installed CUDA drivers   !!!")
        print("!!! and Numba.                           !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        exit(1)  # Exit if no CUDA
    else:
        print(f"Found CUDA device: {cuda.get_current_device().name.decode()}")
        main()  # Run the main function
