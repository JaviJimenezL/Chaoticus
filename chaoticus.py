"""
Chaoticus is a package that allows the integration of dynamical systems using parallel computing and the obtention
of the Lyapunov spectrum, the GALI indicator, the SALI indicator and the chaos indicators derived from Lagrangian Descriptors
"""

"""
Try to import the necessary external libraries
"""
import numpy as np
import traceback
import collections.abc
import numbers
import warnings 

try:
    from numba import cuda, float64
    _NUMBA_AVAILABLE = True # Set to True if import succeeds
except ImportError:
    print("Warning: Numba not found. PCA will not work properly.")
    _NUMBA_AVAILABLE = False # Set to False if import fails
    cuda = None # Set cuda to None to avoid later errors if used without checking
    float64 = None # Set float64 to None to avoid later errors if used without checking

import math

try:
    import cupy as cp
    _CUPY_AVAILABLE = True # Set to True if import succeeds
except ImportError:
    print("Warning: CuPy not found. calc_GALI function will not work properly.")
    _CUPY_AVAILABLE = False # Set to False if import fails
    cp = None # Set cp to None to avoid later errors if used without checking

# =========================================
# Neighbor Generation Function
# =========================================
def neigh_gen(ic: np.ndarray, dims_to_perturb: list[int], d: float = 1e-4) -> np.ndarray:
    """
    Generates neighboring initial conditions by perturbing specified dimensions.

    For each dimension index provided in `dims_to_perturb`, it creates two
    neighboring initial conditions: one by adding 'd' to that dimension
    and one by subtracting 'd'.

    Args:
        ic: The initial condition vector (1D NumPy array). Must contain numeric types.
        dims_to_perturb: A sequence (list, tuple, etc.) of integer indices
                         indicating which dimensions of the initial condition
                         vector to perturb.
        d: The small perturbation amount. Must be a number (int or float).
           Defaults to 1e-4.

    Returns:
        A 2D NumPy array where each row is a neighboring initial condition.
        The shape will be (2 * len(dims_to_perturb), len(ic)).
        The dtype will match the dtype of the input 'ic'.
        Neighbors are ordered in pairs: [ic_dim0_plus, ic_dim0_minus,
                                       ic_dim1_plus, ic_dim1_minus, ...].
        Returns an empty array with shape (0, len(ic)) if dims_to_perturb is empty.

    Raises:
        TypeError: If 'ic' is not a NumPy array.
        ValueError: If 'ic' is not a 1D NumPy array.
        ValueError: If 'ic' does not have a numeric dtype.
        TypeError: If 'dims_to_perturb' is not a sequence (like list or tuple).
        TypeError: If any element in 'dims_to_perturb' is not an integer.
        IndexError: If any index in 'dims_to_perturb' is out of bounds for 'ic'.
        TypeError: If 'd' is not a number (int or float).
    """
    # 1. Validate 'ic'
    if not isinstance(ic, np.ndarray):
        raise TypeError(f"Input 'ic' must be a NumPy array, but got type {type(ic).__name__}.")
    if ic.ndim != 1:
        raise ValueError(f"Input 'ic' must be a 1D NumPy array, but got shape {ic.shape}.")
    if not np.issubdtype(ic.dtype, np.number):
        # Ensure the array contains numbers for perturbation arithmetic
        raise ValueError(f"Input 'ic' must have a numeric dtype, but got {ic.dtype}.")

    num_dims = len(ic) # Get dimension size only after validation

    # 2. Validate 'd'
    if not isinstance(d, (int, float)):
        raise TypeError(f"Perturbation amount 'd' must be a number (int or float), "
                        f"but got type {type(d).__name__}.")
    # Convert d to float for consistent arithmetic, respecting ic's dtype later
    d_float = float(d)

    # 3. Validate 'dims_to_perturb'
    if not isinstance(dims_to_perturb, collections.abc.Sequence):
         # Check if it's a sequence (list, tuple, etc.) but not a string
         # Allows iterables like range() but prevents strings.
         # Using Sequence avoids consuming generators if we needed to iterate twice.
         raise TypeError(f"'dims_to_perturb' must be a sequence (e.g., list, tuple), "
                         f"but got type {type(dims_to_perturb).__name__}.")

    # Check elements within dims_to_perturb *before* the main loop for efficiency
    # and better error reporting if multiple issues exist.
    invalid_indices = []
    non_int_indices = []
    for i in dims_to_perturb:
        if not isinstance(i, int) or isinstance(i, bool): # bool is subclass of int, exclude it
            non_int_indices.append(repr(i)) # Use repr for clarity
        elif not 0 <= i < num_dims:
            invalid_indices.append(i)

    if non_int_indices:
        raise TypeError(f"All elements in 'dims_to_perturb' must be integers. "
                        f"Found non-integers: {', '.join(non_int_indices)}")
    if invalid_indices:
        # Sort for clearer reporting if many indices are out of bounds
        invalid_indices.sort()
        raise IndexError(f"Dimension indices in 'dims_to_perturb' are out of bounds. "
                         f"Valid range is [0, {num_dims - 1}]. "
                         f"Invalid indices found: {invalid_indices}")

    # --- Main Logic ---
    # Pre-allocate array for potential minor performance benefit if list gets large
    # Use the dtype of the input array 'ic' for the output
    neighbors_array = np.empty((2 * len(dims_to_perturb), num_dims), dtype=ic.dtype)
    row_index = 0

    for i in dims_to_perturb:
        # Perturbation (no need for index checks here anymore)
        # Add/subtract d_float, result automatically takes ic's dtype if needed
        # Example: if ic is int and d is float, result is float.
        # If ic is float, result is float. If ic is int and d makes it non-int,
        # result stays int if d is int, or becomes float if d is float.
        # Using copy() is essential.
        state_p = ic.copy()
        state_p[i] += d_float
        neighbors_array[row_index] = state_p

        state_m = ic.copy()
        state_m[i] -= d_float
        neighbors_array[row_index + 1] = state_m

        row_index += 2

    # The case where dims_to_perturb is empty is handled automatically by
    # np.empty((0, num_dims), ...) creating the correct shape.

    return neighbors_array

# =========================================
# Random Vector in Hypersphere Function
# =========================================
def random_vector_in_hypersphere(dim: int, max_norm: float) -> np.ndarray:
    """
    Generates a random vector in a hypersphere of a given dimension and maximum norm.

    Args:
        dim: The dimension of the hypersphere.
        max_norm: The maximum norm of the vector.

    Returns:
        The random vector in the hypersphere (1D NumPy array).

    Raises:
        ValueError: If max_norm is negative.
    """
    if max_norm < 0:
        raise ValueError("max_norm must be non-negative.")
    if max_norm == 0:
        return np.zeros(dim)
    direction = np.random.randn(dim)
    direction_norm = np.linalg.norm(direction)
    if direction_norm == 0:
        return random_vector_in_hypersphere(dim, max_norm)
    unit_direction = direction / direction_norm
    u = np.random.rand()
    radius = max_norm * (u ** (1.0 / dim))
    vector = radius * unit_direction
    return vector

# =========================================
# Perturb Initial Condition Function
# =========================================
def perturb_ic(ic: np.ndarray,
               max_norm: float,
               dims_to_perturb: list[int] | tuple[int, ...] | None = None
               ) -> np.ndarray:
    """
    Perturbs an initial condition by a random vector in a hypersphere.

    Args:
        ic: The initial condition vector (1D NumPy array).
        max_norm: The maximum norm of the vector.
        dims_to_perturb: A sequence (list, tuple, etc.) of integer indices
                         indicating which dimensions of the initial condition
                         vector to perturb.

    Returns:
        The perturbed initial condition vector (1D NumPy array).

    Raises:
        ValueError: If max_norm is negative.
    """
    if max_norm < 0:
        raise ValueError("max_norm must be non-negative.")
    original_dim = len(ic)
    if dims_to_perturb is None:
        dims_to_perturb = list(range(original_dim))
    elif not dims_to_perturb:
        return ic.copy()
    if any(idx < 0 or idx >= original_dim for idx in dims_to_perturb):
         raise IndexError(f"dims_to_perturb contains indices out of bounds for ic of length {original_dim}")
    perturb_dim = len(dims_to_perturb)
    perturbation_values = random_vector_in_hypersphere(perturb_dim, max_norm)
    full_perturbation = np.zeros_like(ic, dtype=float)
    full_perturbation[list(dims_to_perturb)] = perturbation_values
    neighbor_ic = ic + full_perturbation
    return neighbor_ic

# =========================================
# Chaos Indicators based on LD Function
# =========================================
def chaos_indicators(
    L: float,
    L_neighbors: collections.abc.Sequence[float],
    d: float
) -> tuple[float, float, float, float]:
    """
    Calculates chaos indicators (D, R, C, S) derived from Lagrangian Descriptors (LDs).

    Assumes L_neighbors contains LD values from pairs of neighboring trajectories,
    where each pair corresponds to a positive and negative perturbation 'd' along
    a specific initial condition dimension.

    Args:
        L: The Lagrangian Descriptor value (a number) from the central trajectory.
        L_neighbors: A sequence (list, tuple, 1D NumPy array, etc.) containing
                     the numeric LD values from the neighboring trajectories.
                     Must contain an even number of elements, ordered in pairs
                     corresponding to the (+d, -d) perturbations for each
                     dimension perturbed (e.g., [L_dim0_plus, L_dim0_minus, ...]).
                     Cannot be empty.
        d: The perturbation distance (a positive number) used to generate the
           neighboring initial conditions.

    Returns:
        A tuple containing the four chaos indicators: (D, R, C, S).
            - D: Dispersion
            - R: Renormalization
            - C: Curvature
            - S: Smoothness
        Returns (NaN, NaN, C, S) if L is zero, along with a warning, as D and R
        involve division by L.

    Raises:
        TypeError: If L is not a number.
        TypeError: If d is not a number.
        ValueError: If d is not positive.
        TypeError: If L_neighbors is not a sequence or cannot be converted to a
                   numeric NumPy array.
        ValueError: If L_neighbors results in a non-1D array after conversion.
        ValueError: If L_neighbors is empty.
        ValueError: If L_neighbors has an odd number of elements.
    """
    # 1. Validate 'L'
    if not isinstance(L, numbers.Number):
        raise TypeError(f"Input 'L' must be a number, but got type {type(L).__name__}.")
    # Convert L to float for consistent calculations
    L = float(L)

    # 2. Validate 'd'
    if not isinstance(d, numbers.Number):
        raise TypeError(f"Input 'd' must be a number, but got type {type(d).__name__}.")
    if d <= 0:
        raise ValueError(f"Perturbation distance 'd' must be positive, but got {d}.")
    # Convert d to float
    d = float(d)

    # 3. Validate 'L_neighbors' (Type, Convertibility, Shape, Size)
    if not isinstance(L_neighbors, collections.abc.Sequence):
         # Check if it's a sequence (list, tuple, etc.) but not a string
         raise TypeError(f"'L_neighbors' must be a sequence (e.g., list, tuple), "
                         f"but got type {type(L_neighbors).__name__}.")

    try:
        # Attempt conversion to a float NumPy array.
        # This also implicitly checks if elements are numeric.
        L_neighbors_arr = np.asarray(L_neighbors, dtype=float)
    except (ValueError, TypeError) as e:
        # Catch errors during conversion (e.g., contains strings)
        raise TypeError(f"Elements in 'L_neighbors' must be numeric and convertible "
                        f"to a NumPy array. Conversion failed: {e}")

    # Ensure it's effectively 1D after conversion
    if L_neighbors_arr.ndim != 1:
        raise ValueError(f"'L_neighbors' must represent a 1D sequence, but resulted "
                         f"in array with shape {L_neighbors_arr.shape} after conversion.")

    # Check for empty after potential conversion
    if L_neighbors_arr.size == 0:
         raise ValueError("'L_neighbors' cannot be empty.") # Changed from returning NaN

    # Check for even number of elements
    if L_neighbors_arr.size % 2 != 0:
        raise ValueError(f"'L_neighbors' must have an even number of elements "
                         f"(pairs of neighbors). Got size {L_neighbors_arr.size}.")

    # --- Checks complete, proceed with calculation ---

    n = L_neighbors_arr.size // 2 # Number of dimensions perturbed

    # Separate the neighbors corresponding to +d and -d perturbations
    # Slicing is efficient on NumPy arrays
    L1 = L_neighbors_arr[0::2] # Neighbors from +d perturbation
    L2 = L_neighbors_arr[1::2] # Neighbors from -d perturbation

    # --- Calculate intermediate values based on finite differences ---
    # These calculations are safe regarding division by zero
    aux_array_D = np.abs(L - L1) + np.abs(L - L2)
    aux_array_R = L1 + L2
    # d is guaranteed > 0 here
    aux_array_C = np.abs(L1 - L2) / d
    aux_array_S = np.abs(L1 - 2 * L + L2) / (d**2)

    # --- Calculate final indicators ---
    # Use np.sum() for clarity and correctness with array inputs
    sum_aux_D = np.sum(aux_array_D)
    sum_aux_R = np.sum(aux_array_R)
    sum_aux_C = np.sum(aux_array_C)
    sum_aux_S = np.sum(aux_array_S)

    # Handle division by n (guaranteed > 0) and L (check for L == 0)
    C = sum_aux_C / (2 * n)
    S = sum_aux_S / n

    if np.isclose(L, 0.0): # Use np.isclose for float comparison
        D = np.nan
        R = np.nan
        warnings.warn("Input 'L' is close to zero. Indicators D and R involve division by L "
                      "and are returned as NaN.", RuntimeWarning)
    else:
        D = sum_aux_D / (2 * n * L)
        # Calculate R = |1 - sum / (2 * n * L)|
        R = np.abs(1.0 - sum_aux_R / (2 * n * L))


    return D, R, C, S

# =========================================
# GALI Calculation Function (using CuPy)
# =========================================
def calc_GALI(dev_vectors_2d) -> float:
    """
    Calculates the Generalized Alignment Index (GALI_k) using CuPy SVD.

    GALI_k measures the volume spanned by k deviation vectors. A value near
    zero indicates linear dependence (alignment) among the vectors, often
    associated with regular (non-chaotic) dynamics in that subspace.
    This function computes the product of the singular values of the matrix
    formed by the k (unnormalized) deviation vectors provided as its *rows*.
    This product is proportional to the k-dimensional volume spanned by these
    vectors in the n-dimensional phase space.

    Requires the CuPy library to be installed and a compatible GPU environment.

    Args:
        dev_vectors_2d (array-like): A 2D array-like object (e.g., NumPy array,
                                    CuPy array) where each *row* is a deviation
                                    vector. The shape should be (k, n), where k
                                    is the GALI order (number of vectors, M) and n
                                    is the dimension of the phase space (e.g., 2*N).
                                    Requires k <= n for a meaningful volume calculation.

    Returns:
        float: The calculated GALI value (product of singular values), transferred
               back to the CPU. Returns 1.0 if k=0 (convention), 0.0 if k > n
               (linearly dependent), or np.nan if an error occurs during calculation.

    Raises:
        ImportError: If CuPy is required but not installed or cannot be imported.
        ValueError: If the input is not a 2D array, k > n, or contains non-finite values.
        cp.cuda.CuPyError: For CuPy specific errors during GPU execution (e.g., SVD failure).
                           (These are caught internally and NaN is returned, but the error
                            is printed with traceback for debugging).
    """
    # This comprehensive try-except catches errors during execution
    try:
        if not _CUPY_AVAILABLE:
            raise ImportError("CuPy required for calc_GALI but not available.")

        # Check input type right before using cp
        if not isinstance(dev_vectors_2d, (np.ndarray, cp.ndarray)):
             # Attempt conversion if not already numpy or cupy array
             print(f"Warning: Input to calc_GALI was not np/cp array (type: {type(dev_vectors_2d)}). Converting.")
             dev_vectors_2d = np.asarray(dev_vectors_2d, dtype=np.float64)

        # Check for non-finite values BEFORE passing to CuPy
        if not np.all(np.isfinite(dev_vectors_2d)):
            raise ValueError("Input matrix contains non-finite values (NaN or Inf)")

        # --- Attempt CuPy operations ---
        A_cupy = cp.asarray(dev_vectors_2d) # Transfer to GPU if NumPy

        if A_cupy.ndim != 2: raise ValueError(f"Input must be 2D, got ndim={A_cupy.ndim}")
        k, n = A_cupy.shape # k=num_vectors, n=phase_space_dim
        if k == 0: return np.nan # GALI_0 = 1 convention
        if k > n: return 0.0 # Vectors must be linearly dependent

        # Perform SVD - only singular values needed
        s = cp.linalg.svd(A_cupy, full_matrices=False, compute_uv=False)

        # Calculate GALI = product of singular values
        gali_val_cp = cp.prod(s)
        gali_val_float = gali_val_cp.get() # Get result back to CPU
        # --- End CuPy operations ---

        # Check final result validity
        if not np.isfinite(gali_val_float):
            print(f"Warning: GALI calc resulted in non-finite value ({gali_val_float}). Shape {k}x{n}.")
            return np.nan

        return float(gali_val_float)
    
    except ImportError as e:
        # Specifically handle missing CuPy
        print(f"\n--- IMPORT ERROR inside calc_GALI ---")
        print(f" {e}")
        print(" Ensure CuPy is installed correctly in the environment.")
        print(f"-----------------------------------\n")
        raise # Re-raise the import error to stop execution if CuPy is mandatory

    except Exception as e:
        # Catch other errors (like potential CuPy errors during asarray or svd)
        print(f"\n--- ERROR inside calc_GALI ---")
        print(f" Exception Type: {type(e)}")
        print(f" Exception Args: {e.args}")
        print("\n --- Traceback ---")
        traceback.print_exc()
        print(f"--------------------\n")
        return np.nan # Return NaN after printing erro

# =========================================
# SALI Calculation Function
# =========================================
def calc_SALI(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculates the standard Smaller Alignment Index (SALI) for two vectors.

    SALI = min(||v1_hat + v2_hat||, ||v1_hat - v2_hat||), where v_hat are
    normalized vectors. 

    Args:
        v1: The first deviation vector (1D array-like, numeric).
        v2: The second deviation vector (1D array-like, numeric).

    Returns:
        float: The standard SALI value.
               Returns np.nan if either input vector has a norm very close to zero
               (below internal tolerance `_SALI_ZERO_TOLERANCE`), as SALI is
               undefined or trivially zero in this case.

    Raises:
        TypeError: If inputs `v1` or `v2` cannot be converted to numeric
                   NumPy arrays.
        ValueError: If input vectors do not have the same shape.
        ValueError: If input vectors are not 1D.
    """
    # 1. Convert to NumPy arrays and validate type
    try:
        # Use float64 for precision in norm calculations
        v1_arr = np.asarray(v1, dtype=np.float64)
        v2_arr = np.asarray(v2, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise TypeError(f"Input vectors v1 and v2 must be array-like and contain numeric types. "
                        f"Conversion failed: {e}")

    # 2. Validate shape and dimension
    if v1_arr.shape != v2_arr.shape:
        raise ValueError(f"Input vectors must have the same shape, but got "
                         f"{v1_arr.shape} and {v2_arr.shape}.")
    if v1_arr.ndim != 1:
        raise ValueError(f"Input vectors must be 1D, but got ndim={v1_arr.ndim}.")

    # 3. Calculate norms
    norm_v1 = np.linalg.norm(v1_arr)
    norm_v2 = np.linalg.norm(v2_arr)

    # 4. Handle zero or near-zero norms BEFORE normalization
    _SALI_ZERO_TOLERANCE = 1e-15
    if norm_v1 < _SALI_ZERO_TOLERANCE or norm_v2 < _SALI_ZERO_TOLERANCE:
        warnings.warn(f"Input vector norm near zero ({norm_v1:.2e}, {norm_v2:.2e}). Returning SALI = 0.0", RuntimeWarning)
        return np.nan

    # --- Norms are non-zero, proceed with calculation ---

    # 5. Normalize the vectors (division is now safe)
    v1_hat = v1_arr / norm_v1
    v2_hat = v2_arr / norm_v2

    # 6. Calculate norms of the sum and difference of NORMALIZED vectors
    # Use np.linalg.norm for potentially better numerical stability / handling edge cases
    norm_sum = np.linalg.norm(v1_hat + v2_hat)
    norm_diff = np.linalg.norm(v1_hat - v2_hat)

    # 7. SALI is the minimum of these two norms
    sali_value = min(norm_sum, norm_diff) # Standard min is fine for two scalars

    return sali_value

# =========================================
# Modified Gram-Schmidt QR Decomposition
# =========================================
@cuda.jit(device=True)
def mgs_QR(A, R_diag_out, n_rows, n_cols):
    """
    Performs Modified Gram-Schmidt QR decomposition in-place on matrix A.
    A is overwritten with Q. Diagonal elements of R are stored in R_diag_out.

    Args:
        A (cuda.local.array(ndim=2)): Input matrix (n_rows x n_cols) with vectors as columns.
                                      Modified in-place to become Q.
        R_diag_out (cuda.local.array(ndim=1)): Output array (n_cols) for diagonal R_jj elements.
        n_rows (int): Number of rows (dimension of vectors).
        n_cols (int): Number of columns (number of vectors).
    """
    # Outer loop: Iterate through columns (j) to orthogonalize/normalize
    for j in range(n_cols):
        # Calculate norm squared of the current j-th column vector (a_j)
        norm_sq_j = 0.0
        for row in range(n_rows):
            val = A[row, j]
            norm_sq_j += val * val

        # Calculate R_jj = ||a_j||
        R_jj = math.sqrt(norm_sq_j)
        R_diag_out[j] = R_jj

        # Normalize the j-th column (q_j = a_j / R_jj) if norm is sufficient
        inv_R_jj = 1.0 / R_jj
        for row in range(n_rows):
            A[row, j] *= inv_R_jj # A[:, j] now contains q_j

        # Continue to orthogonalize subsequent vectors against previous q's
        # but skip projecting onto this zero vector q_j below.

        # Inner loop: Orthogonalize subsequent columns (i > j) against q_j
        for i in range(j + 1, n_cols):
            # Calculate projection coefficient R_ji = q_j^T * a_i
            R_ji = 0.0
            for row in range(n_rows):
                R_ji += A[row, j] * A[row, i] # Dot product q_j . a_i

            # Subtract the projection: a_i = a_i - R_ji * q_j
            for row in range(n_rows):
                A[row, i] -= R_ji * A[row, j]

# =========================================
# Error Integration Function
# =========================================
@cuda.jit(device=True, inline=True)
def comp_err_int(err_array, n: int):
    """
    Computes the L2 norm (Euclidean norm) of an error vector.

    Args:
        err_array: A Numba array (e.g., cuda.local.array) containing the error
                   vector components.
        n: The number of elements (dimension) in the error vector.

    Returns:
        The L2 norm of the error vector (float64).
    """
    accum = 0.0
    for i in range(n):
        accum += err_array[i] * err_array[i]
    return math.sqrt(accum)

# =========================================
# Solver Kernel Factory Function (Fixed Step)
# =========================================
def create_solver_kernel(ode_func, num_vars: int):
    """
    Factory function that creates a Numba CUDA kernel for a fixed-step
    8th order Dormand-Prince integrator (DP8).

    Args:
        ode_func: A Numba CUDA *device function* representing the system
                  of Ordinary Differential Equations (ODEs). It must have
                  the signature: `ode_func(t, Y, dYdt, params)`, where:
                    - t: current time (float)
                    - Y: current state vector (1D Numba array)
                    - dYdt: output array to store derivatives (1D Numba array)
                    - params: a tuple containing any necessary parameters
                              for the ODE system.
        num_vars: The number of variables (dimension) in the ODE system.
                  This must match the length of Y and dYdt used by ode_func.

    Returns:
        A Numba CUDA kernel function specialized for the given ode_func
        and num_vars. The kernel has the signature:
        `kernel(Y0, t0, dt, params, Y_out, steps)`
          - Y0: GPU array of initial conditions (num_ics, num_vars).
          - t0: Initial time (float).
          - dt: Fixed time step (float).
          - params: Tuple of parameters to pass to ode_func.
          - Y_out: GPU array to store final states (num_ics, num_vars).
          - steps: Number of integration steps (int).
    """

    _NUM_VARS = num_vars # Capture num_vars for use inside the kernel definition

    @cuda.jit
    def solver_fix_step(Y0, t0, dt, params, Y_out, steps):
        """
        CUDA kernel for integration with a 8th order Dormand-Prince
        fixed step method. This kernel is generated by create_solver_kernel.
        (Internal Numba JIT function)
        """
        idx = cuda.grid(1)
        if idx >= Y0.shape[0]:
            return

        # Declare local arrays using the num_vars from the factory scope
        Y = cuda.local.array(_NUM_VARS, float64)
        dYdt = cuda.local.array(_NUM_VARS, float64)
        k1   = cuda.local.array(_NUM_VARS, float64)
        k2   = cuda.local.array(_NUM_VARS, float64)
        k3   = cuda.local.array(_NUM_VARS, float64)
        k4   = cuda.local.array(_NUM_VARS, float64)
        k5   = cuda.local.array(_NUM_VARS, float64)
        k6   = cuda.local.array(_NUM_VARS, float64)
        k7   = cuda.local.array(_NUM_VARS, float64)
        k8   = cuda.local.array(_NUM_VARS, float64)
        k9   = cuda.local.array(_NUM_VARS, float64)
        k10  = cuda.local.array(_NUM_VARS, float64)
        k11  = cuda.local.array(_NUM_VARS, float64)
        k12  = cuda.local.array(_NUM_VARS, float64)
        k13  = cuda.local.array(_NUM_VARS, float64)
        Y_temp = cuda.local.array(_NUM_VARS, float64)

        # Copy initial state from global memory to local registers
        for i in range(_NUM_VARS):
            Y[i] = Y0[idx, i]

        # --- Dormand-Prince 8(5,3) Coefficients ---
        c2 = 1.0 / 18.0; c3 = 1.0 / 12.0; c4 = 1.0 / 8.0; c5 = 5.0 / 16.0
        c6 = 3.0 / 8.0; c7 = 59.0 / 400.0; c8 = 93.0 / 200.0
        c9 = 5490023248.0 / 9719169821.0; c10 = 13.0 / 20.0
        c11 = 1201146811.0 / 1299019798.0; c12 = 1.0; c13 = 1.0
        b1 = 14005451.0 / 335480064.0; b6 = -59238493.0 / 1068277825.0
        b7 = 181606767.0 / 758867731.0; b8 = 561292985.0 / 797845732.0
        b9 = -1041891430.0 / 1371343529.0; b10 = 760417239.0 / 1151165299.0
        b11 = 118820643.0 / 751138087.0; b12 = -528747749.0 / 2220607170.0
        b13 = 1.0 / 4.0
        # ------------------------------------------

        t = t0
        for step_i in range(steps):
            # --- Stage 1 ---
            ode_func(t, Y, dYdt, params)
            for i in range(_NUM_VARS): k1[i] = dt * dYdt[i]
            # --- Stage 2 ---
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (1.0 / 18.0) * k1[i]
            ode_func(t + c2 * dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k2[i] = dt * dYdt[i]
            # --- Stage 3 ---
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (1.0 / 48.0) * k1[i] + (1.0 / 16.0) * k2[i]
            ode_func(t + c3 * dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k3[i] = dt * dYdt[i]
            # --- Stage 4 ---
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (1.0 / 32.0) * k1[i] + (3.0 / 32.0) * k3[i]
            ode_func(t + c4 * dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k4[i] = dt * dYdt[i]
            # --- Stage 5 ---
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (5.0 / 16.0) * k1[i] + (-75.0 / 64.0) * k3[i] + (75.0 / 64.0) * k4[i]
            ode_func(t + c5 * dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k5[i] = dt * dYdt[i]
            # --- Stage 6 ---
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (3.0 / 80.0) * k1[i] + (3.0 / 16.0) * k4[i] + (3.0 / 20.0) * k5[i]
            ode_func(t + c6 * dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k6[i] = dt * dYdt[i]
            # --- Stage 7 ---
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (29443841.0 / 614563906.0) * k1[i] + (77736538.0 / 692538347.0) * k4[i] + (-28693883.0 / 1125000000.0) * k5[i] + (23124283.0 / 1800000000.0) * k6[i]
            ode_func(t + c7 * dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k7[i] = dt * dYdt[i]
            # --- Stage 8 ---
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (16016141.0 / 946692911.0) * k1[i] + (61564180.0 / 158732637.0) * k4[i] + (22789713.0 / 633445777.0) * k5[i] + (545815736.0 / 2771057229.0) * k6[i] + (-180193667.0 / 1043307555.0) * k7[i]
            ode_func(t + c8 * dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k8[i] = dt * dYdt[i]
            # --- Stage 9 ---
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (39632708.0 / 573591083.0) * k1[i] + (-433636366.0 / 683701615.0) * k4[i] + (-421739975.0 / 2616292301.0) * k5[i] + (100302831.0 / 723423059.0) * k6[i] + (790204164.0 / 839813087.0) * k7[i] + (800635310.0 / 3783071287.0) * k8[i]
            ode_func(t + c9 * dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k9[i] = dt * dYdt[i]
            # --- Stage 10 ---
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (246121993.0 / 1340847787.0) * k1[i] + (-37695042795.0 / 15268766246.0) * k4[i] + (-309121744.0 / 1061227803.0) * k5[i] + (-12992083.0 / 490766935.0) * k6[i] + (6005943493.0 / 2108947869.0) * k7[i] + (393006217.0 / 1396673457.0) * k8[i] + (123872331.0 / 1001029789.0) * k9[i]
            ode_func(t + c10 * dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k10[i] = dt * dYdt[i]
            # --- Stage 11 ---
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (-1028468189.0 / 846180014.0) * k1[i] + (8478235783.0 / 508512852.0) * k4[i] + (1311729495.0 / 1432422823.0) * k5[i] + (-10304129995.0 / 1701304382.0) * k6[i] + (-48777925059.0 / 3047939560.0) * k7[i] + (15336726248.0 / 1032824649.0) * k8[i] + (-45442868181.0 / 3398467696.0) * k9[i] + (3065993473.0 / 597172653.0) * k10[i]
            ode_func(t + c11 * dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k11[i] = dt * dYdt[i]
            # --- Stage 12 ---
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (185892177.0 / 718116043.0) * k1[i] + (-3185094517.0 / 667107341.0) * k4[i] + (-477755414.0 / 1098053517.0) * k5[i] + (-703635378.0 / 230739211.0) * k6[i] + (5731566787.0 / 1027545527.0) * k7[i] + (5232866602.0 / 850066563.0) * k8[i] + (-4093664535.0 / 808688257.0) * k9[i] + (3962137247.0 / 1805957418.0) * k10[i] + (65686358.0 / 487910083.0) * k11[i]
            ode_func(t + c12 * dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k12[i] = dt * dYdt[i]
            # --- Stage 13 ---
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (403863854.0 / 491063109.0) * k1[i] + (-5068492393.0 / 434740067.0) * k4[i] + (-411421997.0 / 543043805.0) * k5[i] + (652783627.0 / 914296604.0) * k6[i] + (11173962825.0 / 925320556.0) * k7[i] + (-13158990841.0 / 6184727034.0) * k8[i] + (3936647629.0 / 1978049680.0) * k9[i] + (-160528059.0 / 685178525.0) * k10[i] + (248638103.0 / 1413531060.0) * k11[i]
            ode_func(t + c13 * dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k13[i] = dt * dYdt[i]

            # --- Final State Update (8th order) ---
            for i in range(_NUM_VARS):
                Y[i] += (b1 * k1[i] + b6 * k6[i] + b7 * k7[i] + b8 * k8[i] +
                         b9 * k9[i] + b10 * k10[i] + b11 * k11[i] + b12 * k12[i] +
                         b13 * k13[i])
            t += dt

        # Write the final state back to global memory
        for i in range(_NUM_VARS):
            Y_out[idx, i] = Y[i]

    return solver_fix_step # Return the compiled kernel function

# =========================================
# Solver Kernel Factory Function (Adaptive Step)
# =========================================
def create_solver_kernel_adaptive(ode_func, num_vars: int):
    """
    Factory function that creates a Numba CUDA kernel for an adaptive-step
    8th order Dormand-Prince integrator (DP8) with a 5th order embedded
    method for error estimation (DP8(5)).

    Args:
        ode_func: A Numba CUDA *device function* representing the system
                  of Ordinary Differential Equations (ODEs). It must have
                  the signature: `ode_func(t, Y, dYdt, params)`, where:
                    - t: current time (float)
                    - Y: current state vector (1D Numba array)
                    - dYdt: output array to store derivatives (1D Numba array)
                    - params: a tuple containing any necessary parameters
                              for the ODE system.
        num_vars: The number of variables (dimension) in the ODE system.
                  This must match the length of Y and dYdt used by ode_func.

    Returns:
        A Numba CUDA kernel function specialized for the given ode_func
        and num_vars. The kernel has the signature:
        `kernel(Y0, t0, t_final, params, tol, dt_initial, max_steps, Y_out)`
          - Y0: GPU array of initial conditions (num_ics, num_vars).
          - t0: Initial time (float).
          - t_final: The time to integrate until (float).
          - params: Tuple of parameters to pass to ode_func.
          - tol: Absolute error tolerance for step control (float).
          - dt_initial: Initial guess for the time step (float).
          - max_steps: Maximum number of adaptive steps allowed (int).
          - Y_out: GPU array to store final states (num_ics, num_vars).
                   Note: Contains the state at the *last accepted step*. If
                   max_steps is reached before t_final, the state will be
                   at some t < t_final.
    """

    _NUM_VARS = num_vars # Capture for use inside kernel

    @cuda.jit
    def solver_adaptive_step(Y0, t0, t_final, params, tol, dt_initial, max_steps, Y_out):
        """
        CUDA kernel for adaptive integration with DP8(5).
        Generated by create_solver_kernel_adaptive.
        (Internal Numba JIT function)
        """
        idx = cuda.grid(1)
        if idx >= Y0.shape[0]:
            return

        # Allocate local arrays
        Y      = cuda.local.array(_NUM_VARS, float64)
        dYdt   = cuda.local.array(_NUM_VARS, float64)
        k1     = cuda.local.array(_NUM_VARS, float64)
        k2     = cuda.local.array(_NUM_VARS, float64)
        k3     = cuda.local.array(_NUM_VARS, float64)
        k4     = cuda.local.array(_NUM_VARS, float64)
        k5     = cuda.local.array(_NUM_VARS, float64)
        k6     = cuda.local.array(_NUM_VARS, float64)
        k7     = cuda.local.array(_NUM_VARS, float64)
        k8     = cuda.local.array(_NUM_VARS, float64)
        k9     = cuda.local.array(_NUM_VARS, float64)
        k10    = cuda.local.array(_NUM_VARS, float64)
        k11    = cuda.local.array(_NUM_VARS, float64)
        k12    = cuda.local.array(_NUM_VARS, float64)
        k13    = cuda.local.array(_NUM_VARS, float64)
        Y_temp = cuda.local.array(_NUM_VARS, float64) # Holds 8th order result proposal
        Y_err  = cuda.local.array(_NUM_VARS, float64) # Holds difference Y8 - Y5

        # Load initial state
        for i in range(_NUM_VARS):
            Y[i] = Y0[idx, i]

        # --- DP8(5) Coefficients ---
        # Step Coefficients (c_i) - Same as fixed-step
        c2 = 1.0/18.0; c3 = 1.0/12.0; c4 = 1.0/8.0; c5 = 5.0/16.0
        c6 = 3.0/8.0; c7 = 59.0/400.0; c8 = 93.0/200.0
        c9 = 5490023248.0/9719169821.0; c10 = 13.0/20.0
        c11 = 1201146811.0/1299019798.0; c12 = 1.0; c13 = 1.0
        # 8th Order Update Coefficients (b_i) - Same as fixed-step
        b1 = 14005451.0/335480064.0; b6 = -59238493.0/1068277825.0
        b7 = 181606767.0/758867731.0; b8 = 561292985.0/797845732.0
        b9 = -1041891430.0/1371343529.0; b10 = 760417239.0/1151165299.0
        b11 = 118820643.0/751138087.0; b12 = -528747749.0/2220607170.0
        b13 = 1.0/4.0
        # 5th Order Embedded Update Coefficients (bs_i)
        bs1  =  13451932.0/455176623.0; bs6 = -808719846.0/976000145.0
        bs7  =  1757004468.0/5645159321.0; bs8 =  656045339.0/265891186.0
        bs9  = -3867574721.0/1518517206.0; bs10 = 465885868.0/322736535.0
        bs11 =  53011238.0/667516719.0; bs12 =  2.0/45.0
        bs13 =  0.0 # bs13 is zero for this embedding
        # ------------------------------------------

        t = t0
        dt = dt_initial
        step_count = 0
        # Safety factors for step size control
        safety = 0.9
        min_scale = 0.2
        max_scale = 5.0
        # Use exponent for 8(5) method, typically 1/(order+1) = 1/6 for the error estimator
        exponent = 1.0 / 6.0
        tiny = 1e-30 # To prevent division by zero in error scaling


        # Main adaptive integration loop
        while (t < t_final) and (step_count < max_steps):
            # Adjust dt if it would overshoot t_final
            if (t + dt > t_final):
                dt = t_final - t

            # --- Compute the 13 Runge-Kutta stages (k1 to k13) ---
            # k1
            ode_func(t, Y, dYdt, params)
            for i in range(_NUM_VARS): k1[i] = dt * dYdt[i]
            # k2
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (1.0/18.0) * k1[i] # Using Y_temp as scratch space
            ode_func(t + c2 * dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k2[i] = dt * dYdt[i]
            # k3
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (1.0/48.0)*k1[i] + (1.0/16.0)*k2[i]
            ode_func(t + c3 * dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k3[i] = dt * dYdt[i]
            # k4
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (1.0/32.0)*k1[i] + (3.0/32.0)*k3[i]
            ode_func(t + c4 * dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k4[i] = dt * dYdt[i]
            # k5
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (5.0/16.0)*k1[i] + (-75.0/64.0)*k3[i] + (75.0/64.0)*k4[i]
            ode_func(t + c5 * dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k5[i] = dt * dYdt[i]
            # k6
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (3.0/80.0)*k1[i] + (3.0/16.0)*k4[i] + (3.0/20.0)*k5[i]
            ode_func(t + c6 * dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k6[i] = dt * dYdt[i]
            # k7
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (29443841.0/614563906.0)*k1[i] + (77736538.0/692538347.0)*k4[i] + (-28693883.0/1125000000.0)*k5[i] + (23124283.0/1800000000.0)*k6[i]
            ode_func(t + c7 * dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k7[i] = dt * dYdt[i]
            # k8
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (16016141.0/946692911.0)*k1[i] + (61564180.0/158732637.0)*k4[i] + (22789713.0/633445777.0)*k5[i] + (545815736.0/2771057229.0)*k6[i] + (-180193667.0/1043307555.0)*k7[i]
            ode_func(t + c8 * dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k8[i] = dt * dYdt[i]
            # k9
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (39632708.0/573591083.0)*k1[i] + (-433636366.0/683701615.0)*k4[i] + (-421739975.0/2616292301.0)*k5[i] + (100302831.0/723423059.0)*k6[i] + (790204164.0/839813087.0)*k7[i] + (800635310.0/3783071287.0)*k8[i]
            ode_func(t + c9 * dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k9[i] = dt * dYdt[i]
            # k10
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (246121993.0/1340847787.0)*k1[i] + (-37695042795.0/15268766246.0)*k4[i] + (-309121744.0/1061227803.0)*k5[i] + (-12992083.0/490766935.0)*k6[i] + (6005943493.0/2108947869.0)*k7[i] + (393006217.0/1396673457.0)*k8[i] + (123872331.0/1001029789.0)*k9[i]
            ode_func(t + c10 * dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k10[i] = dt * dYdt[i]
            # k11
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (-1028468189.0/846180014.0)*k1[i] + (8478235783.0/508512852.0)*k4[i] + (1311729495.0/1432422823.0)*k5[i] + (-10304129995.0/1701304382.0)*k6[i] + (-48777925059.0/3047939560.0)*k7[i] + (15336726248.0/1032824649.0)*k8[i] + (-45442868181.0/3398467696.0)*k9[i] + (3065993473.0/597172653.0)*k10[i]
            ode_func(t + c11 * dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k11[i] = dt * dYdt[i]
            # k12
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (185892177.0/718116043.0)*k1[i] + (-3185094517.0/667107341.0)*k4[i] + (-477755414.0/1098053517.0)*k5[i] + (-703635378.0/230739211.0)*k6[i] + (5731566787.0/1027545527.0)*k7[i] + (5232866602.0/850066563.0)*k8[i] + (-4093664535.0/808688257.0)*k9[i] + (3962137247.0/1805957418.0)*k10[i] + (65686358.0/487910083.0)*k11[i]
            ode_func(t + c12 * dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k12[i] = dt * dYdt[i]
            # k13
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (403863854.0/491063109.0)*k1[i] + (-5068492393.0/434740067.0)*k4[i] + (-411421997.0/543043805.0)*k5[i] + (652783627.0/914296604.0)*k6[i] + (11173962825.0/925320556.0)*k7[i] + (-13158990841.0/6184727034.0)*k8[i] + (3936647629.0/1978049680.0)*k9[i] + (-160528059.0/685178525.0)*k10[i] + (248638103.0/1413531060.0)*k11[i] # Assuming a13_12 = 0
            ode_func(t + c13 * dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k13[i] = dt * dYdt[i]


            # --- Compute 8th order solution proposal ---
            # (Store in Y_temp for now, don't update Y until step accepted)
            for i in range(_NUM_VARS):
                y8_i = (Y[i] +
                        b1 * k1[i] + b6 * k6[i] + b7 * k7[i] + b8 * k8[i] +
                        b9 * k9[i] + b10 * k10[i] + b11 * k11[i] + b12 * k12[i] +
                        b13 * k13[i])
                Y_temp[i] = y8_i # Store proposed 8th order result

            # --- Compute embedded 5th order solution for error estimate ---
            for i in range(_NUM_VARS):
                y5_i = (Y[i] +
                        bs1*k1[i] + bs6*k6[i] + bs7*k7[i] + bs8*k8[i] +
                        bs9*k9[i] + bs10*k10[i] + bs11*k11[i] + bs12*k12[i] +
                        bs13*k13[i])
                # --- Calculate error estimate (difference between orders) ---
                Y_err[i] = Y_temp[i] - y5_i # Y_err = y8 - y5

            # --- Calculate the error norm ---
            # Use the extracted comp_err_int function
            err_norm = comp_err_int(Y_err, _NUM_VARS)

            # --- Adaptive step size control ---
            if err_norm <= tol:
                # Step accepted: Update state and time
                t += dt
                step_count += 1
                for i in range(_NUM_VARS):
                    Y[i] = Y_temp[i] # Update Y with the accepted 8th order result
                # Write the accepted state to output (overwrites previous)
                for i in range(_NUM_VARS):
                   Y_out[idx, i] = Y[i]

                # Calculate optimal step size factor
                if err_norm == 0.0: # Handle case of zero error
                    scale = max_scale
                else:
                    scale = safety * math.pow(tol / err_norm, exponent)
                    scale = min(max_scale, max(min_scale, scale)) # Clamp scale factor

                # Suggest new step size (but don't make it zero)
                dt = max(dt * scale, tiny)

            else:
                # Step rejected: Reduce step size and retry
                scale = safety * math.pow(tol / err_norm, exponent)
                scale = max(min_scale, scale) # Clamp scale factor (only clamp minimum on rejection)
                dt = dt * scale
                # Do not advance time or step_count, Y remains unchanged

            # Prevent dt from becoming too small
            if abs(dt) <= tiny:
                # Handle error: step size too small (e.g., break or set flag)
                # For now, just break the loop
                break

        # --- End of while loop ---
        # Final state (at last accepted time t) is already in Y_out

    return solver_adaptive_step # Return the compiled kernel function


# =========================================
# Solver Kernel Variational Factory Function (Fixed Step)
# =========================================
def create_solver_kernel_variational(ode_func, num_vars: int, num_base_vars: int, num_dev_vectors: int):
    """
    Factory for a fixed-step DP8 kernel that integrates base variables,
    deviation vectors, and auxiliary variables (like LD). Includes optional
    renormalization of deviation vectors.

    Assumes state vector Y structure:
    [ base_vars (num_base_vars),
      dev_vec_0 (num_base_vars),
      dev_vec_1 (num_base_vars),
      ...,
      dev_vec_{M-1} (num_base_vars),
      aux_vars (...) ]
    where M = num_dev_vectors. Total size = num_vars.

    Args:
        ode_func: Numba CUDA device function `ode_func(t, Y, dYdt, params)`.
        num_vars (int): Total number of variables in the state vector Y.
        num_base_vars (int): Number of base phase space variables (e.g., 2*N).
        num_dev_vectors (int): Number of deviation vectors (M).

    Returns:
        A Numba CUDA kernel function with signature:
        `kernel(Y0, t0, dt, steps, params, renorm_interval, Y_out)`
          - Y0: Initial states (GPU array).
          - t0: Initial time.
          - dt: Time step.
          - steps: Number of steps.
          - params: Tuple of parameters for ode_func.
          - renorm_interval: Renormalize deviation vectors every N steps.
                             Set <= 0 to disable.
          - Y_out: Output array for final states (GPU array).
    """
    _NUM_VARS = num_vars
    _NUM_BASE_VARS = num_base_vars
    _NUM_DEV_VECTORS = num_dev_vectors

    if _NUM_VARS != _NUM_BASE_VARS + _NUM_DEV_VECTORS * _NUM_BASE_VARS + 1:
         # This check assumes 1 auxiliary variable (LD) at the end. Adjust if needed.
         print(f"Warning: num_vars ({_NUM_VARS}) may not match expected structure: "
               f"base ({_NUM_BASE_VARS}) + M*base ({_NUM_DEV_VECTORS}*{_NUM_BASE_VARS}) + 1 (aux).")


    @cuda.jit
    def solver_fixed_step_variational(Y0, t0, dt, steps, params, renorm_interval, Y_out):
        """
        Fixed-step DP8 kernel for base + deviation vectors.
        Generated by create_solver_kernel_variational.
        (Internal Numba JIT function)
        """
        idx = cuda.grid(1)
        if idx >= Y0.shape[0]:
            return

        # Local arrays sized by TOTAL variables
        Y = cuda.local.array(_NUM_VARS, float64)
        for i in range(_NUM_VARS): Y[i] = Y0[idx, i]

        dYdt   = cuda.local.array(_NUM_VARS, float64)
        k1     = cuda.local.array(_NUM_VARS, float64)
        # ... k2 to k13 ...
        k2   = cuda.local.array(_NUM_VARS, float64); k3 = cuda.local.array(_NUM_VARS, float64)
        k4   = cuda.local.array(_NUM_VARS, float64); k5 = cuda.local.array(_NUM_VARS, float64)
        k6   = cuda.local.array(_NUM_VARS, float64); k7 = cuda.local.array(_NUM_VARS, float64)
        k8   = cuda.local.array(_NUM_VARS, float64); k9 = cuda.local.array(_NUM_VARS, float64)
        k10  = cuda.local.array(_NUM_VARS, float64); k11 = cuda.local.array(_NUM_VARS, float64)
        k12  = cuda.local.array(_NUM_VARS, float64); k13 = cuda.local.array(_NUM_VARS, float64)
        Y_temp = cuda.local.array(_NUM_VARS, float64)

        # DP8 Constants (same as before)
        c2 = 1.0/18.0; c3 = 1.0/12.0; c4 = 1.0/8.0; c5 = 5.0/16.0
        c6 = 3.0/8.0; c7 = 59.0/400.0; c8 = 93.0/200.0
        c9 = 5490023248.0/9719169821.0; c10 = 13.0/20.0
        c11 = 1201146811.0/1299019798.0; c12 = 1.0; c13 = 1.0
        b1 = 14005451.0/335480064.0; b6 = -59238493.0/1068277825.0
        b7 = 181606767.0/758867731.0; b8 = 561292985.0/797845732.0
        b9 = -1041891430.0/1371343529.0; b10 = 760417239.0/1151165299.0
        b11 = 118820643.0/751138087.0; b12 = -528747749.0/2220607170.0
        b13 = 1.0/4.0

        t = t0
        for step_i in range(steps):
            # --- Standard DP8 Stages ---
            ode_func(t, Y, dYdt, params)
            for i in range(_NUM_VARS): k1[i] = dt * dYdt[i]
            # k2
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (1.0/18.0)*k1[i]
            ode_func(t + c2*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k2[i] = dt * dYdt[i]
            # k3
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (1.0/48.0)*k1[i] + (1.0/16.0)*k2[i]
            ode_func(t + c3*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k3[i] = dt * dYdt[i]
            # ... stages k4 to k13 ...
            # k4
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (1.0/32.0)*k1[i] + (3.0/32.0)*k3[i]
            ode_func(t + c4*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k4[i] = dt * dYdt[i]
            # k5
            for i in range(_NUM_VARS): Y_temp[i] = Y[i]+ (5.0/16.0)*k1[i]+ (-75.0/64.0)*k3[i]+ (75.0/64.0)*k4[i]
            ode_func(t + c5*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k5[i] = dt * dYdt[i]
            # k6
            for i in range(_NUM_VARS): Y_temp[i] = Y[i]+ (3.0/80.0)*k1[i]+ (3.0/16.0)*k4[i]+ (3.0/20.0)*k5[i]
            ode_func(t + c6*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k6[i] = dt * dYdt[i]
            # k7
            for i in range(_NUM_VARS): Y_temp[i] = Y[i]+ (29443841.0/614563906.0)*k1[i]+ (77736538.0/692538347.0)*k4[i]+ (-28693883.0/1125000000.0)*k5[i]+ (23124283.0/1800000000.0)*k6[i]
            ode_func(t + c7*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k7[i] = dt * dYdt[i]
            # k8
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (16016141.0/946692911.0)*k1[i] + (61564180.0/158732637.0)*k4[i] + (22789713.0/633445777.0)*k5[i] + (545815736.0/2771057229.0)*k6[i] + (-180193667.0/1043307555.0)*k7[i]
            ode_func(t + c8*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k8[i] = dt * dYdt[i]
            # k9
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (39632708.0/573591083.0)*k1[i] + (-433636366.0/683701615.0)*k4[i] + (-421739975.0/2616292301.0)*k5[i] + (100302831.0/723423059.0)*k6[i] + (790204164.0/839813087.0)*k7[i] + (800635310.0/3783071287.0)*k8[i]
            ode_func(t + c9*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k9[i] = dt * dYdt[i]
            # k10
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (246121993.0/1340847787.0)*k1[i] + (-37695042795.0/15268766246.0)*k4[i] + (-309121744.0/1061227803.0)*k5[i] + (-12992083.0/490766935.0)*k6[i] + (6005943493.0/2108947869.0)*k7[i] + (393006217.0/1396673457.0)*k8[i] + (123872331.0/1001029789.0)*k9[i]
            ode_func(t + c10*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k10[i] = dt * dYdt[i]
            # k11
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (-1028468189.0/846180014.0)*k1[i] + (8478235783.0/508512852.0)*k4[i] + (1311729495.0/1432422823.0)*k5[i] + (-10304129995.0/1701304382.0)*k6[i] + (-48777925059.0/3047939560.0)*k7[i] + (15336726248.0/1032824649.0)*k8[i] + (-45442868181.0/3398467696.0)*k9[i] + (3065993473.0/597172653.0)*k10[i]
            ode_func(t + c11*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k11[i] = dt * dYdt[i]
            # k12
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (185892177.0/718116043.0)*k1[i] + (-3185094517.0/667107341.0)*k4[i] + (-477755414.0/1098053517.0)*k5[i] + (-703635378.0/230739211.0)*k6[i] + (5731566787.0/1027545527.0)*k7[i] + (5232866602.0/850066563.0)*k8[i] + (-4093664535.0/808688257.0)*k9[i] + (3962137247.0/1805957418.0)*k10[i] + (65686358.0/487910083.0)*k11[i]
            ode_func(t + c12*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k12[i] = dt * dYdt[i]
            # k13
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (403863854.0/491063109.0)*k1[i] + (-5068492393.0/434740067.0)*k4[i] + (-411421997.0/543043805.0)*k5[i] + (652783627.0/914296604.0)*k6[i] + (11173962825.0/925320556.0)*k7[i] + (-13158990841.0/6184727034.0)*k8[i] + (3936647629.0/1978049680.0)*k9[i] + (-160528059.0/685178525.0)*k10[i] + (248638103.0/1413531060.0)*k11[i]
            ode_func(t + c13*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k13[i] = dt * dYdt[i]

            # --- Final Update ---
            for i in range(_NUM_VARS):
                Y[i] += (b1 * k1[i] + b6 * k6[i] + b7 * k7[i] + b8 * k8[i] +
                         b9 * k9[i] + b10 * k10[i] + b11 * k11[i] + b12 * k12[i] +
                         b13 * k13[i])
            t += dt

            # --- Optional Renormalization ---
            if renorm_interval > 0 and (step_i + 1) % renorm_interval == 0:
                # Loop over each deviation vector
                for m_idx in range(_NUM_DEV_VECTORS):
                    # Calculate base index for this deviation vector set
                    base = _NUM_BASE_VARS + m_idx * _NUM_BASE_VARS
                    # Calculate L2 norm squared for this vector (size = num_base_vars)
                    norm_sq = 0.0
                    for i in range(_NUM_BASE_VARS):
                        val = Y[base + i]
                        norm_sq += val * val
                    inv_norm = 1.0 / math.sqrt(norm_sq)
                    # Renormalize
                    for i in range(_NUM_BASE_VARS):
                        Y[base + i] *= inv_norm

        # --- End Integration Loop ---
        for i in range(_NUM_VARS):
            Y_out[idx, i] = Y[i]

    return solver_fixed_step_variational

# =========================================
# Solver Kernel Variational Factory Function (Adaptive Step)
# =========================================
def create_solver_kernel_adaptive_variational(ode_func, num_vars: int, num_base_vars: int, num_dev_vectors: int):
    """
    Factory for an *adaptive-step* DP8(5) kernel that integrates base variables,
    deviation vectors, and auxiliary variables. Includes optional renormalization
    of deviation vectors after accepted steps.

    Assumes state vector Y structure:
    [ base_vars (num_base_vars),
      dev_vec_0 (num_base_vars),
      dev_vec_1 (num_base_vars),
      ...,
      dev_vec_{M-1} (num_base_vars),
      aux_vars (...) ]
    where M = num_dev_vectors. Total size = num_vars.

    Args:
        ode_func: Numba CUDA device function `ode_func(t, Y, dYdt, params)`.
        num_vars (int): Total number of variables in the state vector Y.
        num_base_vars (int): Number of base phase space variables (e.g., 2*N).
        num_dev_vectors (int): Number of deviation vectors (M).

    Returns:
        A Numba CUDA kernel function with signature:
        `kernel(Y0, t0, t_final, params, tol, dt_initial, max_steps, renorm_interval, Y_out)`
          - Y0: Initial states (GPU array).
          - t0: Initial time.
          - t_final: Target final time.
          - params: Tuple of parameters for ode_func.
          - tol: Absolute error tolerance for step control.
          - dt_initial: Initial guess for the time step.
          - max_steps: Maximum number of adaptive steps allowed.
          - renorm_interval: Renormalize deviation vectors every N *accepted* steps.
                             Set <= 0 to disable.
          - Y_out: Output array for final states (GPU array). Contains state at
                   last accepted step time.
    """
    _NUM_VARS = num_vars
    _NUM_BASE_VARS = num_base_vars
    _NUM_DEV_VECTORS = num_dev_vectors

    if _NUM_VARS != _NUM_BASE_VARS + _NUM_DEV_VECTORS * _NUM_BASE_VARS + 1:
        # This check assumes 1 auxiliary variable (LD) at the end. Adjust if needed.
         print(f"Warning: num_vars ({_NUM_VARS}) may not match expected structure: "
               f"base ({_NUM_BASE_VARS}) + M*base ({_NUM_DEV_VECTORS}*{_NUM_BASE_VARS}) + 1 (aux).")

    @cuda.jit
    def solver_adaptive_step_variational(Y0, t0, t_final, params, tol, dt_initial, max_steps, renorm_interval, Y_out):
        """
        Adaptive-step DP8(5) kernel for base + deviation vectors.
        Generated by create_solver_kernel_adaptive_variational.
        (Internal Numba JIT function)
        """
        idx = cuda.grid(1)
        if idx >= Y0.shape[0]:
            return

        # Local arrays sized by TOTAL variables
        Y = cuda.local.array(_NUM_VARS, float64)
        for i in range(_NUM_VARS): Y[i] = Y0[idx, i]

        dYdt   = cuda.local.array(_NUM_VARS, float64)
        k1     = cuda.local.array(_NUM_VARS, float64); k2 = cuda.local.array(_NUM_VARS, float64)
        k3     = cuda.local.array(_NUM_VARS, float64); k4 = cuda.local.array(_NUM_VARS, float64)
        k5     = cuda.local.array(_NUM_VARS, float64); k6 = cuda.local.array(_NUM_VARS, float64)
        k7     = cuda.local.array(_NUM_VARS, float64); k8 = cuda.local.array(_NUM_VARS, float64)
        k9     = cuda.local.array(_NUM_VARS, float64); k10 = cuda.local.array(_NUM_VARS, float64)
        k11    = cuda.local.array(_NUM_VARS, float64); k12 = cuda.local.array(_NUM_VARS, float64)
        k13    = cuda.local.array(_NUM_VARS, float64)
        Y_temp = cuda.local.array(_NUM_VARS, float64) # Holds 8th order proposal
        Y_err  = cuda.local.array(_NUM_VARS, float64) # Holds difference Y8 - Y5

        # DP8(5) Coefficients (same as standard adaptive)
        c2 = 1.0/18.0; c3 = 1.0/12.0; c4 = 1.0/8.0; c5 = 5.0/16.0
        c6 = 3.0/8.0; c7 = 59.0/400.0; c8 = 93.0/200.0
        c9 = 5490023248.0/9719169821.0; c10 = 13.0/20.0
        c11 = 1201146811.0/1299019798.0; c12 = 1.0; c13 = 1.0
        b1 = 14005451.0/335480064.0; b6 = -59238493.0/1068277825.0
        b7 = 181606767.0/758867731.0; b8 = 561292985.0/797845732.0
        b9 = -1041891430.0/1371343529.0; b10 = 760417239.0/1151165299.0
        b11 = 118820643.0/751138087.0; b12 = -528747749.0/2220607170.0
        b13 = 1.0/4.0
        bs1 = 13451932.0/455176623.0; bs6 = -808719846.0/976000145.0
        bs7 = 1757004468.0/5645159321.0; bs8 = 656045339.0/265891186.0
        bs9 = -3867574721.0/1518517206.0; bs10 = 465885868.0/322736535.0
        bs11 = 53011238.0/667516719.0; bs12 = 2.0/45.0; bs13 = 0.0

        # Adaptive step control parameters
        t = t0
        dt = dt_initial
        step_count = 0 # Count accepted steps
        safety = 0.9; min_scale = 0.2; max_scale = 5.0
        exponent = 1.0 / 6.0 # For DP8(5) error estimate
        tiny = 1e-30

        # Main adaptive integration loop
        while (t < t_final) and (step_count < max_steps):
            # Adjust dt if it would overshoot t_final
            if (t + dt > t_final):
                dt = t_final - t

            # --- Compute the 13 Runge-Kutta stages ---
            ode_func(t, Y, dYdt, params)
            for i in range(_NUM_VARS): k1[i] = dt * dYdt[i]
            # k2
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + c2*k1[i]
            ode_func(t + c2*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k2[i] = dt * dYdt[i]
            # k3
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (1.0/48.0)*k1[i] + (1.0/16.0)*k2[i]
            ode_func(t + c3*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k3[i] = dt * dYdt[i]
            # ... stages k4 to k13 (identical calculation logic to other DP8 kernels) ...
            # k4
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (1.0/32.0)*k1[i] + (3.0/32.0)*k3[i]
            ode_func(t + c4*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k4[i] = dt * dYdt[i]
            # k5
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (5.0/16.0)*k1[i] + (-75.0/64.0)*k3[i] + (75.0/64.0)*k4[i]
            ode_func(t + c5*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k5[i] = dt * dYdt[i]
            # k6
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (3.0/80.0)*k1[i] + (3.0/16.0)*k4[i] + (3.0/20.0)*k5[i]
            ode_func(t + c6*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k6[i] = dt * dYdt[i]
            # k7
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (29443841.0/614563906.0)*k1[i] + (77736538.0/692538347.0)*k4[i] + (-28693883.0/1125000000.0)*k5[i] + (23124283.0/1800000000.0)*k6[i]
            ode_func(t + c7*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k7[i] = dt * dYdt[i]
            # k8
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (16016141.0/946692911.0)*k1[i] + (61564180.0/158732637.0)*k4[i] + (22789713.0/633445777.0)*k5[i] + (545815736.0/2771057229.0)*k6[i] + (-180193667.0/1043307555.0)*k7[i]
            ode_func(t + c8*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k8[i] = dt * dYdt[i]
            # k9
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (39632708.0/573591083.0)*k1[i] + (-433636366.0/683701615.0)*k4[i] + (-421739975.0/2616292301.0)*k5[i] + (100302831.0/723423059.0)*k6[i] + (790204164.0/839813087.0)*k7[i] + (800635310.0/3783071287.0)*k8[i]
            ode_func(t + c9*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k9[i] = dt * dYdt[i]
            # k10
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (246121993.0/1340847787.0)*k1[i] + (-37695042795.0/15268766246.0)*k4[i] + (-309121744.0/1061227803.0)*k5[i] + (-12992083.0/490766935.0)*k6[i] + (6005943493.0/2108947869.0)*k7[i] + (393006217.0/1396673457.0)*k8[i] + (123872331.0/1001029789.0)*k9[i]
            ode_func(t + c10*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k10[i] = dt * dYdt[i]
            # k11
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (-1028468189.0/846180014.0)*k1[i] + (8478235783.0/508512852.0)*k4[i] + (1311729495.0/1432422823.0)*k5[i] + (-10304129995.0/1701304382.0)*k6[i] + (-48777925059.0/3047939560.0)*k7[i] + (15336726248.0/1032824649.0)*k8[i] + (-45442868181.0/3398467696.0)*k9[i] + (3065993473.0/597172653.0)*k10[i]
            ode_func(t + c11*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k11[i] = dt * dYdt[i]
            # k12
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (185892177.0/718116043.0)*k1[i] + (-3185094517.0/667107341.0)*k4[i] + (-477755414.0/1098053517.0)*k5[i] + (-703635378.0/230739211.0)*k6[i] + (5731566787.0/1027545527.0)*k7[i] + (5232866602.0/850066563.0)*k8[i] + (-4093664535.0/808688257.0)*k9[i] + (3962137247.0/1805957418.0)*k10[i] + (65686358.0/487910083.0)*k11[i]
            ode_func(t + c12*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k12[i] = dt * dYdt[i]
            # k13
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (403863854.0/491063109.0)*k1[i] + (-5068492393.0/434740067.0)*k4[i] + (-411421997.0/543043805.0)*k5[i] + (652783627.0/914296604.0)*k6[i] + (11173962825.0/925320556.0)*k7[i] + (-13158990841.0/6184727034.0)*k8[i] + (3936647629.0/1978049680.0)*k9[i] + (-160528059.0/685178525.0)*k10[i] + (248638103.0/1413531060.0)*k11[i]
            ode_func(t + c13*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k13[i] = dt * dYdt[i]

            # --- Compute 8th order proposal and 5th order for error ---
            for i in range(_NUM_VARS):
                y8_i = (Y[i] + b1*k1[i] + b6*k6[i] + b7*k7[i] + b8*k8[i] +
                        b9*k9[i] + b10*k10[i] + b11*k11[i] + b12*k12[i] + b13*k13[i])
                y5_i = (Y[i] + bs1*k1[i] + bs6*k6[i] + bs7*k7[i] + bs8*k8[i] +
                        bs9*k9[i] + bs10*k10[i] + bs11*k11[i] + bs12*k12[i] + bs13*k13[i])
                Y_temp[i] = y8_i  # Store proposal
                Y_err[i] = y8_i - y5_i # Store error estimate

            # --- Calculate error norm ---
            err_norm = comp_err_int(Y_err, _NUM_VARS) # Use library function

            # --- Adaptive step size control ---
            if err_norm <= tol:
                # Step accepted
                t += dt
                step_count += 1 # Increment accepted step count
                for i in range(_NUM_VARS):
                    Y[i] = Y_temp[i] # Update state
                # Update output array with accepted state
                for i in range(_NUM_VARS):
                    Y_out[idx, i] = Y[i]

                # --- Optional Renormalization (on accepted step) ---
                if renorm_interval > 0 and step_count % renorm_interval == 0:
                    # Loop over each deviation vector
                    for m_idx in range(_NUM_DEV_VECTORS):
                        # Calculate base index for this deviation vector set
                        base = _NUM_BASE_VARS + m_idx * _NUM_BASE_VARS
                        # Calculate L2 norm squared for this vector (size = num_base_vars)
                        norm_sq = 0.0
                        for i in range(_NUM_BASE_VARS):
                            val = Y[base + i]
                            norm_sq += val * val

                        inv_norm = 1.0 / math.sqrt(norm_sq)
                        # Renormalize
                        for i in range(_NUM_BASE_VARS):
                            Y[base + i] *= inv_norm

                # Calculate optimal step size factor for next step
                if err_norm == 0.0: scale = max_scale
                else: scale = safety * math.pow(tol / err_norm, exponent)
                scale = min(max_scale, max(min_scale, scale)) # Clamp scale
                dt = max(dt * scale, tiny) # Suggest new step size

            else:
                # Step rejected: Reduce step size and retry
                scale = safety * math.pow(tol / (err_norm + tiny), exponent) # Add tiny to denominator
                scale = max(min_scale, scale) # Only clamp minimum on rejection
                dt = dt * scale
                # Do not advance time or step_count, Y remains unchanged

            # Prevent dt from becoming excessively small
            if abs(dt) <= tiny:
                # Consider adding a flag or warning mechanism if desired
                break # Exit loop if dt is too small

        # --- End of while loop ---
        # Final state (at last accepted time t) is in Y_out

    return solver_adaptive_step_variational

# =========================================
# Solver Kernel Lyapunov Exponent QR Factory Function (Fixed Step)
# =========================================
def create_solver_kernel_fixed_LE_QR(ode_func, num_vars: int, num_base_vars: int, num_dev_vectors: int):
    """
    Factory for a fixed-step DP8 kernel that integrates base variables,
    deviation vectors, and auxiliary variables, AND calculates Lyapunov Exponent
    sums using periodic QR decomposition (Modified Gram-Schmidt) within the kernel.

    Assumes state vector Y structure:
    [ base_vars (num_base_vars),                 Indices 0 to nb-1
      dev_vec_0 (num_base_vars),                 Indices nb to 2*nb-1
      ...,
      dev_vec_{M-1} (num_base_vars),             Indices nb+M*nb-nb to nb+M*nb-1
      aux_var_0 (e.g., LD),                      Index nb*(1+M)
      LE_sum_0,                                  Index nb*(1+M)+1
      ...,
      LE_sum_{M-1}                               Index nb*(1+M)+M ]
    where nb = num_base_vars, M = num_dev_vectors.
    Total size = num_vars = nb*(1+M) + 1 + M.

    Args:
        ode_func: Numba CUDA device function `ode_func(t, Y, dYdt, params)`.
        num_vars (int): Total number of variables in the state vector Y.
                        Must match nb*(1+M) + 1 + M.
        num_base_vars (int): Number of base phase space variables (e.g., 2*N).
        num_dev_vectors (int): Number of deviation vectors / LEs to calculate (M).

    Returns:
        A Numba CUDA kernel function with signature:
        `kernel(Y0, t0, dt, steps, params, qr_interval, Y_out)`
          - Y0: Initial states (GPU array, must include space for LE sums initialized to 0).
          - t0: Initial time.
          - dt: Fixed time step.
          - steps: Number of integration steps.
          - params: Tuple of parameters for ode_func.
          - qr_interval: Perform QR decomp/LE accumulation every N steps.
                         Set <= 0 to disable LE calculation/QR.
          - Y_out: Output array for final states (GPU array). Contains state at
                   t0 + steps*dt, including final LE sums.
    """
    _NUM_VARS = num_vars
    _NUM_BASE_VARS = num_base_vars
    _NUM_DEV_VECTORS = num_dev_vectors

    # Calculate expected size based on structure (assuming 1 aux var before LE sums)
    _EXPECTED_NUM_VARS = _NUM_BASE_VARS * (1 + _NUM_DEV_VECTORS) + 1 + _NUM_DEV_VECTORS
    if _NUM_VARS != _EXPECTED_NUM_VARS:
         raise ValueError(f"num_vars ({_NUM_VARS}) does not match expected structure for LE calc: "
               f"base({_NUM_BASE_VARS}) + M*base({_NUM_DEV_VECTORS}) + aux(1) + LE_sums({_NUM_DEV_VECTORS}) = {_EXPECTED_NUM_VARS}. "
               "Check state vector definition.")

    # Define indices for clarity within kernel
    _DEV_VEC_START_IDX = _NUM_BASE_VARS
    _LE_SUM_START_IDX = _NUM_BASE_VARS * (1 + _NUM_DEV_VECTORS) + 1 # Start index of LE sums

    # Note on Local Array Sizing: Requires Numba specialization or fixed max sizes.
    # Assuming specialization via factory arguments works here.

    @cuda.jit
    def solver_fixed_step_LE_QR(Y0, t0, dt, steps, params, qr_interval, Y_out):
        """
        Fixed-step DP8 kernel with periodic QR LE calculation.
        Generated by create_solver_kernel_fixed_LE_QR.
        (Internal Numba JIT function)
        """
        idx = cuda.grid(1)
        if idx >= Y0.shape[0]:
            return

        # --- Local Variable Declarations ---
        Y = cuda.local.array(_NUM_VARS, float64)
        for i in range(_NUM_VARS): Y[i] = Y0[idx, i] # Initialize state including LE sums

        dYdt   = cuda.local.array(_NUM_VARS, float64)
        k1     = cuda.local.array(_NUM_VARS, float64); k2 = cuda.local.array(_NUM_VARS, float64)
        k3     = cuda.local.array(_NUM_VARS, float64); k4 = cuda.local.array(_NUM_VARS, float64)
        k5     = cuda.local.array(_NUM_VARS, float64); k6 = cuda.local.array(_NUM_VARS, float64)
        k7     = cuda.local.array(_NUM_VARS, float64); k8 = cuda.local.array(_NUM_VARS, float64)
        k9     = cuda.local.array(_NUM_VARS, float64); k10 = cuda.local.array(_NUM_VARS, float64)
        k11    = cuda.local.array(_NUM_VARS, float64); k12 = cuda.local.array(_NUM_VARS, float64)
        k13    = cuda.local.array(_NUM_VARS, float64)
        Y_temp = cuda.local.array(_NUM_VARS, float64)

        # Local arrays for QR decomposition
        A_local = cuda.local.array((_NUM_BASE_VARS, _NUM_DEV_VECTORS), dtype=float64)
        R_diag_local = cuda.local.array(_NUM_DEV_VECTORS, dtype=float64)

        # DP8 Constants
        c2=1/18; c3=1/12; c4=1/8; c5=5/16; c6=3/8; c7=59/400; c8=93/200
        c9=5490023248/9719169821; c10=13/20; c11=1201146811/1299019798; c12=1; c13=1
        b1=14005451/335480064; b6=-59238493/1068277825; b7=181606767/758867731
        b8=561292985/797845732; b9=-1041891430/1371343529; b10=760417239/1151165299
        b11=118820643/751138087; b12=-528747749/2220607170; b13=1/4

        t = t0
        # --- Main Fixed-Step Integration Loop ---
        for step_i in range(steps):
            # --- Standard DP8 Stages ---
            ode_func(t, Y, dYdt, params)
            for i in range(_NUM_VARS): k1[i] = dt * dYdt[i]
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + c2*k1[i]
            ode_func(t + c2*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k2[i] = dt * dYdt[i]
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (1/48)*k1[i] + (1/16)*k2[i]
            ode_func(t + c3*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k3[i] = dt * dYdt[i]
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (1/32)*k1[i] + (3/32)*k3[i]
            ode_func(t + c4*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k4[i] = dt * dYdt[i]
            for i in range(_NUM_VARS): Y_temp[i] = Y[i]+ (5/16)*k1[i]+ (-75/64)*k3[i]+ (75/64)*k4[i]
            ode_func(t + c5*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k5[i] = dt * dYdt[i]
            for i in range(_NUM_VARS): Y_temp[i] = Y[i]+ (3/80)*k1[i]+ (3/16)*k4[i]+ (3/20)*k5[i]
            ode_func(t + c6*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k6[i] = dt * dYdt[i]
            for i in range(_NUM_VARS): Y_temp[i] = Y[i]+ (29443841/614563906)*k1[i]+ (77736538/692538347)*k4[i]+ (-28693883/1125000000)*k5[i]+ (23124283/1800000000)*k6[i]
            ode_func(t + c7*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k7[i] = dt * dYdt[i]
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (16016141/946692911)*k1[i] + (61564180/158732637)*k4[i] + (22789713/633445777)*k5[i] + (545815736/2771057229)*k6[i] + (-180193667/1043307555)*k7[i]
            ode_func(t + c8*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k8[i] = dt * dYdt[i]
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (39632708/573591083)*k1[i] + (-433636366/683701615)*k4[i] + (-421739975/2616292301)*k5[i] + (100302831/723423059)*k6[i] + (790204164/839813087)*k7[i] + (800635310/3783071287)*k8[i]
            ode_func(t + c9*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k9[i] = dt * dYdt[i]
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (246121993/1340847787)*k1[i] + (-37695042795/15268766246)*k4[i] + (-309121744/1061227803)*k5[i] + (-12992083/490766935)*k6[i] + (6005943493/2108947869)*k7[i] + (393006217/1396673457)*k8[i] + (123872331/1001029789)*k9[i]
            ode_func(t + c10*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k10[i] = dt * dYdt[i]
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (-1028468189/846180014)*k1[i] + (8478235783/508512852)*k4[i] + (1311729495/1432422823)*k5[i] + (-10304129995/1701304382)*k6[i] + (-48777925059/3047939560)*k7[i] + (15336726248/1032824649)*k8[i] + (-45442868181/3398467696)*k9[i] + (3065993473/597172653)*k10[i]
            ode_func(t + c11*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k11[i] = dt * dYdt[i]
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (185892177/718116043)*k1[i] + (-3185094517/667107341)*k4[i] + (-477755414/1098053517)*k5[i] + (-703635378/230739211)*k6[i] + (5731566787/1027545527)*k7[i] + (5232866602/850066563)*k8[i] + (-4093664535/808688257)*k9[i] + (3962137247/1805957418)*k10[i] + (65686358/487910083)*k11[i]
            ode_func(t + c12*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k12[i] = dt * dYdt[i]
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (403863854/491063109)*k1[i] + (-5068492393/434740067)*k4[i] + (-411421997/543043805)*k5[i] + (652783627/914296604)*k6[i] + (11173962825/925320556)*k7[i] + (-13158990841/6184727034)*k8[i] + (3936647629/1978049680)*k9[i] + (-160528059/685178525)*k10[i] + (248638103/1413531060)*k11[i]
            ode_func(t + c13*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k13[i] = dt * dYdt[i]

            # --- Final Update ---
            for i in range(_NUM_VARS):
                Y[i] += (b1 * k1[i] + b6 * k6[i] + b7 * k7[i] + b8 * k8[i] +
                         b9 * k9[i] + b10 * k10[i] + b11 * k11[i] + b12 * k12[i] +
                         b13 * k13[i])
            t += dt # Advance time AFTER state update

            # --- QR Decomposition and LE Summation (at intervals) ---
            # Perform AFTER state Y is updated for the current step
            if qr_interval > 0 and (step_i + 1) % qr_interval == 0:
                # 1. Copy current dev vectors from Y to A_local (column-wise)
                for m in range(_NUM_DEV_VECTORS): # Columns of A_local
                    dev_vec_start_in_Y = _DEV_VEC_START_IDX + m * _NUM_BASE_VARS
                    for row in range(_NUM_BASE_VARS): # Rows
                        A_local[row, m] = Y[dev_vec_start_in_Y + row]

                # 2. Perform MGS QR (A_local becomes Q, R_diag_local gets diag(R))
                mgs_QR(A_local, R_diag_local, _NUM_BASE_VARS, _NUM_DEV_VECTORS)

                # 3. Accumulate log |R_jj| into state vector Y
                for j in range(_NUM_DEV_VECTORS):
                    R_jj = R_diag_local[j]
                    # Accumulate log of absolute value
                    Y[_LE_SUM_START_IDX + j] += math.log(abs(R_jj))

                # 4. Reorthonormalize: Copy Q (now in A_local) back to Y
                for m in range(_NUM_DEV_VECTORS): # Dev vectors in Y
                    dev_vec_start_in_Y = _DEV_VEC_START_IDX + m * _NUM_BASE_VARS
                    for row in range(_NUM_BASE_VARS): # Rows
                        Y[dev_vec_start_in_Y + row] = A_local[row, m]
            # --- End QR ---

        # --- End Integration Loop ---
        # Write final state (including LE sums) to output
        for i in range(_NUM_VARS):
            Y_out[idx, i] = Y[i]

    return solver_fixed_step_LE_QR


# =========================================
# Solver Kernel Lyapunov Exponent QR Factory Function (Adaptive Step)
# =========================================
def create_solver_kernel_LE_QR(ode_func, num_vars: int, num_base_vars: int, num_dev_vectors: int):
    """
    Factory for an adaptive DP8(5) kernel that integrates base variables,
    deviation vectors, and auxiliary variables, AND calculates Lyapunov Exponent
    sums using periodic QR decomposition (Modified Gram-Schmidt) within the kernel.

    Assumes state vector Y structure:
    [ base_vars (num_base_vars),                 Indices 0 to nb-1
      dev_vec_0 (num_base_vars),                 Indices nb to 2*nb-1
      ...,
      dev_vec_{M-1} (num_base_vars),             Indices nb+M*nb-nb to nb+M*nb-1
      aux_var_0 (e.g., LD),                      Index nb*(1+M)
      LE_sum_0,                                  Index nb*(1+M)+1
      ...,
      LE_sum_{M-1}                               Index nb*(1+M)+M ]
    where nb = num_base_vars, M = num_dev_vectors.
    Total size = num_vars = nb*(1+M) + 1 + M.

    Args:
        ode_func: Numba CUDA device function `ode_func(t, Y, dYdt, params)`.
        num_vars (int): Total number of variables in the state vector Y.
                        Must match nb*(1+M) + 1 + M.
        num_base_vars (int): Number of base phase space variables (e.g., 2*N).
        num_dev_vectors (int): Number of deviation vectors / LEs to calculate (M).

    Returns:
        A Numba CUDA kernel function with signature:
        `kernel(Y0, t0, t_final, params, tol, dt_initial, max_steps, qr_interval, Y_out)`
          - Y0: Initial states (GPU array, must include space for LE sums initialized to 0).
          - t0: Initial time.
          - t_final: Target final time.
          - params: Tuple of parameters for ode_func.
          - tol: Absolute error tolerance for step control.
          - dt_initial: Initial guess for the time step.
          - max_steps: Maximum number of adaptive steps allowed.
          - qr_interval: Perform QR decomp/LE accumulation every N *accepted* steps.
                         Set <= 0 to disable LE calculation/QR.
          - Y_out: Output array for final states (GPU array). Contains state at
                   last accepted step time, including final LE sums.
    """
    _NUM_VARS = num_vars
    _NUM_BASE_VARS = num_base_vars
    _NUM_DEV_VECTORS = num_dev_vectors

    # Calculate expected size based on structure (assuming 1 aux var before LE sums)
    _EXPECTED_NUM_VARS = _NUM_BASE_VARS * (1 + _NUM_DEV_VECTORS) + 1 + _NUM_DEV_VECTORS
    if _NUM_VARS != _EXPECTED_NUM_VARS:
         # Provide a more informative warning/error
         raise ValueError(f"num_vars ({_NUM_VARS}) does not match expected structure for LE calc: "
               f"base({_NUM_BASE_VARS}) + M*base({_NUM_DEV_VECTORS}) + aux(1) + LE_sums({_NUM_DEV_VECTORS}) = {_EXPECTED_NUM_VARS}. "
               "Check state vector definition.")

    # Define indices for clarity within kernel
    _DEV_VEC_START_IDX = _NUM_BASE_VARS
    # _AUX_VAR_IDX = _NUM_BASE_VARS * (1 + _NUM_DEV_VECTORS) # Index of the single aux var
    _LE_SUM_START_IDX = _NUM_BASE_VARS * (1 + _NUM_DEV_VECTORS) + 1 # Start index of LE sums

    # Note on Local Array Sizing:
    # Sizing local arrays A_local and R_diag_local based on _NUM_BASE_VARS and
    # _NUM_DEV_VECTORS relies on Numba specializing the kernel via the factory.
    # If these dimensions vary significantly at runtime in a way Numba cannot
    # handle via specialization, you might need to use fixed maximum sizes
    # (e.g., MAX_BASE_VARS, MAX_DEV_VECTORS) known at compile time.

    @cuda.jit
    def solver_adaptive_step_LE_QR(Y0, t0, t_final, params, tol, dt_initial, max_steps, qr_interval, Y_out):
        """
        Adaptive-step DP8(5) kernel with periodic QR-based LE calculation.
        Generated by create_solver_kernel_LE_QR.
        (Internal Numba JIT function)
        """
        idx = cuda.grid(1)
        if idx >= Y0.shape[0]:
            return

        # --- Local Variable Declarations ---
        # State and RK stages (total size)
        Y = cuda.local.array(_NUM_VARS, float64)
        for i in range(_NUM_VARS): Y[i] = Y0[idx, i] # Initialize state including LE sums

        dYdt   = cuda.local.array(_NUM_VARS, float64)
        k1     = cuda.local.array(_NUM_VARS, float64); k2 = cuda.local.array(_NUM_VARS, float64)
        k3     = cuda.local.array(_NUM_VARS, float64); k4 = cuda.local.array(_NUM_VARS, float64)
        k5     = cuda.local.array(_NUM_VARS, float64); k6 = cuda.local.array(_NUM_VARS, float64)
        k7     = cuda.local.array(_NUM_VARS, float64); k8 = cuda.local.array(_NUM_VARS, float64)
        k9     = cuda.local.array(_NUM_VARS, float64); k10 = cuda.local.array(_NUM_VARS, float64)
        k11    = cuda.local.array(_NUM_VARS, float64); k12 = cuda.local.array(_NUM_VARS, float64)
        k13    = cuda.local.array(_NUM_VARS, float64)
        Y_temp = cuda.local.array(_NUM_VARS, float64)
        Y_err  = cuda.local.array(_NUM_VARS, float64)

        # Local arrays for QR decomposition
        # Sized based on the number of base variables and deviation vectors
        A_local = cuda.local.array((_NUM_BASE_VARS, _NUM_DEV_VECTORS), dtype=float64)
        R_diag_local = cuda.local.array(_NUM_DEV_VECTORS, dtype=float64)

        # DP8(5) Coefficients
        c2=1/18; c3=1/12; c4=1/8; c5=5/16; c6=3/8; c7=59/400; c8=93/200
        c9=5490023248/9719169821; c10=13/20; c11=1201146811/1299019798; c12=1; c13=1
        b1=14005451/335480064; b6=-59238493/1068277825; b7=181606767/758867731
        b8=561292985/797845732; b9=-1041891430/1371343529; b10=760417239/1151165299
        b11=118820643/751138087; b12=-528747749/2220607170; b13=1/4
        bs1=13451932/455176623; bs6=-808719846/976000145; bs7=1757004468/5645159321
        bs8=656045339/265891186; bs9=-3867574721/1518517206; bs10=465885868/322736535
        bs11=53011238/667516719; bs12=2/45; bs13=0

        # Adaptive step control parameters
        t = t0
        dt = dt_initial
        step_count = 0 # Count accepted steps
        safety = 0.9; min_scale = 0.2; max_scale = 5.0
        exponent = 1.0 / 6.0
        tiny = 1e-30

        # --- Main adaptive integration loop ---
        while (t < t_final) and (step_count < max_steps):
            if (t + dt > t_final): dt = t_final - t

            # --- Compute RK stages k1-k13 ---
            # (Identical logic using ode_func)
            ode_func(t, Y, dYdt, params)
            for i in range(_NUM_VARS): k1[i] = dt * dYdt[i]
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + c2*k1[i]
            ode_func(t + c2*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k2[i] = dt * dYdt[i]
            # ... k3 to k13 ...
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (1.0/48.0)*k1[i] + (1.0/16.0)*k2[i]
            ode_func(t + c3*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k3[i] = dt * dYdt[i]
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (1.0/32.0)*k1[i] + (3.0/32.0)*k3[i]
            ode_func(t + c4*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k4[i] = dt * dYdt[i]
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (5.0/16.0)*k1[i] + (-75.0/64.0)*k3[i] + (75.0/64.0)*k4[i]
            ode_func(t + c5*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k5[i] = dt * dYdt[i]
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (3.0/80.0)*k1[i] + (3.0/16.0)*k4[i] + (3.0/20.0)*k5[i]
            ode_func(t + c6*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k6[i] = dt * dYdt[i]
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (29443841.0/614563906.0)*k1[i] + (77736538.0/692538347.0)*k4[i] + (-28693883.0/1125000000.0)*k5[i] + (23124283.0/1800000000.0)*k6[i]
            ode_func(t + c7*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k7[i] = dt * dYdt[i]
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (16016141.0/946692911.0)*k1[i] + (61564180.0/158732637.0)*k4[i] + (22789713.0/633445777.0)*k5[i] + (545815736.0/2771057229.0)*k6[i] + (-180193667.0/1043307555.0)*k7[i]
            ode_func(t + c8*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k8[i] = dt * dYdt[i]
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (39632708.0/573591083.0)*k1[i] + (-433636366.0/683701615.0)*k4[i] + (-421739975.0/2616292301.0)*k5[i] + (100302831.0/723423059.0)*k6[i] + (790204164.0/839813087.0)*k7[i] + (800635310.0/3783071287.0)*k8[i]
            ode_func(t + c9*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k9[i] = dt * dYdt[i]
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (246121993.0/1340847787.0)*k1[i] + (-37695042795.0/15268766246.0)*k4[i] + (-309121744.0/1061227803.0)*k5[i] + (-12992083.0/490766935.0)*k6[i] + (6005943493.0/2108947869.0)*k7[i] + (393006217.0/1396673457.0)*k8[i] + (123872331.0/1001029789.0)*k9[i]
            ode_func(t + c10*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k10[i] = dt * dYdt[i]
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (-1028468189.0/846180014.0)*k1[i] + (8478235783.0/508512852.0)*k4[i] + (1311729495.0/1432422823.0)*k5[i] + (-10304129995.0/1701304382.0)*k6[i] + (-48777925059.0/3047939560.0)*k7[i] + (15336726248.0/1032824649.0)*k8[i] + (-45442868181.0/3398467696.0)*k9[i] + (3065993473.0/597172653.0)*k10[i]
            ode_func(t + c11*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k11[i] = dt * dYdt[i]
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (185892177.0/718116043.0)*k1[i] + (-3185094517.0/667107341.0)*k4[i] + (-477755414.0/1098053517.0)*k5[i] + (-703635378.0/230739211.0)*k6[i] + (5731566787.0/1027545527.0)*k7[i] + (5232866602.0/850066563.0)*k8[i] + (-4093664535.0/808688257.0)*k9[i] + (3962137247.0/1805957418.0)*k10[i] + (65686358.0/487910083.0)*k11[i]
            ode_func(t + c12*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k12[i] = dt * dYdt[i]
            for i in range(_NUM_VARS): Y_temp[i] = Y[i] + (403863854.0/491063109.0)*k1[i] + (-5068492393.0/434740067.0)*k4[i] + (-411421997.0/543043805.0)*k5[i] + (652783627.0/914296604.0)*k6[i] + (11173962825.0/925320556.0)*k7[i] + (-13158990841.0/6184727034.0)*k8[i] + (3936647629.0/1978049680.0)*k9[i] + (-160528059.0/685178525.0)*k10[i] + (248638103.0/1413531060.0)*k11[i]
            ode_func(t + c13*dt, Y_temp, dYdt, params)
            for i in range(_NUM_VARS): k13[i] = dt * dYdt[i]


            # --- Compute proposal and error ---
            for i in range(_NUM_VARS):
                y8_i = (Y[i] + b1*k1[i] + b6*k6[i] + b7*k7[i] + b8*k8[i] +
                        b9*k9[i] + b10*k10[i] + b11*k11[i] + b12*k12[i] + b13*k13[i])
                y5_i = (Y[i] + bs1*k1[i] + bs6*k6[i] + bs7*k7[i] + bs8*k8[i] +
                        bs9*k9[i] + bs10*k10[i] + bs11*k11[i] + bs12*k12[i] + bs13*k13[i])
                Y_temp[i] = y8_i
                Y_err[i] = y8_i - y5_i

            err_norm = comp_err_int(Y_err, _NUM_VARS)

            # --- Adaptive step size control ---
            if err_norm <= tol:
                # Step accepted
                t += dt
                step_count += 1
                for i in range(_NUM_VARS): Y[i] = Y_temp[i] # Update state

                # --- QR Decomposition and LE Summation ---
                if qr_interval > 0 and step_count % qr_interval == 0:
                    # 1. Copy dev vectors from Y to A_local (column-wise)
                    for m in range(_NUM_DEV_VECTORS): # Columns of A_local
                        dev_vec_start_in_Y = _DEV_VEC_START_IDX + m * _NUM_BASE_VARS
                        for row in range(_NUM_BASE_VARS): # Rows
                            A_local[row, m] = Y[dev_vec_start_in_Y + row]

                    # 2. Perform MGS QR (A_local becomes Q, R_diag_local gets diag(R))
                    mgs_QR(A_local, R_diag_local, _NUM_BASE_VARS, _NUM_DEV_VECTORS)

                    # 3. Accumulate log |R_jj| into state vector Y
                    for j in range(_NUM_DEV_VECTORS):
                        R_jj = R_diag_local[j]
                        # Note: Division by time happens on host after integration
                        Y[_LE_SUM_START_IDX + j] += math.log(abs(R_jj))

                    # 4. Reorthonormalize: Copy Q (now in A_local) back to Y
                    for m in range(_NUM_DEV_VECTORS): # Dev vectors in Y
                        dev_vec_start_in_Y = _DEV_VEC_START_IDX + m * _NUM_BASE_VARS
                        for row in range(_NUM_BASE_VARS): # Rows
                            Y[dev_vec_start_in_Y + row] = A_local[row, m]
                # --- End QR ---

                # Update output array with accepted state (incl. updated LE sums & Q)
                for i in range(_NUM_VARS): Y_out[idx, i] = Y[i]

                # Calculate step size factor for next step
                if err_norm == 0.0: scale = max_scale
                else: scale = safety * math.pow(tol / err_norm, exponent)
                scale = min(max_scale, max(min_scale, scale))
                dt = max(dt * scale, tiny)

            else:
                # Step rejected
                scale = safety * math.pow(tol / (err_norm + tiny), exponent)
                scale = max(min_scale, scale)
                dt = dt * scale

            if abs(dt) <= tiny: break

        # --- End of while loop ---
        # Final state (at last accepted time t) is in Y_out

    return solver_adaptive_step_LE_QR