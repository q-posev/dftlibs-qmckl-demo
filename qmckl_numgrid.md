# Demo: TREX meets dftlibs

In this demo we will numerically calculate some molecular integrals using the pre-computed quantum chemistry data from the `trexio` file, numerical integration grid from the `numgrid` package, atomic orbital values and some matrix operations from the `qmckl` library and finally the exchange-correlation contribution from the `xcfun` package.

## Setup

First we load all necessary Python packages


```python
import numpy as np
import qmckl
import numgrid
import xcfun
import trexio
from timeit import default_timer as timer
```

Creating the QMCkl context before proceeding to use the library


```python
ctx = qmckl.context_create()
```

Load the quamtum chemistry data from the `trexio_filename` file (HDF5 file in the TREXIO format)


```python
trexio_filename = "trexio-files/h2o-sto3g.h5"
qmckl.trexio_read(ctx, trexio_filename)
# read the AO overlap matrix from the TREXIO file for benchmarking later on
with trexio.File(trexio_filename, 'r', trexio.TREXIO_AUTO) as tf:
    overlap_ao_ref = trexio.read_ao_1e_int_overlap(tf)
```

Get some basic information about the molecule from the context


```python
nucleus_num    = qmckl.get_nucleus_num(ctx)
nucleus_coord  = qmckl.get_nucleus_coord(ctx, 'N', nucleus_num*3)
nucleus_charge = qmckl.get_nucleus_charge(ctx, nucleus_num)
```

## Numerical calculation of the atomic orbital overlap 

Check that the basis is using Gaussian functions


```python
try:
    assert qmckl.get_ao_basis_type(ctx) ==  'G'
except AssertionError:
    print("Only Gaussian basis functions can be used in this tutorial.")
```

Get the basis set information from the context


```python
# dimensions
ao_num    = qmckl.get_ao_basis_ao_num(ctx)
shell_num = qmckl.get_ao_basis_shell_num(ctx)
prim_num  = qmckl.get_ao_basis_prim_num(ctx)
# mappings
shell_ang_mom    = qmckl.get_ao_basis_shell_ang_mom(ctx, shell_num)
# nucleus_index and shell_prim_index of QMCkl are still in the old format (trexio < 2.0)
nucleus_index     = qmckl.get_ao_basis_nucleus_index(ctx, nucleus_num)    
shell_prim_index  = qmckl.get_ao_basis_shell_prim_index(ctx, shell_num)
nucleus_shell_num = qmckl.get_ao_basis_nucleus_shell_num(ctx, nucleus_num)
shell_prim_num    = qmckl.get_ao_basis_shell_prim_num(ctx, shell_num)
# normalization factors
shell_factor = qmckl.get_ao_basis_shell_factor(ctx, shell_num)
prim_factor  = qmckl.get_ao_basis_prim_factor(ctx, prim_num)
# basis set parameters
coefficient = qmckl.get_ao_basis_coefficient(ctx, prim_num)
exponent    = qmckl.get_ao_basis_exponent(ctx, prim_num)
```

Setup of the basis set in `numgrid` requires to prepare a list of `alpha_max` (per atom) 
and a dict of `alpha_min` (per shell per atom) where alpha is an exponent of the primitive


```python
# TODO: fix bug for larger moleculess (e.g. Alz_small.h5)
id0 = 0
id1 = 0
shift = 0
exponents = []
for shell_num in nucleus_shell_num:
    idx = shell_num + id0    
    idy = 0
    exp_atomic = []
    for j in range(shell_num):
        # per-shell splitting
        idy = shell_prim_num[j+shift] + id1
        exp_atomic.append(exponent[id1:idy])
        id1 = idy
    
    exponents.append(exp_atomic)
    id0 = idx
    id1 = idy
    shift = shell_num
```


```python
tmp_max = [
    [max(exp_per_l) if len(exp_per_l) > 1 else exp_per_l[0] for exp_per_l in exp]
    for exp in exponents
]
tmp_min = [
    [min(exp_per_l) for exp_per_l in exp]
    for exp in exponents
]

alpha_max = [max(e) for e in tmp_max]

alpha_min = [{
    shell_ang_mom[index+index2] : value 
    for (index,value) in enumerate(min_l_per_atom)
    } for (index2,min_l_per_atom) in enumerate(tmp_min)
]
```

Specify some parameters that define the numerical integration grid (following `README` of the `numgrid`)


```python
radial_precision = 1.0e-12
#min_num_angular_points = 86
#max_num_angular_points = 302
min_num_angular_points = 770
max_num_angular_points = 3470
proton_charges = np.array(nucleus_charge, dtype=np.int32)
center_coordinates_bohr = [
    tuple(coord) for coord in nucleus_coord.reshape(nucleus_num,3)
]
```

Generate integration grids for each atom (this step can be parallelized using MPI or `multiprocessing`)


```python
num_points = 0
coord_all  = []
weight_all = []
for center_index in range(len(center_coordinates_bohr)):
    coordinates, weights  = numgrid.atom_grid(
        alpha_min[center_index],
        alpha_max[center_index],
        radial_precision,
        min_num_angular_points,
        max_num_angular_points,
        proton_charges,
        center_index,
        center_coordinates_bohr,
        hardness=3
    )
    coord_all  += coordinates
    weight_all += weights
    num_points += len(weights)
```


```python
print("Number of points:", num_points)
```

    Number of points: 450926


Prepare the context before computing the atomic orbital (AO) values on a grid


```python
coord_all_flat  = np.asarray(coord_all).flatten()
weight_all_flat = np.asarray(weight_all).flatten()

qmckl.set_point(ctx, 'N', num_points, coord_all_flat)
```

Calculate the AO values `ao_v_w` on a grid (`w` is for weighted)


```python
start = timer()
ao_v = qmckl.get_ao_basis_ao_value(ctx, num_points*ao_num)
end = timer()
print(f"Timing for qmckl_ao_value: {(end - start):.6f}") # Time in seconds

ao_v.shape = (num_points, ao_num)
# elementwise multiplication of numpy arrays
tmp = ao_v.T * weight_all_flat
ao_v_w = tmp.T
# flatten the matrices as required by the QMCkl API
ao_v.shape   = num_points*ao_num
ao_v_w.shape = num_points*ao_num
```

    Timing for qmckl_ao_value: 0.040345


Compute the overlap using the `qmckl_dgemm_safe` function


```python
start = timer()
overlap = qmckl.dgemm_safe(
    ctx, 'N', 'T', 
    ao_num, ao_num, num_points, 1.0, 
    ao_v_w, ao_num,
    ao_v, ao_num, 0.0,
    ao_num*ao_num, # ---> dimension of the returned array (a.k.a size_max)
    ao_num
)
end = timer()
print(f"Timing for qmckl_dgemm: {(end - start):.6f}") # Time in seconds
```

    Timing for qmckl_dgemm: 0.019855


Compute the adjugate matrix using the `qmckl_adjugate_safe` function


```python
overlap_adj, detA = qmckl.adjugate_safe(
    ctx, ao_num, overlap, ao_num, 
    ao_num*ao_num, # ---> dimension of the returned array (a.k.a size_max)
    ao_num
)
```

QMCkl operates on flat (contiguous) arrays, but for matrix operations in `numpy` it is better to reshape 1D arrays into 2D matrices.


```python
overlap.shape     = (ao_num, ao_num)
overlap_adj.shape = (ao_num, ao_num)
ao_v.shape        = (num_points, ao_num)
ao_v_w.shape      = (num_points, ao_num)
```

Now compute the overlap matrix using the `numpy` routines instead of `qmckl_dgemm_safe`


```python
start = timer()
overlap_np = ao_v_w.T @ ao_v
end = timer()
print(f"Timing for numpy dgemm: {(end - start):.6f}") # Time in seconds
assert np.allclose(overlap_np, overlap)
```

    Timing for numpy dgemm: 0.023641


And also compare with the overlap matrix that was stored in the TREXIO file. **Note:** we have to lower the tolerance here due to the differences between the reference method (code) that produced the TREXIO file and the grid used in this demo.


```python
# atol=1e-8 works with larger grids like [770, 3470] ; atol=1e-7 is for the smaller ones
print(np.linalg.norm(overlap_ao_ref-overlap))
print(np.linalg.norm(overlap_ao_ref-overlap_np))
assert np.allclose(overlap_ao_ref, overlap, atol=1e-8)
assert np.allclose(overlap_ao_ref, overlap_np, atol=1e-8)
```

    1.2363966849449855e-06
    1.2363966849309476e-06


Dummy check that the computed determinant and adjugate matrix are actually consistent


```python
det_ref = np.diagonal(overlap_adj @ overlap)[0]
print(f"Error: {(det_ref-detA):0.4e}")
```

    Error: 5.5511e-17



```python
try:
    assert(np.equal(det_ref, detA))
except AssertionError:
    try:
        assert(np.isclose(det_ref, detA, atol=1e-14))
    except AssertionError:
        print("Numerical bug detected!")
```

## Numerical calculation of the exchange-correlation contribution via DFT

Get the molecular orbital (MO) information from the context


```python
mo_num = qmckl.get_mo_basis_mo_num(ctx)
coefficients = qmckl.get_mo_basis_coefficient(ctx, mo_num*ao_num)
coefficients.shape = (mo_num,ao_num)
```

Get the number of spin-up and spin-down electrons. Writing for an open-shell case from the beginning to avoiding messing up later.


```python
el_num_up = qmckl.get_electron_up_num(ctx)
el_num_dn = qmckl.get_electron_down_num(ctx)
```

Compute the ground state density matrix in the AO basis. Normally one should use different set of the MO coefficients for spin-up and spin-down electrons, but this is not the case yet in `qmckl` or `trexio`, so we use the closed-shell set MO coefficients from the context. **TODO:** change later when running the SCF cycle.


```python
p_mat_up = coefficients[0:el_num_up,:].T @ coefficients[0:el_num_up,:]
p_mat_dn = coefficients[0:el_num_dn,:].T @ coefficients[0:el_num_dn,:]
```

Compute the number of electrons numerically as a product of the density and overlap matrices


```python
# tensordot on matrices is equivalent to a double dot product (sum_i sum_j A_ij*B_ij)
el_num_up_calc = np.tensordot(p_mat_up, overlap)
el_num_dn_calc = np.tensordot(p_mat_dn, overlap)
```


```python
el_num      = el_num_up + el_num_dn
el_num_calc = el_num_up_calc + el_num_dn_calc
assert np.around(el_num) == np.around(el_num_calc)
```

Computing the density on a grid now in order to evaluate the XC contribution later 


```python
ao_v.shape = (num_points, ao_num)
```

### Version 1: explicit for loops (very slow)


```python
# compute density on a grid with loops (the ugly)
#n_r = np.zeros(num_points)
#for i in range(num_points):
#    for mu in range(ao_num):
#        for nu in range(ao_num):
#            n_r[i] += p_mat[mu, nu] * ao_v[i, mu] * ao_v[i, nu]
```

### Version 2: numpy matrix operations (fast)

First compute the matrix product, i.e. `D_mk = sum_n (P_mn * AO_nk)`


```python
n_up_tmp = p_mat_up @ ao_v.T
n_dn_tmp = p_mat_dn @ ao_v.T
```

Then compute the row-wise dot product, i.e. `sum_m (D_mk * AO_mk)`


```python
# Solution 1: numpy.sum over given axis (the bad)
n_up0 = np.sum(n_up_tmp * ao_v.T, axis=0)
n_dn0 = np.sum(n_dn_tmp * ao_v.T, axis=0)
```


```python
# Solution 2: numpy.einsum with tensor contraction (the good)
n_up  = np.einsum('ij, ij->j', n_up_tmp, ao_v.T)
n_dn  = np.einsum('ij, ij->j', n_dn_tmp, ao_v.T)
density = np.array([n_up, n_dn])
```

Some dummy checks


```python
assert n_up.size == num_points and n_dn.size == num_points
assert np.allclose(n_up, n_up0, rtol=1e-15, atol=1e-12)
assert np.allclose(n_dn, n_dn0, rtol=1e-15, atol=1e-12)
#assert np.allclose(n_1, n_r, rtol=1e-15, atol=1e-12)
#assert np.allclose(n_2, n_r, rtol=1e-15, atol=1e-12)
```

## Computing the XC energy contribution

Firstly, we set up some parameters of XCFun library (see the documentation)


```python
fun_xc_name   = 'LDA'
fun_xc_weight = 1.0
fun_xc        = xcfun.Functional({fun_xc_name: fun_xc_weight})
```

Now we are ready to evaluate the XC funtional contribution using the previously computed density on a grid


```python
result = fun_xc.eval_potential_ab(density.T)
# the output contains energy density and XC potential (spin-resolved) on a grid
energy_density = result[:, 0]
potential_up   = result[:, 1]
potential_dn   = result[:, 2]
```


```python
energy_density_test = fun_xc.eval_energy_n(n_up + n_dn)
assert np.allclose(energy_density_test, energy_density)
```

Finally, to compute the XC energy contribution we need to integrate the energy density on our grid with weights


```python
e_xc   = energy_density @ weight_all_flat
print(f"XC energy contribution with the {fun_xc_name} functional = {e_xc}")
```

    XC energy contribution with the LDA functional = -8.873142309483445


## Final

Destroy the context to clear the memory allocated internally by the QMCkl


```python
qmckl.context_destroy(ctx)
```
