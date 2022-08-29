```python
import numpy as np
import qmckl
import numgrid
import xcfun
```


```python
ctx = qmckl.context_create()
```


```python
trexio_filename = "trexio-files/h2o-sto3g.h5"
qmckl.trexio_read(ctx, trexio_filename)
```


```python
nucleus_num    = qmckl.get_nucleus_num(ctx)
nucleus_coord  = qmckl.get_nucleus_coord(ctx, 'N', nucleus_num*3)
nucleus_charge = qmckl.get_nucleus_charge(ctx, nucleus_num)
```


```python
print(nucleus_coord.reshape((nucleus_num,3)))
```

    [[ 0.          0.          0.        ]
     [-1.43042871  0.         -1.10715696]
     [ 1.43042871  0.         -1.10715696]]



```python
try:
    assert qmckl.get_ao_basis_type(ctx) ==  'G'
except AssertionError:
    print("Only Gaussian basis functions can be used in this tutorial.")
```


```python
# dimensions
ao_num    = qmckl.get_ao_basis_ao_num(ctx)
shell_num = qmckl.get_ao_basis_shell_num(ctx)
prim_num  = qmckl.get_ao_basis_prim_num(ctx)
# mappings
shell_ang_mom    = qmckl.get_ao_basis_shell_ang_mom(ctx, shell_num)
# nucleus_index and shell_prim_index of QMCkl are still in the old format (trexio < 2.0)
nucleus_index     = qmckl.get_ao_basis_nucleus_index(ctx, nucleus_num)     # should be shell_num
shell_prim_index  = qmckl.get_ao_basis_shell_prim_index(ctx, shell_num)    # should be prim_num
nucleus_shell_num = qmckl.get_ao_basis_nucleus_shell_num(ctx, nucleus_num) # not needed at all
shell_prim_num    = qmckl.get_ao_basis_shell_prim_num(ctx, shell_num)      # not needed at all
# normalization factors
shell_factor = qmckl.get_ao_basis_shell_factor(ctx, shell_num)
prim_factor  = qmckl.get_ao_basis_prim_factor(ctx, prim_num)
# basis set parameters
coefficient = qmckl.get_ao_basis_coefficient(ctx, prim_num)
exponent    = qmckl.get_ao_basis_exponent(ctx, prim_num)
#print(qmckl.last_error(ctx))
```


```python
#print(ao_num, shell_num, prim_num)
#print(nucleus_index, nucleus_shell_num)
#print(shell_prim_index, shell_ang_mom,  shell_prim_num)
```


```python
# numgrid basis set up requires to prepare a list of alpha_max (per atom) 
# and a dict of alpha_min (per shell per atom) where alpha is an exponent

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
#print(exponents)
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


```python
# numgrid setup (following README.md)
radial_precision = 1.0e-12
min_num_angular_points = 86
max_num_angular_points = 302
#min_num_angular_points = 770
#max_num_angular_points = 3470

proton_charges = np.array(nucleus_charge, dtype=np.int32)
center_coordinates_bohr = [
    tuple(coord) for coord in nucleus_coord.reshape(nucleus_num,3)
]
```


```python
# loop to generate per-atom grids
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
# Alternative functions from the numgrid Python API
#
# radial grid (LMG) using explicit basis set parameters for one atom
#radii, weights = numgrid.radial_grid_lmg(
#    alpha_min=alpha_min[0],
#    alpha_max=alpha_max[0],
#    radial_precision=radial_precision,
#    proton_charge=proton_charges[0]
#)
# radial grid with 100 points using Krack-Koster approach
#radii, weights = numgrid.radial_grid_kk(num_points=100)
# angular grid with 14 points
#coordinates, weights = numgrid.angular_grid(num_points=14)
```


```python
print("Number of points:", num_points)
```

    Number of points: 41234



```python
coord_all_flat  = np.asarray(coord_all).flatten()
weight_all_flat = np.asarray(weight_all).flatten()

qmckl.set_point(ctx, 'N', num_points, coord_all_flat)
```


```python
ao_v = qmckl.get_ao_basis_ao_value(ctx, num_points*ao_num)
ao_v.shape = (num_points, ao_num)
# elementwise multiplication of numpy arrays
tmp = ao_v.T * weight_all_flat
ao_v_w = tmp.T

ao_v.shape   = num_points*ao_num
ao_v_w.shape = num_points*ao_num
```


```python
#ao_vgl = qmckl.get_ao_basis_ao_vgl(ctx, 5*num_points*ao_num)
#for i in range(num_points):
#    assert(ao_v[i*ao_num]==ao_vgl[i*5*ao_num])
```


```python
overlap = qmckl.dgemm_safe(
    ctx, 'N', 'T', 
    ao_num, ao_num, num_points, 1.0, 
    ao_v_w, ao_num,
    ao_v, ao_num, 0.0,
    ao_num*ao_num, # ---> dimension of the returned array (a.k.a size_max)
    ao_num
)
```


```python
overlap_adj, detA = qmckl.adjugate_safe(
    ctx, ao_num, overlap, ao_num, 
    ao_num*ao_num, # ---> dimension of the returned array (a.k.a size_max)
    ao_num
)
```


```python
overlap.shape     = (ao_num,ao_num)
overlap_adj.shape = (ao_num,ao_num)
```


```python
ao_v.shape   = (num_points, ao_num)
ao_v_w.shape = (num_points, ao_num)
# TODO: measure performance of numpy @ matrix product VS qmckl_dgemm_safe
overlap_np = ao_v_w.T @ ao_v
assert np.allclose(overlap_np, overlap)
```


```python
#print(overlap_mat[0,0], overlap_mat[ao_num-1,ao_num-1])
#print(np.diagonal(overlap_mat).copy())
#print(np.diagonal(overlap_adj_mat @ overlap_mat))
#print(detA)
```


```python
det_ref = np.diagonal(overlap_adj @ overlap)[0]
print(f"Error: {(det_ref-detA):0.4e}")
```

    Error: 0.0000e+00



```python
try:
    assert(np.equal(det_ref, detA))
except AssertionError:
    try:
        assert(np.isclose(det_ref, detA, atol=1e-14))
    except AssertionError:
        print("Numerical bug detected!")
```


```python
mo_num = qmckl.get_mo_basis_mo_num(ctx)
coefficients = qmckl.get_mo_basis_coefficient(ctx, mo_num*ao_num)
coefficients.shape = (mo_num,ao_num)
```


```python
if qmckl.get_electron_up_num(ctx) ==  qmckl.get_electron_down_num(ctx):
    el_num = int(qmckl.get_electron_num(ctx)/2)
else:
    raise Exception("Open-shell case is not supported yet.")
```


```python
p_mat = coefficients[0:el_num,:].T @ coefficients[0:el_num,:]
```


```python
# tensordot on matrices is equivalent to a double dot product (sum_i sum_j A_ij*B_ij)
el_num_calc = np.tensordot(p_mat, overlap)
```


```python
#print(el_num, el_num_calc)
assert np.around(el_num) == np.around(el_num_calc)
```


```python
ao_v.shape = (num_points, ao_num)
# compute density on a grid with loops (the ugly)
#n_r = np.zeros(num_points)
#for i in range(num_points):
#    for mu in range(ao_num):
#        for nu in range(ao_num):
#            n_r[i] += p_mat[mu, nu] * ao_v[i, mu] * ao_v[i, nu]
```


```python
# compute density on a grid using numpy matrix operations
# first compute the matrix product, i.e. D_mk = sum_n (P_mn * AO_nk)
n_tmp = p_mat @ ao_v.T
# then compute the row-wise dot product, i.e. sum_m (D_mk * AO_mk)
# Solution 1: numpy.sum over given axis (the bad)
n_1 = np.sum(n_tmp * ao_v.T, axis=0)
# Solution 2: numpy.einsum with tensor contraction (the good)
n_2 = np.einsum('ij, ij->j', n_tmp, ao_v.T)
```


```python
#print(max(n_dgemm))
assert n_1.size == num_points
assert n_2.size == num_points
assert np.allclose(n_1, n_2, rtol=1e-15, atol=1e-12)
#assert np.allclose(n_1, n_r, rtol=1e-15, atol=1e-12)
#assert np.allclose(n_2, n_r, rtol=1e-15, atol=1e-12)
```


```python
# the XCFun library
fun_xc_name = 'LDA'
fun_xc_weight = 1.0
fun_xc = xcfun.Functional({fun_xc_name: fun_xc_weight})

# xc.eval_potential_n receives density on a grid as an argument
result = fun_xc.eval_potential_n(n_2)
#print("Shape of the XCfun eval_potential_n output:", result.shape)

# the output contains energy density and XC potential values on a grid
energy    = result[:, 0]
potential = result[:, 1]

# to compute the XC energy contribution we need to 
# integrate the energy density on a grid with weights 
e_xc = energy @ weight_all_flat
```


```python
print(f"XC energy contribution with the {fun_xc_name} functional = {e_xc}")
```

    XC energy contribution with the LDA functional = -3.563430888505756


The XC energy contribution is significantly different from the one computed using `pyscf` code. Possibly due to the different set of MO coefficients.


```python
qmckl.context_destroy(ctx)
```
