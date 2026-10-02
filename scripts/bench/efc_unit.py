import os
from mjlab_sycl._bootstrap import prepare_sycl_runtime_path
prepare_sycl_runtime_path()
import numpy as np
import warp as wp
from mujoco_warp._src import types as _mw_t
wp.init()

NW, NV, NJMAX, NACON, ADRS = 8, 20, 176, 16, 4
rng = np.random.default_rng(77)

NE = np.zeros(NW, dtype=np.int32)
NF = np.zeros(NW, dtype=np.int32)
NEFC = rng.integers(1, NJMAX, NW).astype(np.int32)
# spread rows across equality/friction/limit+contact families
for w in range(NW):
    ne = int(rng.integers(0, 4)); nf = int(rng.integers(0, 4))
    NE[w], NF[w] = ne, nf
DONE = rng.random(NW) < 0.25
TYPE = rng.integers(0, 8, (NW, NJMAX)).astype(np.int32)  # incl. ELLIPTIC(7)
ID = rng.integers(0, NACON, (NW, NJMAX)).astype(np.int32)
D = rng.random((NW, NJMAX)).astype(np.float32) + 0.5
FRICLOSS = rng.random((NW, NJMAX)).astype(np.float32) * 2
JAREF = rng.standard_normal((NW, NJMAX)).astype(np.float32)
IMPR = rng.random(1).astype(np.float32)
FRIC = rng.random((NACON, 5)).astype(np.float32)
CDIM = rng.integers(1, 4, NACON).astype(np.int32)
ADR = rng.integers(0, NJMAX, (NACON, ADRS)).astype(np.int32)
NACONV = np.array([NACON], dtype=np.int32)

def a(x, dt=float):
    return wp.array(x, dtype=dt, device="sycl")

wne, wnf, wnefc = a(NE, int), a(NF, int), a(NEFC, int)
wdone = a(DONE, bool)
wtype, wid, wadr, wcdim = a(TYPE, int), a(ID, int), a(ADR, int), a(CDIM, int)
wD, wfl, wj = a(D), a(FRICLOSS), a(JAREF)
wimp, wfric, wnacon = a(IMPR), wp.array(FRIC, dtype=_mw_t.vec5, device="sycl"), a(NACONV, int)

# ---- warp reference (track_changes=True) ----
from mujoco_warp._src.solver import update_constraint_efc
K = update_constraint_efc(True)
f1 = wp.zeros((NW, NJMAX), dtype=float, device="sycl")
s1 = wp.zeros((NW, NJMAX), dtype=int, device="sycl")
c1 = wp.zeros(NW, dtype=float, device="sycl")
ci1 = wp.zeros((NW, NJMAX), dtype=int, device="sycl")
cc1 = wp.zeros(NW, dtype=int, device="sycl")
wp.launch(K, dim=(NW, NJMAX),
          inputs=[wimp, wne, wnf, wnefc, wfric, wcdim, wadr, wtype, wid, wD, wfl, wnacon, wj, wdone],
          outputs=[f1, s1, c1, ci1, cc1], device="sycl")

# ---- native + fold ----
from mjlab_sycl import native_kernels
f2 = wp.zeros((NW, NJMAX), dtype=float, device="sycl")
s2 = wp.zeros((NW, NJMAX), dtype=int, device="sycl")
p2 = wp.zeros((NW, NJMAX), dtype=float, device="sycl")
ci2 = wp.zeros((NW, NJMAX), dtype=int, device="sycl")
cc2 = wp.zeros(NW, dtype=int, device="sycl")
c2 = wp.zeros(NW, dtype=float, device="sycl")
ok = native_kernels.efc_force(wimp, wne, wnf, wnefc, wfric, wcdim, wadr, wtype, wid,
                              wD, wfl, wnacon, wj, wdone, f2, s2, p2, ci2, cc2,
                              NJMAX, NJMAX, ADRS, True)
ok &= native_kernels.cost_fold(p2, wnefc, wdone, c2, NJMAX)
wp.synchronize_device("sycl")
print("native ran:", bool(ok))

F1, S1, C1, CC1 = f1.numpy(), s1.numpy(), c1.numpy(), cc1.numpy()
F2, S2, C2, CC2 = f2.numpy(), s2.numpy(), c2.numpy(), cc2.numpy()
act = ~DONE
print("force bit-equal (active):", np.array_equal(F1[act], F2[act]))
bad = np.argwhere((F1 != F2) & act[:, None])
print("n force mismatches:", len(bad))
w, e = bad[0] if len(bad) else (0, 0)
conid = int(ID[w, e]); efcid0 = int(ADR[conid, 0])
mu = float(FRIC[conid, 0]) * float(IMPR[0])
N = float(JAREF[w, efcid0]) * mu
dim = int(CDIM[conid])
TT = 0.0
for j in range(1, dim):
    j2 = int(ADR[conid, j])
    uj = float(JAREF[w, j2]) * float(FRIC[conid, j - 1])
    TT += uj * uj
T = TT ** 0.5 if TT > 0 else 0.0
print(f"case w={w} e={e} conid={conid} efcid0={efcid0} e==efcid0={e==efcid0}")
print(f"mu={mu:.6f} N={N:.6f} T={T:.6f} mu*T={mu*T:.6f}")
print(f"top: {N >= mu*T or (T <= 0 and N >= 0)}  bottom: {mu*N + T <= 0 or (T <= 0 and N < 0)}")
print(f"Jaref[w,e]={JAREF[w,e]:.6f} D={D[w,e]:.6f} -> bottom force {-D[w,e]*JAREF[w,e]:.6f}")
print(f"observed: f1={F1[w,e]:.6f} s1={S1[w,e]}   f2={F2[w,e]:.6f} s2={S2[w,e]}")
print("state bit-equal (active):", np.array_equal(S1[act], S2[act]))
print("changed_count equal:", np.array_equal(CC1[act], CC2[act]))
rel = np.abs(C1[act] - C2[act]) / np.maximum(np.abs(C1[act]), 1e-6)
print(f"cost max rel diff (atomic-order vs fold): {rel.max():.2e}")
# changed ids as multisets per world
ci1n, ci2n = ci1.numpy(), ci2.numpy()
same = all(sorted(ci1n[w][:CC1[w]]) == sorted(ci2n[w][:CC2[w]]) for w in range(NW) if act[w])
print("changed_ids multisets equal:", same)
