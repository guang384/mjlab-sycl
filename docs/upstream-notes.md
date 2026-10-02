# Suggested upstream fixes (task packages)

## bam.mjlab `_dof_friction_fo` — drop the zero-fill temporaries

The friction term allocates two `(num_envs, njmax)` fillers per act call
(2.6 MB each at 4096 envs) purely as `torch.where` defaults — measured
~1.25 GB of alloc/free churn per 2 PPO iterations, retained by the CPU
allocator and competing with the GPU for the LPDDR bus.

`torch.where` accepts a scalar fill; the change is value-identical:

```python
# before
contrib = torch.where(is_fric, efc_force, torch.zeros_like(efc_force))
idx = torch.where(is_fric, efc_id, torch.zeros_like(efc_id)).long()

# after (same semantics, no temporaries)
contrib = torch.where(is_fric, efc_force, 0.0)
idx = torch.where(is_fric, efc_id, 0).long()
```

Verified value-identical (scalar fill vs zeros_like fill); see
`docs/performance.md` "Memory map" for the measurement.
