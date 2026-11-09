# Verification Matrix

| Requirement | Semantic oracle | Chisel test | Public fixture or audit |
| --- | --- | --- | --- |
| Five-observation impact and structural protection | `test_siu_warmup_confirmation_and_structural_protection` | `SiuSpec` | six-frame trace |
| `h/d` fusion, priority, aging, and cold-refill accounting | `test_fru_fuses_events_allocates_ages_and_averages_energy`, `test_fru_breaks_equal_priority_with_gid_and_uses_partition_local_rows` | `FruSpec` (including ascending-GID equal-priority tie), `MemorySpec` | six-frame trace |
| SplatSPM trajectory, FrameSPM full-tagged representative-pixel records, and event/allocation MetaSPM snapshots | not applicable, no timing abstraction | `FruSpec`, `MemorySpec` | direct full-record stimuli |
| Current-view pairing, complete StateSPM response, non-diagonal projected covariance, raw low-order color, `P/kappa/rho/omega`, and full/light/filter rule | `test_ciu_equations_and_training_filter_fallback`, `test_ciu_saturates_p_after_the_raw_low_order_color_product`, `test_ciu_nontrivial_jcov_sigma_projection_and_decision_fields` | `CiuSpec` (including mismatched-view and incomplete-state rejection, short-history non-candidate full-path rule, and candidate exception) | synthetic issue records |
| Degree-3 SH, FP16 light path, global compositing order across tiers, early stop, one-frame closed batches, path-domain rejection, and terminal-frame or non-finite-depth rejection | `test_tru_global_depth_order_early_stop_and_training_trace`, `test_tru_merges_three_tiers_by_tile_depth_and_gid`, `test_tru_rejects_a_closed_batch_with_multiple_frame_ids`, `test_tru_uses_full_degree_three_sh_and_fp16_light_rounding`, `test_tru_rejects_terminal_frame_training_work_before_trajectory_generation`, `test_tru_rejects_nonfinite_depth_before_global_compositing` | `TruSpec`, including signed-zero, later-empty-`frameEnd`, cross-frame-backpressure, undefined-path, terminal-frame, and non-finite-depth regressions | synthetic tile records |
| One-target trajectory to feedback update | `test_tuu_forms_a_surrogate_update_and_closes_the_feedback_transition`, `test_tuu_keeps_one_target_until_a_matching_seeded_trace_consumes_it` | `TuuSpec` | target and trace records |
| Integrated stage wiring, current user/training views, and refill | not applicable, oracle tests stages separately | `ScdGsLoopSpec`, `FixtureReplaySpec` | bootstrap feedback fixture and separate candidate RTL stimulus |
| Trace grammar, pinned 4096-GID, 12-bit tile, and 64-entry blend-order domains, and release boundary | none | `TraceReplaySpec` | `check_public_tree.py` |

The acceptance sequence is Python oracle tests, Chisel tests, trace replay,
SystemVerilog generation, optional structural lint, then public-tree audit.
