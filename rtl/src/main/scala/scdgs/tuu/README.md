# TUU

`TrainingUpdateUnit.scala` holds one target, consumes a matching forward
trajectory, forms a scalar surrogate update, writes the Gaussian and optimizer
state, and only then returns the next canonical update to the top-level
feedback loop. This transaction order prevents feedback from observing the
pre-update state when the StateSPM write is backpressured.

Only training trajectories tagged `full` or `light` are accepted. A `filter`
tag or a user-view record is not a legal TRU training trajectory; TUU rejects
it, increments its invalid-pair counter, and retains the buffered target.
