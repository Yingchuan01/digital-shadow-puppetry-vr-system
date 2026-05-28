# Processed objective validation results

This folder contains processed objective validation results for the digital shadow-puppetry interaction system.

The objective validation compares three system conditions:

1. Direct Transform / No Physics
2. Planar CCD-IK without Physics
3. Ours with Physics

The task set includes:

1. Free Motion
2. Door-Knocking
3. Two-Character Contact

## Recovery metric note

The variable `recovery_success_rate` was not used in the manuscript because recovery was treated as a time-based post-contact metric. The manuscript reports recovery behavior using `recovery_time` rather than a binary success-rate metric.

Accordingly, recovery-related results should be interpreted through post-contact recovery time and the visible layer's return toward the Solver Skeleton target after contact release.
