# CPU training environment for AffineAI.
# Usage: source ./env_cpu.sh   (must be sourced, not executed)
# Add future machine-tuning variables here so launch stays one line.

# Threads spin instead of sleeping between the many small OpenMP parallel
# regions (~15% faster training). Must be set before process start.
export OMP_WAIT_POLICY=active

# tcmalloc thread cache beats glibc malloc for the many small per-step
# allocations (~30% faster end-to-end). Applies to processes launched
# AFTER sourcing (loader reads it at exec, so sourcing mid-shell is fine
# as long as python starts after). Fallback: jemalloc (~unchanged).
# Adjust the path per machine (ldconfig -p | grep tcmalloc).
export LD_PRELOAD=/usr/lib/libtcmalloc_minimal.so.4
