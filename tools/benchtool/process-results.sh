#!/bin/sh

# Turns `benchtool bench-hash --csv` output into per-algorithm markdown tables.
#
# Usage:
#   benchtool bench-hash --csv blake2b sha256 keccak256 > results.csv
#   ./process-results.sh results.csv > result-table.md
#
# The artifact whose name ends in `_simd` is used as the baseline for ratios;
# if none is present, the first artifact (in input order) is the baseline.
# Checksums are cross-checked: rows of the same (algo, size) must agree.

set -eu

awk -F, '
NR == 1 && $1 == "artifact" { next }                # header
NF < 6 { next }
{
    artifact = $1; algo = $2; size = $3; ns = $4; checksum = $6

    if (!(artifact in artifact_seen)) {
        artifact_seen[artifact] = ++artifact_count
        artifacts[artifact_count] = artifact
        if (artifact ~ /_simd$/) baseline = artifact
    }
    if (!((algo, size) in row_seen)) {
        row_seen[algo, size] = 1
        if (!(algo in algo_seen)) { algo_seen[algo] = ++algo_count; algos[algo_count] = algo }
        sizes_for[algo] = sizes_for[algo] " " size
    }

    time[artifact, algo, size] = ns

    if ((algo, size) in want_checksum) {
        if (want_checksum[algo, size] != checksum) {
            printf "WARNING: checksum mismatch for %s/%s: %s vs %s\n", \
                algo, size, want_checksum[algo, size], checksum > "/dev/stderr"
        }
    } else {
        want_checksum[algo, size] = checksum
    }
}

function fmt(ns) {
    if (ns < 1000) return sprintf("%.0f ns", ns)
    if (ns < 1000000) return sprintf("%.2f µs", ns / 1000)
    return sprintf("%.2f ms", ns / 1000000)
}

function fmt_size(bytes) {
    if (bytes < 1024) return bytes " B"
    if (bytes < 1048576) return (bytes / 1024) " KiB"
    return (bytes / 1048576) " MiB"
}

END {
    if (baseline == "") baseline = artifacts[1]

    # Reorder so the baseline column comes first.
    ordered_count = 0
    ordered[++ordered_count] = baseline
    for (i = 1; i <= artifact_count; i++) {
        if (artifacts[i] != baseline) ordered[++ordered_count] = artifacts[i]
    }
    for (i = 1; i <= artifact_count; i++) artifacts[i] = ordered[i]

    for (a = 1; a <= algo_count; a++) {
        algo = algos[a]
        printf "## %s\n\n", algo

        printf "| size |"
        for (i = 1; i <= artifact_count; i++) {
            printf " %s |", artifacts[i]
            if (artifacts[i] != baseline) printf " vs %s |", baseline
        }
        printf "\n|---:|"
        for (i = 1; i <= artifact_count; i++) {
            printf "---:|"
            if (artifacts[i] != baseline) printf "---:|"
        }
        printf "\n"

        n = split(sizes_for[algo], size_list, " ")
        for (s = 1; s <= n; s++) {
            size = size_list[s]
            printf "| %s |", fmt_size(size)
            base_ns = time[baseline, algo, size]
            for (i = 1; i <= artifact_count; i++) {
                ns = time[artifacts[i], algo, size]
                if (ns == "") { printf " - |"; if (artifacts[i] != baseline) printf " - |"; continue }
                printf " %s |", fmt(ns)
                if (artifacts[i] != baseline) {
                    if (base_ns > 0) printf " %.2fx |", ns / base_ns
                    else printf " - |"
                }
            }
            printf "\n"
        }
        printf "\n"
    }
}
' "$@"
