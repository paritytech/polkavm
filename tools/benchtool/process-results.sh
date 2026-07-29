#!/bin/sh

# Turns `benchtool bench-hash --csv` output into per-algorithm markdown tables.
#
# Usage:
#   benchtool bench-hash --csv blake2_256 sha2_256 ... > results.csv
#   ./process-results.sh results.csv > result-table.md
#
# Artifact roles:
#   bench-hash          -> pvm       (the PVM blob)
#   libbench_hash_simd  -> native    (built with -C target-cpu=native)
#   libbench_hash       -> portable  (plain native build)
# The PVM column gets ratios against both native and portable; a
# native/portable ratio shows what -C target-cpu=native buys (or costs) on
# the host. Any other artifact gets its own column with a ratio vs native.
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
        if (artifact ~ /_simd$/)            { native = artifact }
        else if (artifact ~ /^libbench/)    { portable = artifact }
        else if (pvm == "")                 { pvm = artifact }
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
    if (ns == "") return "-"
    if (ns < 1000) return sprintf("%.0f ns", ns)
    if (ns < 1000000) return sprintf("%.2f µs", ns / 1000)
    return sprintf("%.2f ms", ns / 1000000)
}

function ratio(a, b) {
    if (a == "" || b == "" || b == 0) return "-"
    return sprintf("%.2fx", a / b)
}

function fmt_size(bytes) {
    if (bytes < 1024) return bytes " B"
    if (bytes < 1048576) return (bytes / 1024) " KiB"
    return (bytes / 1048576) " MiB"
}

END {
    # Other artifacts = everything that is not one of the three roles.
    other_count = 0
    for (i = 1; i <= artifact_count; i++) {
        if (artifacts[i] != pvm && artifacts[i] != native && artifacts[i] != portable) {
            others[++other_count] = artifacts[i]
        }
    }

    for (a = 1; a <= algo_count; a++) {
        algo = algos[a]
        printf "## %s\n\n", algo

        printf "| size |"
        if (pvm != "")      printf " pvm |"
        if (native != "")   printf " native |"
        if (portable != "") printf " portable |"
        if (pvm != "" && native != "")      printf " pvm/native |"
        if (pvm != "" && portable != "")    printf " pvm/portable |"
        if (native != "" && portable != "") printf " native/portable |"
        for (i = 1; i <= other_count; i++) printf " %s | %s/native |", others[i], others[i]
        printf "\n|---:|"
        cols = (pvm != "") + (native != "") + (portable != "") \
             + (pvm != "" && native != "") + (pvm != "" && portable != "") \
             + (native != "" && portable != "") + 2 * other_count
        for (i = 0; i < cols; i++) printf "---:|"
        printf "\n"

        n = split(sizes_for[algo], size_list, " ")
        for (s = 1; s <= n; s++) {
            size = size_list[s]
            t_pvm = time[pvm, algo, size]
            t_nat = time[native, algo, size]
            t_por = time[portable, algo, size]
            printf "| %s |", fmt_size(size)
            if (pvm != "")      printf " %s |", fmt(t_pvm)
            if (native != "")   printf " %s |", fmt(t_nat)
            if (portable != "") printf " %s |", fmt(t_por)
            if (pvm != "" && native != "")      printf " %s |", ratio(t_pvm, t_nat)
            if (pvm != "" && portable != "")    printf " %s |", ratio(t_pvm, t_por)
            if (native != "" && portable != "") printf " %s |", ratio(t_nat, t_por)
            for (i = 1; i <= other_count; i++) {
                t_o = time[others[i], algo, size]
                printf " %s | %s |", fmt(t_o), ratio(t_o, t_nat)
            }
            printf "\n"
        }
        printf "\n"
    }
}
' "$@"
