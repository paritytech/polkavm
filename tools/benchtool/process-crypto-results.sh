#!/bin/sh

# Turns the generic-harness output of run-crypto-benches into a markdown
# summary table: per benchmark, host vs PVM (64-bit, compiler, sync gas).
#
# The "host" column is whichever build the benchmark libraries were compiled
# with (host portable by default; host native if built with
# -C target-cpu=native) - label the resulting table accordingly.
#
# Usage:
#   ./run-crypto-benches > crypto.txt
#   ./process-crypto-results.sh crypto.txt > crypto-table.md

set -eu

awk '
# lines look like: runtime/<bench>/<backend>: ...runtime/<bench>/<backend>: 355.61us
$1 ~ /^runtime\// {
    split($1, path, "/")
    bench = path[2]
    backend = path[3]; sub(/:$/, "", backend)
    time = $NF

    # normalize to nanoseconds
    ns = time
    if (time ~ /ns$/)      { sub(/ns$/, "", ns); ns += 0 }
    else if (time ~ /us$/) { sub(/us$/, "", ns); ns *= 1000 }
    else if (time ~ /ms$/) { sub(/ms$/, "", ns); ns *= 1000000 }
    else if (time ~ /s$/)  { sub(/s$/,  "", ns); ns *= 1000000000 }
    else next

    if (!(bench in seen)) { seen[bench] = ++count; benches[count] = bench }
    if (backend == "native")                       native[bench] = ns
    if (backend == "polkavm64_compiler_sync_gas")  pvm[bench] = ns
}

function fmt(ns) {
    if (ns == "") return "-"
    if (ns < 1000) return sprintf("%.0f ns", ns)
    if (ns < 1000000) return sprintf("%.2f µs", ns / 1000)
    return sprintf("%.2f ms", ns / 1000000)
}

END {
    print "| benchmark | host | PVM (64-bit, sync gas) | ratio |"
    print "|---|---:|---:|---:|"
    for (i = 1; i <= count; i++) {
        b = benches[i]
        ratio = (native[b] != "" && pvm[b] != "") ? sprintf("%.2fx", pvm[b] / native[b]) : "-"
        printf "| %s | %s | %s | %s |\n", b, fmt(native[b]), fmt(pvm[b]), ratio
    }
}
' "$@"
