# Redefine pushd command.
pushd() {
    command pushd "$@" > /dev/null
}

# Redefine popd command.
popd() {
    command popd "$@" > /dev/null
}
