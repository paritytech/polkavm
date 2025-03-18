SERVICES = bootstrap tribonacci megatron transfer balances delay null_authorizer auth_copy blake2b fib corevm corevm_child

TARGET_DIR = riscv64emac-unknown-none-polkavm/release

.PHONY: all $(SERVICES) clean

all: $(SERVICES)

$(SERVICES):
	@echo "Building $@ service..."
	cd ./services/$@ && cargo build --release --target-dir ./target
	cd ../../
	@if [ "$@" = "null_authorizer" ]; then \
		cargo run -p polkatool jam-service services/$@/target/$(TARGET_DIR)/$@ -o services/$@/$@.pvm -d services/$@/$@_blob.pvm -i; \
	elif [ "$@" = "corevm_child" ]; then \
		cargo run -p polkatool jam-service services/$@/target/$(TARGET_DIR)/$@ -o services/$@/$@.pvm -d services/$@/$@_blob.pvm -m; \
	else \
		cargo run -p polkatool jam-service services/$@/target/$(TARGET_DIR)/$@ -o services/$@/$@.pvm -d services/$@/$@_blob.pvm; \
	fi
	cargo run -p polkatool disassemble services/$@/$@_blob.pvm --show-raw-bytes > ./services/$@/$@.txt
	@echo "Service $@ built successfully!"

clean:
	@echo "Cleaning up all build artifacts..."
	rm -rf services/*/target
	rm -rf services/*/*.pvm
	rm -rf services/*/*.txt
	@echo "Cleanup complete!"
