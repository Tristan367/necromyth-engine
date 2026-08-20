.PHONY: all configure build clean shaders test

BUILD_DIR ?= build
BUILD_TYPE ?= Release
VULKAN_SDK_ROOT ?= $(HOME)/opt/vulkan-sdk/default/x86_64

all: build

configure:
	cmake -S . -B "$(BUILD_DIR)" \
		-DCMAKE_BUILD_TYPE="$(BUILD_TYPE)" \
		-DVULKAN_SDK_ROOT="$(VULKAN_SDK_ROOT)"

build: configure
	cmake --build "$(BUILD_DIR)" -j$$(nproc)

shaders: configure
	cmake --build "$(BUILD_DIR)" --target vce_shaders -j$$(nproc)

# CPU-side invariants; needs no GPU. `build` already compiles every engine
# header, so this is the second half of "did I break the engine?".
test: build
	ctest --test-dir "$(BUILD_DIR)" --output-on-failure

clean:
	rm -rf "$(BUILD_DIR)"

# Run the demo from the sibling app repo:
#   cd ../necromyth-engine-demo && make debug
