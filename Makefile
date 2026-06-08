.PHONY: all configure build clean shaders

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

clean:
	rm -rf "$(BUILD_DIR)"

# Run the demo from the sibling app repo:
#   cd ../Vulkan-C-App && make debug
