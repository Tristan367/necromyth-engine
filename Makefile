.PHONY: all configure build run clean debug release

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

debug:
	$(MAKE) BUILD_TYPE=Debug build
	$(MAKE) BUILD_TYPE=Debug run

release:
	$(MAKE) BUILD_TYPE=Release build

run: configure
	cmake --build "$(BUILD_DIR)" --target run -j$$(nproc)

clean:
	rm -rf "$(BUILD_DIR)"
