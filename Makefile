.PHONY: all configure build run clean debug release

BUILD_DIR ?= build
BUILD_TYPE ?= Release

all: build

configure:
	cmake -S . -B "$(BUILD_DIR)" \
		-DCMAKE_BUILD_TYPE="$(BUILD_TYPE)" \
		$(if $(VULKAN_SDK),-DVULKAN_SDK_ROOT="$(VULKAN_SDK)",)

build: configure
	cmake --build "$(BUILD_DIR)" -j$$(nproc)

debug:
	$(MAKE) BUILD_TYPE=Debug build

release:
	$(MAKE) BUILD_TYPE=Release build

run: configure
	cmake --build "$(BUILD_DIR)" --target run -j$$(nproc)

clean:
	rm -rf "$(BUILD_DIR)"
