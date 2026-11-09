# ft8nn_sidecar: this project's FT8 decoder as WSJT-X's external FT8 decoder (see README.md).
#
#   make                 build build/ft8nn_sidecar and build/fake_wsjtx
#   make test [NSLOTS=4] decode a few real slots through the sidecar protocol
#
# Needs libft8nn (make -C ../c) and the ONNX Runtime it uses; nothing else. The protocol is JSON
# lines and raw audio on a pipe, so there is no Qt and no shared memory here.
ORT_PREFIX ?= $(or $(wildcard $(abspath ../third_party/onnxruntime)),$(shell brew --prefix onnxruntime 2>/dev/null),/usr/local)
FT8NN_LIB = ../c/build/libft8nn.a

CXX ?= c++
CXXFLAGS ?= -O2 -Wall -Wextra -Wno-unused-parameter -std=c++17
# timegm() and gmtime_r() are POSIX/BSD, and glibc hides them when -std=c++17 asks for strict ANSI.
DEFS = -D_DEFAULT_SOURCE
CPPFLAGS += -I$(ORT_PREFIX)/include -I$(ORT_PREFIX)/include/onnxruntime
LDLIBS += -L$(ORT_PREFIX)/lib -lonnxruntime -lm -lpthread -Wl,-rpath,$(ORT_PREFIX)/lib

all: build/ft8nn_sidecar build/fake_wsjtx

build:
	mkdir -p build

$(FT8NN_LIB):
	$(MAKE) -C ../c build/libft8nn.a

build/ft8nn_sidecar: ft8nn_sidecar.cpp json.h linereader.h wav.h $(FT8NN_LIB) | build
	$(CXX) $(CPPFLAGS) $(DEFS) $(CXXFLAGS) $< $(FT8NN_LIB) -o $@ $(LDLIBS)

build/fake_wsjtx: fake_wsjtx.cpp json.h linereader.h wav.h | build
	$(CXX) $(DEFS) $(CXXFLAGS) $< -o $@

NSLOTS ?= 4
test: all
	./test_sidecar.sh $(NSLOTS)

clean:
	rm -rf build

.PHONY: all clean test
