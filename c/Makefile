# libft8nn: needs ONNX Runtime and nothing else. Default: the official release unpacked by
# tools/get_onnxruntime.sh into third_party/onnxruntime (Homebrew's build runs this model 2.3x slower);
# falls back to `brew --prefix onnxruntime`, then /usr/local. Override with make ORT_PREFIX=/path.
# make NATIVE=1 adds -march=native (faster on the build machine, but the binary then needs a CPU like it).
UNAME := $(shell uname -s)
ORT_PREFIX ?= $(or $(wildcard $(abspath ../third_party/onnxruntime)),$(shell brew --prefix onnxruntime 2>/dev/null),/usr/local)
CC ?= cc
CFLAGS ?= -O3 -fcx-limited-range -Wall -Wextra -Wno-unused-parameter -std=c11
ifeq ($(NATIVE),1)
CFLAGS += -march=native
endif
# official releases put the headers in include/, Homebrew in include/onnxruntime/
CPPFLAGS += -I$(ORT_PREFIX)/include -I$(ORT_PREFIX)/include/onnxruntime -D_DEFAULT_SOURCE

ifeq ($(OS),Windows_NT)
# Windows: MinGW-w64 gcc in an MSYS2 UCRT64 shell (MSVC can't build this code: it uses C99 `double complex`).
# Windows has no rpath, so onnxruntime.dll is copied next to the programs; libgcc is linked statically and the
# _WIN32 code uses no pthreads, so the .exe and .dll need only Windows' own DLLs and onnxruntime.dll.
# MinGW's <math.h> hides M_PI under -std=c11 unless asked (glibc shows it for _DEFAULT_SOURCE)
CPPFLAGS += -D_USE_MATH_DEFINES
LDLIBS += -L$(ORT_PREFIX)/lib -lonnxruntime -lm -static-libgcc
SHLIB = build/ft8nn.dll
SHFLAGS = -shared -Wl,--out-implib,build/libft8nn.dll.a
EXE = .exe
RUNTIME = build/onnxruntime.dll
else
LDLIBS += -L$(ORT_PREFIX)/lib -lonnxruntime -lm -lpthread -Wl,-rpath,$(ORT_PREFIX)/lib
ifeq ($(UNAME),Darwin)
SHLIB = build/libft8nn.dylib
SHFLAGS = -dynamiclib
else
SHLIB = build/libft8nn.so
SHFLAGS = -shared
endif
endif

SRC = src/fft.c src/ft8.c src/ldpc.c src/frontend.c src/subtract.c src/unpack.c src/decode.c
OBJ = $(SRC:src/%.c=build/%.o)

all: build/libft8nn.a $(SHLIB) build/ft8nn_cli$(EXE) $(RUNTIME)

build/%.o: src/%.c src/internal.h include/ft8nn.h src/ldpc_tables.h | build
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC -c $< -o $@

build/libft8nn.a: $(OBJ)
	ar rcs $@ $^

$(SHLIB): $(OBJ)
	$(CC) $(SHFLAGS) -o $@ $^ $(LDLIBS)

build/ft8nn_cli$(EXE): tools/ft8nn_cli.c build/libft8nn.a
	$(CC) $(CPPFLAGS) $(CFLAGS) $< build/libft8nn.a -o $@ $(LDLIBS)

build/onnxruntime.dll: $(ORT_PREFIX)/lib/onnxruntime.dll | build
	cp $< $@

build:
	mkdir -p build

clean:
	rm -rf build

.PHONY: all clean
