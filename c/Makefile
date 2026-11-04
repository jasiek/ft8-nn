# libft8nn: needs ONNX Runtime (brew install onnxruntime). No other dependencies.
ORT_PREFIX ?= $(shell brew --prefix onnxruntime 2>/dev/null || echo /usr/local)
CC ?= cc
CFLAGS ?= -O3 -march=native -fcx-limited-range -Wall -Wextra -Wno-unused-parameter -std=c11
CPPFLAGS += -I$(ORT_PREFIX)/include -D_DEFAULT_SOURCE
LDLIBS += -L$(ORT_PREFIX)/lib -lonnxruntime -lm -lpthread -Wl,-rpath,$(ORT_PREFIX)/lib

SRC = src/fft.c src/ft8.c src/ldpc.c src/frontend.c src/subtract.c src/unpack.c src/decode.c
OBJ = $(SRC:src/%.c=build/%.o)

all: build/libft8nn.a build/libft8nn.dylib build/ft8nn_cli

build/%.o: src/%.c src/internal.h include/ft8nn.h src/ldpc_tables.h | build
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC -c $< -o $@

build/libft8nn.a: $(OBJ)
	ar rcs $@ $^

build/libft8nn.dylib: $(OBJ)
	$(CC) -dynamiclib -o $@ $^ $(LDLIBS)

build/ft8nn_cli: tools/ft8nn_cli.c build/libft8nn.a
	$(CC) $(CPPFLAGS) $(CFLAGS) $< build/libft8nn.a -o $@ $(LDLIBS)

build:
	mkdir -p build

clean:
	rm -rf build

.PHONY: all clean
