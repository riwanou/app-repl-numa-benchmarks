CC = clang --target=x86_64-linux-gnu --gcc-toolchain=/usr
CFLAGS = -g -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wstrict-prototypes 
CFLAGS += -MMD -MP
LDFLAGS = -lnuma -pthread

srcs :=  bench_mem.c bench_pgtable.c bench_alloc.c
bins := $(patsubst %.c,%,$(srcs))

all: $(bins)

%: %.c
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

clean:
	-rm $(bins) *.d

-include *.d
