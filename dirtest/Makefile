CFLAGS = -O2 -Wall -Wextra -pthread

all: dirtest

dirtest: dirtest.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	-rm -f dirtest

.PHONY: all clean
