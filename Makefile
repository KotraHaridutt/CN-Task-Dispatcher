CC      = gcc
CFLAGS  = -Wall -Wextra -pthread
LDFLAGS = -lm

.PHONY: all clean

all: master worker

master: master.c common.h
	$(CC) $(CFLAGS) -o master master.c $(LDFLAGS)

worker: worker.c common.h
	$(CC) $(CFLAGS) -o worker worker.c $(LDFLAGS)

clean:
	rm -f master worker