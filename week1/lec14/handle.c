#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>
#include "util.h"

/*
 * First, print out the process ID of this process.
 *
 * Then, set up the signal handler so that ^C causes
 * the program to print "Nice try.\n" and continue looping.
 *
 * Finally, loop forever, printing "Still here\n" once every
 * second.
 */

void handle_sigint(int sig)
{
  size_t bytes;
  const int STDOUT = 1;
  bytes = write(STDOUT, "Nice try.\n", 10);
  if (bytes != 10)
    exit(-999);
}
void handle_sigusr1(int sig)
{
  size_t bytes;
  const int STDOUT = 1;
  bytes = write(STDOUT, "terminating\n", 10);
    exit(0);
}

int main(int argc, char **argv)
{
  pid_t p = getpid();
  printf("%d\n", p);
  signal(SIGINT, handle_sigint);
  signal(SIGUSR1, handle_sigusr1);
  while (1)
  {
    sleep(1);
    printf("Still here\n");
  }

  return 0;
}
