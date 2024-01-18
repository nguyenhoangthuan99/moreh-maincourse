#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>


int main(int argc, char **argv)
{
  int print, arg;
  if (argc != 2)
  {
    fprintf(stderr, "Usage: mykill <pid>\n");
    exit(-1);
  }

  if (argc >= 3)
  {
    print = 1;
  }

  arg = atoi(argv[1]);
  kill(arg, SIGUSR1);
  return 0;
}
