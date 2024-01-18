#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <ctype.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <errno.h>

const int MAX = 13;

static void doFib(int n, int doPrint);

/*
 * unix_error - unix-style error routine.
 */
inline static void
unix_error(char *msg)
{
  fprintf(stdout, "%s: %s\n", msg, strerror(errno));
  exit(1);
}

int main(int argc, char **argv)
{
  int arg;
  int print;

  if (argc != 2)
  {
    fprintf(stderr, "Usage: fib <num>\n");
    exit(-1);
  }

  if (argc >= 3)
  {
    print = 1;
  }

  arg = atoi(argv[1]);
  if (arg < 0 || arg > MAX)
  {
    fprintf(stderr, "number must be between 0 and %d\n", MAX);
    exit(-1);
  }

  doFib(arg, 1);

  return 0;
}

/*
 * Recursively compute the specified number. If print is
 * true, print it. Otherwise, provide it to my parent process.
 *
 * NOTE: The solution must be recursive and it must fork
 * a new child for each call. Each process should call
 * doFib() exactly once.
 */
static void
doFib(int n, int doPrint)
{
  // if (doPrint)
  // {
  //   printf("%d\n", n);
  // }
  if (n == 2 || n == 1)
  {
    if (doPrint)
    {
      printf("%d th fib is %d\n", n, 1);
    }
    exit(1);
  }
  int status1, status2, done_1=0, done_2=0;
  pid_t p1 = -1, p2 = -1;
  while (p1 < 0 || p2 < 0)
  {
    if (p1 < 0)
      p1 = fork();

    if (p2 < 0)
      p2 = fork();
    // child process because return value zero
    if (p1 == 0 && done_1==0)
    {
      doFib(n - 1, 1);
      done_1=1;
    }
    else if (p2 == 0 && done_2==0)
    {
      /* code */
      doFib(n - 2, 1);
      done_2=1;
    }
    // parent process because return value non-zero.
    else
    {
      if (p1 < 0 || p2 < 0)
      {
        sleep(0.01);
        continue;
      }
      waitpid(p1, &status1, 0);
      waitpid(p2, &status2, 0);

      if (WIFEXITED(status1) && WIFEXITED(status2))
      {
        /* Child process exited normally, through `return` or `exit` */
        if (doPrint)
        {
          printf("%d th fib is %d\n", n, WEXITSTATUS(status1) + WEXITSTATUS(status2));
        }
        exit(WEXITSTATUS(status1) + WEXITSTATUS(status2));
      }
    }
  }
}
