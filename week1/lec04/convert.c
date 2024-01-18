#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void fallback_print_usage()
{
  printf("Usage: ./convert [int|long|float|double] number\n");
  printf("Example: ./convert float 3.14\n");
  exit(0);
}

void print_int(int x)
{
  // one character per one bit
  char output[32 + 1] = {
      0,
  };

  /* YOUR CODE START HERE */
  int y = abs(x);

  for (int i = 31; i >= 0; i--)
  {
    if (y % 2)
    {
      output[i] = '1';
    }
    else
    {
      output[i] = '0';
    }

    y = y / 2;
  }
  if (x < 0)
  {
    // reverse bit
    for (int i = 0; i <= 31; i++)
    {
      if (output[i] == '0')
      {
        output[i] = '1';
      }
      else
      {
        output[i] = '0';
      }
    }
    // add 1
    int i = 31;
    while (1)
    {
      if (output[i] == '0')
      {
        output[i] = '1';
        break;
      }
      else
      {
        output[i] = '0';
        i--;
      }
    }
  }

  // another option
  if (x > 0)
  {
    output[0] = '0';
  }
  else
  {
    output[0] = '1';
  }
  for (int i = 1; i <= 31; i++)
  {
    output[i] = (x & (int)1 << (31 - i)) ? '1' : '0';
  }

  //another option

  


  /* YOUR CODE END HERE */

  printf("%s\n", output);
}

void print_long(long x)
{
  // one character per one bit
  char output[64 + 1] = {
      0,
  };

  /* YOUR CODE START HERE */
  

  //another option
  unsigned long z = x;
  for (int i = 0; i <= 63; i++)
    {
      output[i] = (z & (long)1 << (64 - i - 1)) ? '1' : '0';
    }

  /* YOUR CODE END HERE */

  printf("%s\n", output);
}

void print_float(float x)
{
  // one character per one bit
  char output[32 + 1] = {
      0,
  };

  /* YOUR CODE START HERE */
  int length = 8*sizeof(float);
  union unions
  {
    float f;
    unsigned int ui;
  };
  union unions u;
  u.f = x;
  unsigned int y = u.ui;
  for (int i = 0; i < length; i++)
  {
    output[length - 1 - i] = ((y >> i) & 1) ? '1' : '0';
  }

  /* YOUR CODE END HERE */

  printf("%s\n", output);
}

void print_double(double x)
{
  // one character per one bit
  char output[64 + 1] = {
      0,
  };

  /* YOUR CODE START HERE */
  int length = 8*sizeof(double);
  union unions
  {
    double f;
    unsigned long ui;
  };
  union unions u;
  u.f = x;
  unsigned long y = u.ui;
  for (int i = 0; i < length; i++)
  {
    output[length - 1 - i] = ((y >> i) & 1) ? '1' : '0';
  }

  /* YOUR CODE END HERE */

  printf("%s\n", output);
}

int main(int argc, char **argv)
{
  if (argc != 3)
    fallback_print_usage();
  if (strcmp(argv[1], "int") == 0)
  {
    print_int(atoi(argv[2]));
  }
  else if (strcmp(argv[1], "long") == 0)
  {
    print_long(atol(argv[2]));
  }
  else if (strcmp(argv[1], "float") == 0)
  {
    print_float(atof(argv[2]));
  }
  else if (strcmp(argv[1], "double") == 0)
  {
    print_double(atof(argv[2]));
  }
  else
  {
    fallback_print_usage();
  }
  return 0;
}
