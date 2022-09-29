/* error.c */

#include "lithosim/pvbandsim/parse_types.h"
#include "lithosim/pvbandsim/parse_print.h"
#include "lithosim/pvbandsim/parse_error.h"

/* global variables */

print *errorGlobalPrint = NULL;

/* functions */

void errorInitPrint(void)
{
    if (!errorGlobalPrint)
        errorGlobalPrint = printCreateFile(stderr);
}

void errorSetPrint(print *p, mybool dispose)
{
    if (dispose && errorGlobalPrint)
        printDispose(errorGlobalPrint);
    errorGlobalPrint = p;
}

print *errorGetPrint(void)
{
    return errorGlobalPrint;
}

/* end error.c */
