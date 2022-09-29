#include "utils/debug.h"
#include <stdio.h>
#include <stdlib.h>

#if defined(NDEBUG) && defined(__GNUC__)
/* Nothing. dmesg has been "defined away" in debug.h already. */
#else
void dmesg(const char *format, ...)
{
#ifdef NDEBUG
/* Empty body, so a good compiler will optimise calls
   to dmesg away */
#else
    va_list args;

    fprintf(stderr, "[DEBUG]:");
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
#endif /* NDEBUG */
}
#endif /* NDEBUG && __GNUC__ */

#if defined(NDEBUG) && defined(__GNUC__)
/* Nothing. emesg has been "defined away" in debug.h already. */
#else
void emesg(const char *format, ...)
{
#ifdef NDEBUG
    /* Empty body, so a good compiler will optimise calls
       to emesg away */
#else
    va_list args;

    fprintf(stderr, "[ERROR]:  ");
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
    fprintf(stderr, "ABRT:  Will abort\n");
    abort();
#endif /* NDEBUG */
}
#endif /* NDEBUG && __GNUC__ */

#if defined(NDEBUG) && defined(__GNUC__)
/* Nothing. imesg has been "defined away" in debug.h already. */
#else
void imesg(const char *format, ...)
{
#ifdef NDEBUG
    /* Empty body, so a good compiler will optimise calls
       to imesg away */
#else
    va_list args;

    fprintf(stdout, "[INFO]:");
    va_start(args, format);
    vfprintf(stdout, format, args);
    va_end(args);
#endif /* NDEBUG */
}
#endif /* NDEBUG && __GNUC__ */

#if defined(NDEBUG) && defined(__GNUC__)
/* Nothing. wmesg has been "defined away" in debug.h already. */
#else
void wmesg(const char *format, ...)
{
#ifdef NDEBUG
    /* Empty body, so a good compiler will optimise calls
       to wmesg away */
#else
    va_list args;

    fprintf(stderr, "[WARNING]:");
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
#endif /* NDEBUG */
}
#endif /* NDEBUG && __GNUC__ */
