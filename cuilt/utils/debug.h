/*
 * =======================================================================
 * 
 *       Filename:  debug.h
 * 
 *    Description:  Header file for debug print function
 * 
 *        Version:  1.0
 *        Created:  12/16/2009 07:57:24 PM CST
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Ashutosh Chakraborty (), ashutosh@mail.utexas.edu
 *        Company:  University of Texas, Austin
 * 
 * =======================================================================
 */

// Copied and enhanced this file after downloading from
// http://oopweb.com/CPP/Documents/DebugCPP/Volume/techniques.html
//   - Ashutosh 17 Dec 2009

#ifndef DEBUG_H
#define DEBUG_H

#include <stdarg.h>

#if defined(NDEBUG) && defined(__GNUC__)
/* gcc's cpp has extensions; it allows for macros with a variable number of
   arguments. We use this extension here to preprocess dmesg away. */
#define dmesg(format, args...) ((void)0)
#define emesg(format, args...) ((void)0)
#define imesg(format, args...) ((void)0)
#define wmesg(format, args...) ((void)0)
#else
void dmesg(const char *format, ...);
void emesg(const char *format, ...);
void imesg(const char *format, ...);
void wmesg(const char *format, ...);
/* print a message, if it is considered significant enough.
      Adapted from [K&R2], p. 174 */
#endif

#endif /* DEBUG_H */
