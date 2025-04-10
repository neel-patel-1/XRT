#ifndef PRINT_UTILS_H
#define PRINT_UTILS_H

#include <stdio.h>
#include <stdbool.h>

extern int gLogLevel;
#define LOG_ERR -2
#define LOG_PERF -1
#define LOG_WARN 0
#define LOG_MONITOR 2
#define LOG_DEBUG 3
#define LOG_VERBOSE 4
#define LOG_TOO_VERBOSE 5

/* Printing */
/**< Prints the name of the function and the arguments only if gDebugParam is
 * CPA_TRUE.
 */

#ifdef PERF
#define PRINT_DBG
#define PRINT_ERR
#define LOG_PRINT
#endif

#ifndef LOG_PRINT
#define LOG_PRINT(log_level, args...)                                         \
    do                                                                         \
    {                                                                          \
        if(gLogLevel >= log_level)                                             \
        {                                                                      \
            printf("%s(): ", __func__);                                        \
            printf(args);                                                      \
            fflush(stdout);                                                    \
        }                                                                      \
    } while (0)
#endif

#ifndef PRINT_DBG
#define PRINT_DBG(args...)                                                     \
    do                                                                         \
    {                                                                          \
        if (gLogLevel >= LOG_DEBUG)                                           \
        {                                                                      \
            printf("%s(): ", __func__);                                        \
            printf(args);                                                      \
            fflush(stdout);                                                    \
        }                                                                      \
    } while (0)
#endif

/**< Prints the arguments */
#ifndef PRINT
#define PRINT(args...)                                                         \
    do                                                                         \
    {                                                                          \
        printf(args);                                                          \
    } while (0)
#endif

/**< Prints the name of the function and the arguments */
#ifndef PRINT_ERR
#define PRINT_ERR(args...)                                                     \
    do                                                                         \
    {                                                                          \
        printf("%s(): ", __func__);                                            \
        printf(args);                                                          \
    } while (0)
#endif


#endif