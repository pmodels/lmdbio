AC_PREREQ(2.63)
#
# (C) 2017 by Argonne National Laboratory.
#     See COPYRIGHT in top-level directory.
#

AC_INIT([LMDBIO], [0.1])

AC_CONFIG_SRCDIR([src/lmdbio.h])

AM_INIT_AUTOMAKE([subdir-objects] [-Wall -Werror foreign 1.12.3])

AC_PROG_CXX([mpicxx])
AM_PROG_AR

LT_PREREQ([2.2.6])

LT_INIT()

# Non-verbose make
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])

AC_C_CONST
AC_C_RESTRICT
AC_C_INLINE

dnl Final output
AC_CONFIG_FILES([Makefile])
AC_OUTPUT
