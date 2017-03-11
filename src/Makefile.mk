# -*- Mode: Makefile; -*-
#
# (C) 2017 by Argonne National Laboratory.
#     See COPYRIGHT in top-level directory.
#

AM_CPPFLAGS += -I$(top_srcdir)/src -I$(top_builddir)/src

liblmdbio_la_SOURCES += \
	src/lmdbio.cpp

include_HEADERS += src/lmdbio.h
