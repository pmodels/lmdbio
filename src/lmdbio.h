/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *  (C) 2017 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#ifndef LMDBIO_H_INCLUDED
#define LMDBIO_H_INCLUDED

#include <iostream>
#include <string.h>
#include <stdlib.h>
#include "mpi.h"

class lmdbio
{
public:
    lmdbio(MPI_Comm parent_comm, const char *filename, int batch_size) {
        MPI_Comm_dup(parent_comm, &comm);
        fname = strdup(filename);
        bsize = batch_size;
    }

    ~lmdbio(void) {
        MPI_Comm_free(&comm);
        delete[] fname;
    }

    int read_record_batch(void ***records, int *num_records, int **record_sizes);

private:
    MPI_Comm comm;
    const char *fname;
    int bsize;
};

#endif  /* LMDBIO_H_INCLUDED */
