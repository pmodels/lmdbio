#include <iostream>
#include "lmdbio.h"

#define BATCH_SIZE (128)

int main()
{
    void **records;
    int num_records, *record_sizes;

    MPI_Init(NULL, NULL);

    lmdbio obj(MPI_COMM_WORLD, "/dev/null", BATCH_SIZE);

    obj.read_record_batch(&records, &num_records, &record_sizes);

    MPI_Finalize();
    return 0;
}
