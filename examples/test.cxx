#include <iostream>
#include "lmdbio.h"

#define BATCH_SIZE (4)
#define TEST_PHASE (0)
#define TRAIN_PHASE (1)
#define MAX_ITER (1)

int main()
{
    void **records;
    int num_records, *record_sizes;
    const char *filename = "/lcrc/project/radix/pumma/dl_benchmark/data/cifar10_alexnet_data/cifar10_alexnet_train_lmdb";
    MPI_Init(NULL, NULL);

    lmdbio::db *db = new lmdbio::db(MPI_COMM_WORLD, filename, 
        BATCH_SIZE, TRAIN_PHASE, MAX_ITER);
    std::cout << "batch size: " << db->get_batch_size() << std::endl;

    db->read_record_batch();
    
    std::cout << "num records: " << db->get_num_records() << std::endl;

    for (int i = 0; i < db->get_num_records(); i++) {

      lmdbio::record *record = db->get_record(i);

      std::cout << "record size " << i << ": " 
        << record->get_record_size() << std::endl;
    }

    delete db;

    MPI_Finalize();
    return 0;
}
