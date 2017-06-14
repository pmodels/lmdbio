#include <iostream>
#include "lmdbio.h"

#define BATCH_SIZE (2)
#define MAX_ITER (1)

int main()
{
    void **records;
    int num_records, *record_sizes;
    const char *filename = "/home/pumma/cifar10_alexnet_train_lmdb";
    MPI_Init(NULL, NULL);

    lmdbio::db *db = new lmdbio::db(MPI_COMM_WORLD, filename, BATCH_SIZE);

    std::cout << "batch size: " << db->get_batch_size() << std::endl;

    int total_read_size = 0;
    for (int iter = 0; iter < MAX_ITER; iter++) {
      //std::cout << "read record batch " << std::endl;
      db->read_record_batch();

      std::cout << "num records: " << db->get_num_records() << std::endl;

      for (int i = 0; i < db->get_num_records(); i++) {

        lmdbio::record *record = db->get_record(i);

        total_read_size += record->get_record_size();
      }
    }

    std::cout << "total read size: " << total_read_size << std::endl;

    delete db;

    MPI_Finalize();
    return 0;
}
