#include <iostream>
#include "lmdbio.h"

#define BATCH_SIZE (5)
#define MAX_ITER (1)
#define READER_SIZE (2)

int main()
{
    int num_records = 0, rank = 0;
    const char *filename = "/lcrc/project/radix/pumma/dl_data/cifar10_alexnet_train_lmdb";
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    lmdbio::db *db = new lmdbio::db();
    db->set_mode(MODE_SHMEM, MODE_STRIDE);
    db->init(MPI_COMM_WORLD, filename, BATCH_SIZE, READER_SIZE);


    //std::cout << "batch size: " << db->get_batch_size() << std::endl;

    int total_read_size = 0;
    int size = 0;
    for (int iter = 0; iter < MAX_ITER; iter++) {
      std::cout << "rank " << rank << " read record batch" << std::endl;
      db->read_record_batch();

      std::cout << "rank " << rank << " num records " << iter << ": " << db->get_num_records() << std::endl;

      for (int i = 0; i < db->get_num_records(); i++) {

        lmdbio::record *record = db->get_record(i);
        size = record->get_record_size();
        total_read_size += size;
        //std::cout << "data size  " << i << ": " << size << std::endl;
        //std::string data = string(static_cast<const char*>(record->get_record()),
        //    size);
        //std::cout << "data in test " << i << ": " << data << std::endl;
        //std::cout << "" << std::endl;
      }
    }

    std::cout << "rank " << rank << " total read size: " << total_read_size << std::endl;

    delete db;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
