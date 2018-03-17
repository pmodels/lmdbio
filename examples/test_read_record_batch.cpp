#include <iostream>
#include "lmdbio.h"

#define BATCH_SIZE (10)
#define MAX_ITER (12)
#define READER_SIZE (1)
#define PREFETCH_SIZE (4)
#define STAGGERING_SIZE (0)
#define BULK_READ_NUM_BATCHES (2)

int main() {
    int num_records = 0, rank = 0, total_num_records = 0;
    const char *filename = "/lcrc/project/radix/pumma/dl_benchmark/data/cifar10_alexnet_data/cifar10_alexnet_train_lmdb";
    const int* bulk_sizes;
    const void* bulk_bytes;
    const long long int* bulk_offsets;

    MPI_Init(NULL, NULL);

    lmdbio::db *db = new lmdbio::db();
    db->set_mode(
        lmdbio::dist_mode_enum::SHMEM,
        lmdbio::read_mode_enum::STRIDE,
        lmdbio::prov_info_mode_enum::DISABLE);
    db->set_stagger_size(STAGGERING_SIZE);
    db->init(
        MPI_COMM_WORLD,
        filename,
        BATCH_SIZE,
        READER_SIZE,
        PREFETCH_SIZE,
        MAX_ITER);

    for (int iter = 0; iter < MAX_ITER; iter++) {
      std::cout << "iter " << iter << std::endl;
      db->read_record_batch();
      for (int item_id = 0; item_id < db->get_num_records(); item_id++) {
        auto record = db->get_record(item_id);
        std::cout << "item " << item_id << ", size " << record->get_record_size()
          << std::endl;
      }
    }

    delete db;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
