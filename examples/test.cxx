#include <iostream>
#include "lmdbio.h"

#define BATCH_SIZE (128)

int main()
{
    void **records;
    int num_records, *record_sizes;

    MPI_Init(NULL, NULL);

    //lmdbio obj(MPI_COMM_WORLD, "/dev/null", BATCH_SIZE);
    lmdbio::db *db = new lmdbio::db(MPI_COMM_WORLD, "/dev/null", BATCH_SIZE);
    std::cout << "batch size: " << db->get_batch_size() << std::endl;

    db->read_record_batch();

    for (int i = 0; i < db->get_num_records(); i++) {
      lmdbio::record *record = db->get_record(i);

      std::cout << "record size: " << record->get_record_size() << std::endl;
    }

    //obj.read_record_batch(&records, &num_records, &record_sizes);
    delete db;

    MPI_Finalize();
    return 0;
}
