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
#include "lmdb.h"
#include <vector>
#include <string>

using std::vector;
using std::string;

#define TEST_PHASE (0)
#define TRAIN_PHASE (1)

namespace lmdbio {

class record {
public:
  record()
  {
    data = NULL;
    record_size = 0;
  }

  ~record() {}

  int get_record_size() { return record_size; }
  void *get_record() { return data; }
  void set_record(void* data, int record_size) {
    this->data = data;
    this->record_size = record_size;
  }

private:
  void *data;
  int record_size;
};

class db
{
public:
  db(MPI_Comm parent_comm, const char *filename, int batch_size, 
      int phase, int max_iter) {
    MPI_Comm_dup(parent_comm, &comm);
    fname = strdup(filename);
    this->batch_size = batch_size;
    this->phase = phase;  
    this->max_iter = max_iter;
    this->num_records = 0;
    records = NULL;
    init();
    compute_num_readers();
  }

  ~db(void) {
    MPI_Comm_free(&comm);
    delete[] fname;
    delete records;
  }

  int read_record_batch(void ***records, int *num_records, int **record_sizes);
  int get_sbatch_size() const { return sbatch_size; }
  int get_batch_size(void) { return batch_size; }
  int read_record_batch(void);
  int get_num_records(void) { return get_sbatch_size(); }
  record *get_record(int i) { return &records[i]; }
  void set_readers(int readers);

private:
  MPI_Comm comm;
  const char *fname;
  record *records;
  char** batch_ptrs;
  int num_records;
  int phase;
  void init();
  void compute_num_readers();
  int datum_byte_size;
  int batch_size;
  int sbatch_size;
  MDB_cursor* cursor;
  int rank;
  int np;
  int current_batch;
  void open_db();
  //void validate_cursor();
  int get_sbatch_size(int sbatch_id);
  void read_batch();
  void parse_sbatch_bytes(char* sbatch_bytes, int* r_rsize);
  bool is_identical_sbatch_size();
  void send_batch(char* batch_bytes, char** sbatch_bytes, int* s_rsize, 
      int** r_rsize);
  void serialize_data(char** batch_bytes, int** s_rsize);
  int max_iter;
  int total_images;
  int fetch_size;
  int fetch_batch_num;
  bool has_diff_data_size;
  bool has_diff_batch_size;
  bool is_full_batch;
  int total_byte_size;
  vector<int> rsize_vec;
  int rsize_len;
  int* sbatch_sendcounts;
  int* sbatch_senddispls;
  int* sbatch_recvcounts;
  int* sbatch_recvdispls;
  int readers;
  void check_diff_batch();
  bool is_reader(int proc);
  int recv_group(int proc, int receiver_size);
  int reader_id(int proc);
  int read_group(int proc, int reader_per_batch);
  void dist_alltoallv(char* sbuf, char* rbuf, int* sendcounts, int* senddispls,
      int* recvcounts, int* recvdispls, int* s_rsize, int* r_rsize);
  void rsize_alltoallv(int* s_rsize, int* r_rsize, int* sendcounts,
      int* senddispls, int* recvcounts, int* recvdispls);
  void dist_scatter(char* sbuf, char* rbuf);
  void adjust_readers(int fetch_images, bool invalid);
  void compute_fetch_size(bool invalid);
  void init_readers(void);
  bool has_comm;
  string dist_mode;
  bool is_init;

  // DB variables
  MDB_env* mdb_env_;
  MDB_txn* mdb_txn;
  MDB_cursor* mdb_cursor;
  MDB_dbi mdb_dbi_;
  MDB_val mdb_key_, mdb_value_;
  int valid_;

  void check_lmdb(int success, const char* msg, bool verbose = true) {
    if (success == 0) {
      if (verbose)
        std::cout << "Successfully " << msg << std::endl;
    }
    else {
      std::cout << "Not successfully " << msg << std::endl;
    }
  }

  void printl(const char* msg) {
    std::cout << msg << std::endl;
  }

  void lmdb_seek(MDB_cursor_op op) {
    int mdb_status = mdb_cursor_get(mdb_cursor, &mdb_key_, &mdb_value_, op);
    if (mdb_status == MDB_NOTFOUND) {
      valid_ = false;
    } else {
      check_lmdb(mdb_status, "Seek", false);
      valid_ = true;
    }
  }

  void lmdb_seek_to_first() {
      lmdb_seek(MDB_FIRST);
  }

  void lmdb_next() {
      lmdb_seek(MDB_NEXT);
  }

  void lmdb_seek_multiple(int skip_size) {
    int mdb_status = 0;
    for (int i = 0; i < skip_size - 1; i++) {
      mdb_status = mdb_cursor_get(mdb_cursor, NULL, NULL, MDB_NEXT);
      if (mdb_status == MDB_NOTFOUND) {
        lmdb_seek_to_first();
      } else {
        check_lmdb(mdb_status, "Seek multiple", false);
      }
    }
    lmdb_seek(MDB_NEXT);
  }

  void lmdb_next_fetch() {
    lmdb_seek_multiple((readers - 1) * fetch_size);
  }

  void lmdb_init_cursor() {
    lmdb_seek_to_first();
    if (rank != 0)
      lmdb_seek_multiple(reader_id(rank) * fetch_size);
  }

  size_t lmdb_value_size() {
    return mdb_value_.mv_size;
  }

  void* lmdb_value_data() {
    return mdb_value_.mv_data;
  }
};

};

#endif  /* LMDBIO_H_INCLUDED */
