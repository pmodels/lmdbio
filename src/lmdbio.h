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
  db(MPI_Comm parent_comm, const char *filename, int batch_size) {
    MPI_Comm_dup(parent_comm, &global_comm);
    fname = strdup(filename);
    this->batch_size = batch_size;
    this->subbatch_size = 0;
    records = NULL;
    init();
  }

  ~db(void) {
    if (is_reader()) {
      mdb_cursor_close(mdb_cursor);
      mdb_dbi_close(mdb_env_, mdb_dbi_);
      mdb_env_close(mdb_env_);
    }

    MPI_Comm_free(&global_comm);
    // close database
    delete[] fname;
    delete records;
  }

  int read_record_batch(void ***records, int *num_records, int **record_sizes);
  int get_batch_size(void) { return batch_size; }
  int read_record_batch(void);
  int get_num_records(void) { return subbatch_size; }
  record *get_record(int i) { return &records[i]; }

private:
  MPI_Comm global_comm;
  MPI_Comm local_comm;
  MPI_Comm reader_comm;
  int global_rank;
  int global_np;
  int local_rank;
  int local_np;
  int reader_id;
  const char *fname;
  record *records;
  char** batch_ptrs;
  int* send_sizes;
  int* sizes;
  int total_byte_size;
  int batch_size;
  int subbatch_size;
  int readers;
  int fetch_size;
  char* batch_bytes; 
  int* send_displs;
  int* send_counts;
  char* subbatch_bytes;
  MDB_cursor* cursor;
  MDB_env* mdb_env_;
  MDB_txn* mdb_txn;
  MDB_cursor* mdb_cursor;
  MDB_dbi mdb_dbi_;
  MDB_val mdb_key_, mdb_value_;
  int valid_;

  void init();
  void assign_readers();
  void open_db();
  void send_batch();
  void read_batch();
  void check_diff_batch();
  bool is_reader(int local_rank);
  bool is_reader();
  
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
    std::cout << "READER ID: " << reader_id << " rank " << global_rank << std::endl; 
    if (reader_id != 0)
      lmdb_seek_multiple(reader_id * fetch_size);
  }

  size_t lmdb_value_size() {
    return mdb_value_.mv_size;
  }

  void* lmdb_value_data() {
    return mdb_value_.mv_data;
  }

  string key() {
    return string(static_cast<const char*>(mdb_key_.mv_data), mdb_key_.mv_size);
  }
};

};

#endif  /* LMDBIO_H_INCLUDED */
