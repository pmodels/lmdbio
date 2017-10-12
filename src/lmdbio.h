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
#include <sys/time.h>
#include <sys/resource.h>
#include <vector>
#include <string>
#include <sys/mman.h>

using std::vector;
using std::string;

#define TEST_PHASE (0)
#define TRAIN_PHASE (1)
#define MODE_SCATTERV (0)
#define MODE_SHMEM (1)
#define MODE_STRIDE (0)
#define MODE_CONT (1)

namespace lmdbio {
class record {
public:
  record() {
    data = NULL;
    record_size = 0;
  }

  ~record() {}

  int get_record_size() { return record_size; }
  void *get_record() { return data; }
  void set_record(void* data, int record_size) {
    string str = string(static_cast<const char*>(data),
        record_size);
    this->data = data;
    this->record_size = record_size;
  }

public:
  void *data;
  int record_size;
};

#ifdef BENCHMARK
class io_stat {
public:
  io_stat() {
    ctx_switches = 0.0;
    inv_ctx_switches = 0.0;
    ttime = 0.0;
    utime = 0.0;
    stime = 0.0;
    sltime = 0.0;
  }

  void add_stat(double ctx_switches, double inv_ctx_switches,
      double ttime, double utime, double stime, double sltime) {
    this->ctx_switches += ctx_switches;
    this->inv_ctx_switches += inv_ctx_switches;
    this->ttime += ttime;
    this->utime += utime;
    this->stime += stime;
    this->sltime += sltime;
  }

  void set_stat(double ctx_switches, double inv_ctx_switches,
      double ttime, double utime, double stime, double sltime) {
    this->ctx_switches = ctx_switches;
    this->inv_ctx_switches = inv_ctx_switches;
    this->ttime = ttime;
    this->utime = utime;
    this->stime = stime;
    this->sltime = sltime;
  }

  double get_ctx_switches() { return ctx_switches; }
  double get_inv_ctx_switches() { return inv_ctx_switches; }
  double get_ttime() { return ttime; }
  double get_utime() { return utime; }
  double get_stime() { return stime; }
  double get_sltime() { return sltime; }

private:
  double ctx_switches;
  double inv_ctx_switches;
  double ttime;
  double utime;
  double stime;
  double sltime;
};
#endif

class db
{
public:
  db() {
    dist_mode = MODE_SHMEM;
    read_mode = MODE_STRIDE;
  }

  void init(MPI_Comm parent_comm, const char *fname, int batch_size);
  void set_mode(int dist_mode, int read_mode);

  ~db() {
      if (dist_mode == MODE_SHMEM) {
          MPI_Win_unlock_all(batch_win);
          MPI_Win_unlock_all(size_win);
          MPI_Win_free(&batch_win);
          MPI_Win_free(&size_win);
      }
    /*if (is_reader()) {
      mdb_cursor_close(mdb_cursor);
      mdb_dbi_close(mdb_env_, mdb_dbi_);
      mdb_env_close(mdb_env_);
    }
   
    if (mode == MODE_SHMEM) {
      MPI_Win_free(&batch_win);
      MPI_Win_free(&size_win);
    }
    delete[] records;*/
    //MPI_Comm_free(&global_comm);
  }

  int read_record_batch(void ***records, int *num_records, int **record_sizes);
  int get_batch_size(void);
  int read_record_batch(void);
  int get_num_records(void);
  record* get_record(int i);
  bool is_reader();

#ifdef BENCHMARK
  double get_mpi_time();
  double get_set_record_time();
  double get_init_var_time();
  double get_init_db_time();
  double get_init_db_1_time();
  double get_init_db_barrier_1_time();
  double get_open_db_time();
  double get_init_db_barrier_2_time();
  double get_init_db_2_time();

  io_stat get_read_stat();
  io_stat get_parse_stat();
#endif

private:
  MPI_Comm global_comm;
  MPI_Comm local_comm;
  MPI_Comm reader_comm;
  MPI_Win batch_win;
  MPI_Win size_win;
  record *records;
  int global_rank;
  int global_np;
  int local_rank;
  int local_np;
  int reader_id;
  char** batch_ptrs;
  int* send_sizes;
  int* sizes;
  int total_byte_size;
  int batch_size;
  int subbatch_size;
  int readers;
  int fetch_size;
  int* send_displs;
  int* send_counts;
  char* batch_bytes; 
  char* subbatch_bytes;
  int win_size;
  int win_displ;
  bool is_large_dataset;
  MDB_cursor* cursor;
  MDB_env* mdb_env_;
  MDB_txn* mdb_txn;
  MDB_cursor* mdb_cursor;
  MDB_dbi mdb_dbi_;
  MDB_val mdb_key_, mdb_value_;
  int valid_;
  int dist_mode;
  int read_mode;
  size_t mmap_addr;

  void assign_readers(const char* fname, int batch_size);
  void open_db(const char* fname);
  void send_batch();
  void read_batch();
  void check_diff_batch();
  bool is_reader(int local_rank);
  void set_records();

#ifdef BENCHMARK
  double mpi_time;
  double set_record_time;
  double init_var_time;
  double init_db_time;
  double init_db_1_time;
  double init_db_barrier_1_time;
  double open_db_time;
  double init_db_barrier_2_time;
  double init_db_2_time;
  io_stat read_stat;
  io_stat parse_stat;

  double get_utime(rusage rstart, rusage rend);
  double get_stime(rusage rstart, rusage rend);
  double get_sltime(double ttime, double utime, double stime);
  double get_ctx_switches(rusage rstart, rusage rend);
  double get_inv_ctx_switches(rusage rstart, rusage rend);
  double get_elapsed_time(double start, double end);
#endif
  
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

  void lmdb_print_stat() {
  }

  void lmdb_seek_to_first() {
      lmdb_seek(MDB_FIRST);
  }

  void lmdb_next() {
      lmdb_seek(MDB_NEXT);
  }

  void lmdb_seek_multiple(int skip_size) {
    int mdb_status = 0;
    MDB_envinfo stat;

    /* protect the buffer against read accesses */
    mdb_env_info(mdb_env_, &stat);
    char *e = getenv("ENABLE_MPROTECT");
    if (e && !strcmp(e, "1")) {
      printf("protecting buffer %p, starting seek %d\n", (void*) mmap_addr, skip_size - 1);
      mprotect((void*) mmap_addr, (size_t) stat.me_mapsize, PROT_NONE);
    }

    for (int i = 0; i < skip_size - 1; i++) {
      mdb_status = mdb_cursor_get(mdb_cursor, NULL, NULL, MDB_NEXT);
      if (mdb_status == MDB_NOTFOUND) {
        lmdb_seek_to_first();
      } else {
        check_lmdb(mdb_status, "Seek multiple", false);
      }
    }

    if (e && !strcmp(e, "1")) {
      printf("unprotecting buffer\n");
      mprotect((void*) mmap_addr, (size_t) stat.me_mapsize, PROT_READ);
    }

    lmdb_seek(MDB_NEXT);
  }

  void lmdb_next_fetch() {
    lmdb_seek_multiple((readers - 1) * fetch_size);
  }

  size_t lmdb_value_size() {
    return mdb_value_.mv_size;
  }

  void* lmdb_value_data() {
    return mdb_value_.mv_data;
  }

  void lmdb_init_cursor() {
    int offset = 0;
    lmdb_seek_to_first();
    //std::cout << "Read mode " << read_mode << " MODE_STRIDE " << 
    //  MODE_STRIDE << " MODE_CONT " << MODE_CONT <<  std::endl;
    if (reader_id != 0) {
      if (read_mode == MODE_STRIDE) {
        offset = fetch_size;
      }
      else if (read_mode == MODE_CONT) {
        MDB_stat stat;
        mdb_env_stat(mdb_env_, &stat);
        std::cout << "Number of records " << stat.ms_entries << std::endl;
        offset = stat.ms_entries / readers;
      }
      std::cout << "Reader " << reader_id << " offset " << offset << std::endl;
      lmdb_seek_multiple(reader_id * offset);
    }
  }

  string key() {
    return string(static_cast<const char*>(mdb_key_.mv_data), mdb_key_.mv_size);
  }
};

};

#endif  /* LMDBIO_H_INCLUDED */
