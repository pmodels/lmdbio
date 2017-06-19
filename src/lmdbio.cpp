/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *  (C) 2017 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "lmdbio.h"
#include <map>
#include <iostream>
#include <assert.h>
#include <new>

using std::cout;
using std::endl;

void lmdbio::db::init(MPI_Comm parent_comm, const char* fname, int batch_size)
{
#ifdef BENCHMARK
  this->mpi_time = 0.0;
  this->set_record_time = 0.0;
#endif

  //MPI_Comm_dup(parent_comm, &global_comm);

  /* initialize class attributes */
  global_np = 0; 
  global_rank = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &global_np);
  MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

  cout << "Global rank " << global_rank << endl;
  cout << "Global np " << global_np << endl;

  this->batch_size = batch_size;
  this->subbatch_size = batch_size / global_np;
  cout << "Subbatch size " << this->subbatch_size << endl;

  assign_readers(fname, batch_size); 
}

/* assign one reader per node */
void lmdbio::db::assign_readers(const char* fname, int batch_size) {
  int size = 0;

  local_rank = 0;
  local_np = 0;
  readers = 0;
  reader_id = 0;

  /* communicator between processes within a node */
  local_comm = MPI_COMM_NULL;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, global_rank,
      MPI_INFO_NULL, &local_comm);
  assert(local_comm != MPI_COMM_NULL);
  MPI_Comm_size(local_comm, &local_np);
  MPI_Comm_rank(local_comm, &local_rank);

  /* get number of readers */
  int is_reader_ = is_reader(local_rank) ? 1 : 0;

  /* communicator between readers */
  reader_comm = MPI_COMM_NULL;
  MPI_Comm_split(MPI_COMM_WORLD, is_reader_, global_rank, &reader_comm);
  assert(reader_comm != MPI_COMM_NULL);
  MPI_Comm_size(reader_comm, &readers);
  MPI_Comm_rank(reader_comm, &reader_id);

  /* open database and set fetch size */
  if (is_reader(local_rank)) {
    char hostname[256];
    gethostname(hostname, 255);
    cout << "Number of readers " << readers << endl;
    cout << "Rank " << global_rank << " is a reader id " << reader_id << 
      " on host " << hostname << endl;

    fetch_size = batch_size / readers;
    this->batch_ptrs = new char*[fetch_size];
    open_db(fname);

    /* get size */
    if (global_rank == 0) {
      size = lmdb_value_size();
    }
  }

  /* broadcast a size of data to allocate the shared buffer */
  MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  this->win_size = subbatch_size * size * 2;
  MPI_Win_allocate_shared(sizeof(int) * subbatch_size, sizeof(int),
      MPI_INFO_NULL, local_comm, &sizes, &size_win);
  MPI_Win_allocate_shared(win_size, sizeof(char), MPI_INFO_NULL, local_comm,
      &subbatch_bytes, &batch_win);

  /* allocate record's array */
  this->records = new (std::nothrow) record[subbatch_size];
}

/* open the database and initialize a position of a cursor */
void lmdbio::db::open_db(const char* fname) {
  int flags = MDB_RDONLY | MDB_NOTLS | MDB_NOLOCK;
  check_lmdb(mdb_env_create(&mdb_env_), "Created environment", false);
  check_lmdb(mdb_env_set_maxreaders(mdb_env_, readers), "Set maxreaders",
      false);
  check_lmdb(mdb_env_open(mdb_env_, fname, flags, 0664),
      "Opened environment", false);
  printf("Rank %d source %s\n", global_rank, fname);
  check_lmdb(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn),
      "Begun transaction", false);
  check_lmdb(mdb_dbi_open(mdb_txn, NULL, 0, &mdb_dbi_),
      "Opened database", false);
  check_lmdb(mdb_cursor_open(mdb_txn, mdb_dbi_, &mdb_cursor),
      "Opened cursor", false);
  lmdb_init_cursor();
}

/* a function for readers */
void lmdbio::db::read_batch() {
  int size = 0;
  int id = 0;
  int count = 0;
#ifdef BENCHMARK
  struct rusage rstart, rend;
  double ttime, utime, stime, sltime, start, end;

  MPI_Barrier(MPI_COMM_WORLD);
  start = MPI_Wtime();
  getrusage(RUSAGE_SELF, &rstart);
#endif
  /* compute size of send buffer and get pointers */
  count = 0;
  for (int i = 0; i < fetch_size; i++) {
    size = lmdb_value_size();
    batch_ptrs[i] = (char*) lmdb_value_data();
    sizes[i] = size;
    count += size;
    lmdb_next();
  }

  /* determine if the data is larger than a buffer */
  assert(count <= win_size * global_np);

  /* move a cursor to the next location */
  if (global_np != 1)
    lmdb_next_fetch();

#ifdef BENCHMARK
  getrusage(RUSAGE_SELF, &rend);
  end = MPI_Wtime();
  ttime = get_elapsed_time(start, end);
  utime = get_utime(rstart, rend);
  stime = get_stime(rstart, rend);
  sltime = get_sltime(ttime, utime, stime);
  read_stat.add_stat(get_ctx_switches(rstart, rend), 
      get_inv_ctx_switches(rstart, rend),
      ttime, utime, stime, sltime);

  MPI_Barrier(MPI_COMM_WORLD);
  start = MPI_Wtime();
  getrusage(RUSAGE_SELF, &rstart);
#endif

  count = 0;
  for (int i = 0; i < fetch_size; i++) {
    size = sizes[i];
    if (i % subbatch_size == 0)
      count = (i / subbatch_size) * win_size;
    memcpy(subbatch_bytes + count, batch_ptrs[i], size);
    count += size;
  }

#ifdef BENCHMARK
  getrusage(RUSAGE_SELF, &rend);
  end = MPI_Wtime();
  ttime = get_elapsed_time(start, end);
  utime = get_utime(rstart, rend);
  stime = get_stime(rstart, rend);
  sltime = get_sltime(ttime, utime, stime);
  parse_stat.add_stat(get_ctx_switches(rstart, rend), 
      get_inv_ctx_switches(rstart, rend),
      ttime, utime, stime, sltime);
#endif

}

void lmdbio::db::send_batch() {
  int count = 0;
  int size = 0;
#ifdef BENCHMARK
  double start;
  MPI_Barrier(MPI_COMM_WORLD);
  start = MPI_Wtime();
#endif
  /* send data size */
  MPI_Scatter(send_sizes, subbatch_size, MPI_INT, sizes, subbatch_size, MPI_INT,
      0, local_comm);
  for (int i = 0; i < subbatch_size; i++) {
    count += sizes[i];
  }

  subbatch_bytes = (char*) malloc(count);

  /* send sub-batch */
  MPI_Scatterv(batch_bytes, send_counts, send_displs, MPI_BYTE,
      subbatch_bytes, count, MPI_BYTE, 0, local_comm);
#ifdef BENCHMARK
  mpi_time += get_elapsed_time(start, MPI_Wtime());

  MPI_Barrier(MPI_COMM_WORLD);
  start = MPI_Wtime();
#endif
  count = 0;
  for (int i = 0; i < subbatch_size; i++) {
    size = sizes[i];
    records[i].set_record(subbatch_bytes + count, size);
    count += size;
  }
#ifdef BENCHMARK
  set_record_time += get_elapsed_time(start, MPI_Wtime());
#endif
}

/* set records */
void lmdbio::db::set_records() {
  int count = 0;
  int size = 0;
#ifdef BENCHMARK
  double start;
  MPI_Barrier(MPI_COMM_WORLD);
  start = MPI_Wtime();
#else
  MPI_Barrier(local_comm);
#endif
  for (int i = 0; i < subbatch_size; i++) {
    size = sizes[i];
    records[i].set_record(subbatch_bytes + count, size);
    count += size;
  }
  int fsize = records[0].get_record_size();
#ifdef BENCHMARK
 set_record_time += get_elapsed_time(start, MPI_Wtime());
#endif
}

/* a process with local rank = 0 is a reader  */
bool lmdbio::db::is_reader(int local_rank) {
  return local_rank == 0;
}

bool lmdbio::db::is_reader() {
  return is_reader(local_rank);
}

int lmdbio::db::read_record_batch(void) 
{
  if (is_reader())
    read_batch();
  set_records();
  return 0;
}

int lmdbio::db::get_batch_size() {
  return batch_size;
}

int lmdbio::db::get_num_records() {
  return subbatch_size;
}

lmdbio::record* lmdbio::db::get_record(int i) {
  return &records[i];
}

#ifdef BENCHMARK
/* calculate users time */
double lmdbio::db::get_utime(rusage rstart, rusage rend) {
  return 1e6 * (rend.ru_utime.tv_sec - rstart.ru_utime.tv_sec) +
    rend.ru_utime.tv_usec - rstart.ru_utime.tv_usec;
}

/* calculate kernel time */
double lmdbio::db::get_stime(rusage rstart, rusage rend) {
  return 1e6 * (rend.ru_stime.tv_sec - rstart.ru_stime.tv_sec) +
    rend.ru_stime.tv_usec - rstart.ru_stime.tv_usec;
}

/* calculate sleep time */
double lmdbio::db::get_sltime(double ttime, double utime, double stime) {
  return ttime - utime - stime;
}

double lmdbio::db::get_ctx_switches(rusage rstart, rusage rend) {
  return rend.ru_nvcsw - rstart.ru_nvcsw;
}

double lmdbio::db::get_inv_ctx_switches(rusage rstart, rusage rend) {
  return rend.ru_nivcsw - rstart.ru_nivcsw;
}

double lmdbio::db::get_elapsed_time(double start, double end) {
  return 1e6 * (end - start);
}

double lmdbio::db::get_mpi_time() {
  return mpi_time;
}

double lmdbio::db::get_set_record_time() {
  return set_record_time;
}

lmdbio::io_stat lmdbio::db::get_read_stat() {
  return read_stat;
}

lmdbio::io_stat lmdbio::db::get_parse_stat() {
  return parse_stat;
}
#endif
