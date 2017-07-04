/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *  (C) 2017 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "lmdbio.h"
#include <map>
#include <iostream>
#include <unistd.h>
#include <assert.h>
#include <new>

using std::cout;
using std::endl;

void lmdbio::db::init(MPI_Comm parent_comm, const char* fname, int batch_size)
{
#ifdef BENCHMARK
  double start, end;
  this->mpi_time = 0.0;
  this->set_record_time = 0.0;
  this->init_var_time = 0.0;
  this->init_db_time = 0.0;
  this->init_db_1_time = 0.0;
  this->init_db_barrier_1_time = 0.0;
  this->open_db_time = 0.0;
  this->init_db_barrier_2_time = 0.0;
  this->init_db_2_time = 0.0;
  start = MPI_Wtime();
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
#ifdef BENCHMARK
  end = MPI_Wtime();
  this->init_var_time = get_elapsed_time(start, end);
  start = MPI_Wtime();
#endif
  assign_readers(fname, batch_size);
#ifdef BENCHMARK
  end = MPI_Wtime();
  this->init_db_time = get_elapsed_time(start, end);
#endif
}

/* assign one reader per node */
void lmdbio::db::assign_readers(const char* fname, int batch_size) {
  int size = 0;
  string mmap_env;
  local_rank = 0;
  local_np = 0;
  readers = 0;
  reader_id = 0;
  fetch_size = 0;

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
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (is_reader(local_rank)) {
#ifdef BENCHMARK
    double end_db, start_db;
    start_db = MPI_Wtime();
#endif
    //cout << "OPEN DB" << endl;
    open_db(fname);

#ifdef BENCHMARK
    end_db = MPI_Wtime();
    this->open_db_time = get_elapsed_time(start_db, end_db);
#endif
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (global_rank == 0) {
    /* get size */
    size = lmdb_value_size();
  }

  /* broadcast a size of data to allocate the shared buffer */
  MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  

  this->win_size = subbatch_size * size * 2;

  /* allocate neccessary buffer */
  cout << "FETCH SIZE " << fetch_size << endl;
  if (is_reader(local_rank))
    batch_ptrs = new char*[fetch_size];
  if (dist_mode == MODE_SCATTERV) {
    if (is_reader(local_rank)) {
      send_sizes = new int[fetch_size];
      send_displs = new int[local_np];
      send_counts = new int[local_np];
      batch_bytes = (char*) malloc(fetch_size * size * 2 * sizeof(char));
    }
    sizes = new int[subbatch_size];
    subbatch_bytes = (char*) malloc(win_size);
  }
  else if (dist_mode == MODE_SHMEM) {
    MPI_Win_allocate_shared(subbatch_size * sizeof(int), sizeof(int),
        MPI_INFO_NULL, local_comm, &sizes, &size_win);
    MPI_Win_allocate_shared(win_size, sizeof(char), MPI_INFO_NULL, local_comm,
        &subbatch_bytes, &batch_win);
  }

  /* allocate record's array */
  this->records = new (std::nothrow) record[subbatch_size];
}

/* open the database and initialize a position of a cursor */
void lmdbio::db::open_db(const char* fname) {
  int flags = MDB_RDONLY | MDB_NOTLS | MDB_NOLOCK;
  size_t addr;
  char addr_str[100]; // enough to hold all numbers up to 64-bits
  int max_try = 200;
  int rc = 0;
  int success = 0;

  //srand(time(NULL));
  check_lmdb(mdb_env_create(&mdb_env_), "Created environment", false);
  check_lmdb(mdb_env_set_maxreaders(mdb_env_, readers), "Set maxreaders",
      false);

#ifdef ICPADS
  size_t foos[] = { 7980184936448, 8639120420864, 2858116001792 };

  /* random an address and fix mmap's address */
  for (int i = 0; i < 3; i++) {
    if (global_rank == 0) {
      //addr = (size_t) PAGE_SIZE * rand();
      addr = foos[i];
      //addr = (size_t) 7980184936448;
      snprintf(addr_str, 100, "MMAP_ADDRESS=%llu", addr);
      cout << addr_str << endl;
      putenv(addr_str);
      cout << "mapping file " << fname << endl;
      rc = mdb_env_open(mdb_env_, fname, flags, 0664);
      cout << "reader " << reader_id << " error code " << rc << endl;
      if (rc != 0) {
        cout << "mmap failed; trying again\n";
        continue;
      }
    }
    MPI_Bcast(addr_str, 100, MPI_CHAR, 0, reader_comm);
    if (global_rank != 0) {
      putenv(addr_str);
      rc = mdb_env_open(mdb_env_, fname, flags, 0664);
      cout << "reader " << reader_id << " error code " << rc << endl;
    }
    MPI_Allreduce(&rc, &success, 1, MPI_INT, MPI_MAX, reader_comm);
    if (success == 0)
      break;
    if (rc == 0)
      mdb_env_close(mdb_env_);
    rc = success;
  }
  if (rc != 0) {
    cout << "Cannot open db" << endl;
    exit(1);
  }
  /* set lmdb buffer */
  lmdb_buffer = (char*) addr;
#else
  rc = mdb_env_open(mdb_env_, fname, flags, 0664);
  cout << "reader " << reader_id << " error code " << rc << endl;
#endif

  check_lmdb(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn),
      "Begun transaction", false);
  check_lmdb(mdb_dbi_open(mdb_txn, NULL, 0, &mdb_dbi_),
      "Opened database", false);
  check_lmdb(mdb_cursor_open(mdb_txn, mdb_dbi_, &mdb_cursor),
      "Opened cursor", false);
  lmdb_init_cursor();
  sample_size = lmdb_value_size();
  sample_size = (sample_size & (~(PAGE_SIZE - 1))) 
    + (PAGE_SIZE * !!(sample_size % PAGE_SIZE));
  //cout << "sample size " << sample_size << " fetch_size " << fetch_size << 
  //  " page_size "  << PAGE_SIZE << endl;
  num_read_pages = (sample_size * fetch_size) / PAGE_SIZE;
  start_pg = reader_id * num_read_pages;
  end_pg = start_pg + num_read_pages;
  //cout << "Reader " << reader_id << ": Touching from " << start_pg << " to " 
  //  << end_pg << endl;
  //lmdb_touch_pages();
}

/* a function for readers */
void lmdbio::db::read_batch() {
  int *read_sizes;
  int size = 0;
  int id = 0;
  int count = 0;
  int cursor_buffer_size = 0;
  void* start_cursor_buffer;
  void* end_cursor_buffer;
#ifdef BENCHMARK
  struct rusage rstart, rend;
  double ttime, utime, stime, sltime, start, end;

  //MPI_Barrier(reader_comm);
  start = MPI_Wtime();
  getrusage(RUSAGE_SELF, &rstart);
#endif

#ifdef ICPADS
  cout << "touching pages\n";
  lmdb_touch_pages();
  cout << "done touching pages\n";

  if (reader_id != 0) {
    //cout << "receiving cursor buf size\n";
    MPI_Recv(&cursor_buffer_size, 1, MPI_INT, reader_id - 1, 0, reader_comm, 
        MPI_STATUS_IGNORE);
    //cout << "got cursor buffer size " << cursor_buffer_size << endl;
    start_cursor_buffer = malloc(cursor_buffer_size);
    //cout << "allocated cursor buffer " << start_cursor_buffer << endl;
    MPI_Recv(start_cursor_buffer, cursor_buffer_size, MPI_BYTE, reader_id - 1, 0, 
        reader_comm, MPI_STATUS_IGNORE);
    //cout << "received data from the cursor buffer\n";
    mdb_deserialize_cursor(start_cursor_buffer, cursor_buffer_size, mdb_cursor);
    //cout << "done deserializing\n";
    //free(cursor_buffer);
  }
  else {
    mdb_serialize_cursor(mdb_cursor, &start_cursor_buffer, &cursor_buffer_size);
  }

  if (reader_id != readers - 1) {
    /* shift the cursor */
    lmdb_seek_multiple(fetch_size);

    //cout << "serializing buffer\n";
    mdb_serialize_cursor(mdb_cursor, &end_cursor_buffer, &cursor_buffer_size);
    //cout << "serialized buffer into " << end_cursor_buffer << " size " << cursor_buffer_size << endl;
    MPI_Send(&cursor_buffer_size, 1, MPI_INT, reader_id + 1, 0, reader_comm);
    MPI_Send(end_cursor_buffer, cursor_buffer_size, MPI_BYTE, reader_id + 1, 0, 
        reader_comm);
    //cout << "sent end cursor buffer\n";
    free(end_cursor_buffer);
    /* restore the cursor */
    //cout << "deserializing cursor buffer " << start_cursor_buffer << " size " << cursor_buffer_size << endl;
    mdb_deserialize_cursor(start_cursor_buffer, cursor_buffer_size, mdb_cursor);
    //cout << "done deserializing cursor buffer " << start_cursor_buffer << " size " << cursor_buffer_size << endl;
    free(start_cursor_buffer);
  }
#endif

  /* compute size of send buffer and get pointers */
  count = 0;
  total_byte_size = 0;
  if (dist_mode == MODE_SCATTERV) {
    read_sizes = this->send_sizes;
    for (int i = 0; i < fetch_size; i++) {
      if (i % subbatch_size == 0) {
        send_displs[id] = total_byte_size;
        //cout << "send displs " << id << " " << total_byte_size;
        total_byte_size += count;
        if (i != 0)
          send_counts[id - 1] = count;
        id++;
        count = 0;
      }
      this->batch_ptrs[i] = (char*) lmdb_value_data();
      size = lmdb_value_size();
      count += size;
      read_sizes[i] = size;
      lmdb_next();
    }
    total_byte_size += count;
    send_counts[id - 1] = count;
  }
  else if (dist_mode == MODE_SHMEM) {
    read_sizes = this->sizes;
    lmdb_get_current();
    for (int i = 0; i < fetch_size; i++) {
      cout << "Reader " << reader_id << " reads item " << i << " key " << key() << endl;
      size = lmdb_value_size();
      batch_ptrs[i] = (char*) lmdb_value_data();
      read_sizes[i] = size;
      total_byte_size += size;
      lmdb_next();
    }
  }

  /* determine if the data is larger than a buffer */
  assert(total_byte_size <= win_size * global_np);

#ifdef ICPADS
  /* move a cursor to the next location */
  if (read_mode == MODE_STRIDE) {
    if (reader_id == readers - 1) {
      mdb_serialize_cursor(mdb_cursor, &end_cursor_buffer, &cursor_buffer_size);
      MPI_Send(&cursor_buffer_size, 1, MPI_INT, 0, 0, reader_comm);
      MPI_Send(end_cursor_buffer, cursor_buffer_size, MPI_BYTE, 0, 0, reader_comm);
      free(end_cursor_buffer);
    }
    else if (reader_id == 0) {
      MPI_Recv(&cursor_buffer_size, 1, MPI_INT, readers - 1, 0, reader_comm, 
          MPI_STATUS_IGNORE);
      start_cursor_buffer = malloc(cursor_buffer_size);
      MPI_Recv(start_cursor_buffer, cursor_buffer_size, MPI_BYTE, readers - 1, 0, 
          reader_comm, MPI_STATUS_IGNORE);
      mdb_deserialize_cursor(start_cursor_buffer, cursor_buffer_size, mdb_cursor);
    }
  }
#else
  /* shift the cursor */
  lmdb_seek_multiple(fetch_size * (readers - 1));
#endif


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

  //MPI_Barrier(reader_comm);
  start = MPI_Wtime();
  getrusage(RUSAGE_SELF, &rstart);
#endif

  count = 0;
  if (dist_mode == MODE_SCATTERV) {
    for (int i = 0; i < fetch_size; i++) {
      size = read_sizes[i];
      memcpy(batch_bytes + count, batch_ptrs[i], size);
      count += size;
    }
  }
  else if (dist_mode == MODE_SHMEM) {
    for (int i = 0; i < fetch_size; i++) {
      size = read_sizes[i];
      if (i % subbatch_size == 0)
        count = (i / subbatch_size) * win_size;
      memcpy(subbatch_bytes + count, batch_ptrs[i], size);
      count += size;
    }
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
  start = MPI_Wtime();
#endif
  //cout << "send batch " << send_sizes[0] << endl;
  /* send data size */
  MPI_Scatter(send_sizes, subbatch_size, MPI_INT, sizes, subbatch_size, MPI_INT,
      0, local_comm);

  //cout << "scatter size completed" << endl; 
  /* get total size */
  for (int i = 0; i < subbatch_size; i++) {
    count += sizes[i];
  }

  //cout << "count size completed" << endl;
  /* send sub-batch */
  MPI_Scatterv(batch_bytes, send_counts, send_displs, MPI_BYTE,
      subbatch_bytes, count, MPI_BYTE, 0, local_comm);

  //cout << "scatterv batch completed" << endl;
#ifdef BENCHMARK
  mpi_time += get_elapsed_time(start, MPI_Wtime());
#endif
}

/* set records */
void lmdbio::db::set_records() {
  int count = 0;
  int size = 0;
#ifdef BENCHMARK
  double start;
#endif
  if (dist_mode == MODE_SHMEM)
    MPI_Barrier(local_comm);
#ifdef BENCHMARK
  start = MPI_Wtime();
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

void lmdbio::db::set_mode(int dist_mode, int read_mode) {
  if (dist_mode == MODE_SCATTERV)
    cout << "Set dist mode to SCATTERV" << endl;
  else if (dist_mode == MODE_SHMEM)
    cout << "Set dist mode to SHMEM" << endl;
  if (read_mode == MODE_STRIDE)
    cout << "Set read mode to STRIDE" << endl;
  else if (read_mode == MODE_CONT)
    cout << "Set read mode to CONT" << endl;

  this->dist_mode = dist_mode;
  this->read_mode = read_mode;
}

void lmdbio::db::lmdb_touch_pages() {
  int tmp = 0;
  for (size_t i = start_pg;  i < end_pg; i++) {
    tmp += lmdb_buffer[i] + (PAGE_SIZE * i);
  }
  start_pg += num_read_pages;
  end_pg += num_read_pages;
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
  if (dist_mode == MODE_SCATTERV)
    send_batch();
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

double lmdbio::db::get_init_var_time() {
  return init_var_time;
}

double lmdbio::db::get_init_db_time() {
  return init_db_time;
}

double lmdbio::db::get_init_db_1_time() {
  return init_db_1_time;
}

double lmdbio::db::get_init_db_barrier_1_time() {
  return init_db_barrier_1_time;
}

double lmdbio::db::get_open_db_time() {
  return open_db_time;
}

double lmdbio::db::get_init_db_barrier_2_time() {
  return init_db_barrier_2_time;
}

double lmdbio::db::get_init_db_2_time() {
  return init_db_2_time;
}

lmdbio::io_stat lmdbio::db::get_read_stat() {
  return read_stat;
}

lmdbio::io_stat lmdbio::db::get_parse_stat() {
  return parse_stat;
}


#endif
