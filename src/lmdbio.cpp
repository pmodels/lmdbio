/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *  (C) 2017 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "lmdbio.h"
#include <map>
#include <iostream>

using std::cout;
using std::endl;

void lmdbio::db::init()
{
  /* initialize class attributes */
  global_np = 0; 
  global_rank = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &global_np);
  MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

  cout << "global rank " << global_rank << endl;
  cout << "global np " << global_np << endl;

  subbatch_size = batch_size / global_np;
  sizes = (int*) malloc(sizeof(int) * subbatch_size);

  assign_readers(); 
}

/* assign one reader per node */
void lmdbio::db::assign_readers() {
  int size = 0;
  local_rank = 0;
  local_np = 0;
  readers = 0;
  reader_id = 0;

  MPI_Info info;
  MPI_Info_create(&info);

  /* communicator between processes within a node */
  MPI_Comm_split_type(global_comm, MPI_COMM_TYPE_SHARED, global_rank, info, 
      &local_comm);
  MPI_Comm_size(local_comm, &local_np);
  MPI_Comm_rank(local_comm, &local_rank);
  cout << "global rank " << global_rank <<  " local rank " << local_rank << endl;


  /* get number of readers */
  int is_reader_ = is_reader(local_rank) ? 1 : 0;

  /* communicator between readers */
  MPI_Comm_split(global_comm, is_reader_, global_rank, &reader_comm);
  MPI_Comm_size(reader_comm, &readers);
  MPI_Comm_rank(reader_comm, &reader_id);

  /* open database and set fetch size */
  if (is_reader(local_rank)) {
    cout << "num readers " << readers << endl;
    cout << "rank " << global_rank << " is a reader id " << reader_id << endl;

    fetch_size = batch_size / readers;

    open_db();
   
    batch_ptrs = (char**) malloc(sizeof(char*) * fetch_size);
    send_sizes = (int*) malloc(sizeof(int) * fetch_size);
    send_displs = (int*) malloc(sizeof(int) * local_np);
    send_counts = (int*) malloc(sizeof(int) * local_np);
  }

  records = (record *) operator new[] (subbatch_size * sizeof(record));
}

/* open the database and initialize a position of a cursor */
void lmdbio::db::open_db() {
  int flags = MDB_RDONLY | MDB_NOTLS | MDB_NOLOCK;
  check_lmdb(mdb_env_create(&mdb_env_), "Created environment", false);
  check_lmdb(mdb_env_set_maxreaders(mdb_env_, readers), "Set maxreaders", false);
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
  bool has_diff_data_size = false;
  int id = 0;
  int count = 0;
#ifdef BENCHMARK
  struct rusage rstart, rend;
  float ttime, utime, stime, sltime, start;

  MPI_Barrier(MPI_COMM_WORLD);
  start = MPI_Wtime();
  getrusage(RUSAGE_SELF, &rstart);
#endif
  //cout << "rank " << global_rank <<  " read batch -- fetch size " << fetch_size << endl;

  /* compute size of send buffer and get pointers */
  total_byte_size = 0;
  //cout << "subbatch size " << subbatch_size << endl;
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
    send_sizes[i] = size;
    //cout << "item " << i << " size " << size << " key " << key() << endl;
    lmdb_next();
  }
  total_byte_size += count;
  send_counts[id - 1] = count;

  /*for (int i = 0; i < local_np; i++) {
    cout << "send count " << i << " " << send_counts[i] << endl;
    cout << "send displs " << i << " " << send_displs[i] << endl;
  }*/

  /* move a cursor to the next location */
  if (global_np != 1)
    lmdb_next_fetch();

#ifdef BENCHMARK
  getrusage(RUSAGE_SELF, &rend);
  ttime = get_elapsed_time(start, MPI_Wtime());
  utime = get_utime(rstart, rend);
  stime = get_stime(rstart, rend);
  sltime = get_sltime(ttime, utime, stime);
  read_ctx_switches += get_ctx_switches(rstart, rend);
  read_inv_ctx_switches += get_inv_ctx_switches(rstart, rend);
  read_batch_time += ttime;
  read_utime += utime;
  read_stime += stime;
  read_sltime += sltime;

  MPI_Barrier(MPI_COMM_WORLD);
  start = MPI_Wtime();
  getrusage(RUSAGE_SELF, &rstart);
#endif

  /* prepare the send buffer (accessing memory) */
  batch_bytes = (char*) malloc (total_byte_size);
  count = 0;
  for (int i = 0; i < fetch_size; i++) {
    size = send_sizes[i]; 
    memcpy(batch_bytes + count, batch_ptrs[i], size);
    count += size;
  }

#ifdef BENCHMARK
  getrusage(RUSAGE_SELF, &rend);
  ttime = get_elapsed_time(start, MPI_Wtime());
  utime = get_utime(rstart, rend);
  stime = get_stime(rstart, rend);
  sltime = get_sltime(ttime, utime, stime);
  parse_ctx_switches += get_ctx_switches(rstart, rend);
  parse_inv_ctx_switches += get_inv_ctx_switches(rstart, rend);
  serialize_batch_time += ttime;
  parse_utime += utime;
  parse_stime += stime;
  parse_sltime += sltime;
#endif
}

void lmdbio::db::send_batch() {
  int count = 0;
  int size = 0;
#ifdef BENCHMARK
  float start;
  MPI_Barrier(MPI_COMM_WORLD);
  start = MPI_Wtime();
#endif
  /* send data size */
  MPI_Scatter(send_sizes, subbatch_size, MPI_INT, sizes, subbatch_size, MPI_INT, 
      0, local_comm);
  for (int i = 0; i < subbatch_size; i++) {
    //cout << "Rank " << local_rank << " " << sizes[i] << endl;
    count += sizes[i];
  }

  //cout << "rank " << global_rank << " count from scatter " << count << endl;
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
  send_batch();
  return 0;
}

