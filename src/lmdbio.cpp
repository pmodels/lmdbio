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
#include <signal.h>

using std::cout;
using std::endl;

static uint64_t num_page_faults = 0;

#define GET_PAGE(x) ((char *) (((unsigned long long) (x)) & ~(getpagesize() - 1)))

int sigsegv_handler(int dummy1, siginfo_t *__sig, void *dummy2)
{
  num_page_faults++;
  printf("got a SIGSEGV at %p, num_page_faults %llu\n", __sig->si_addr,
      num_page_faults);
  /* set the page to be accessible again */
  printf("unprotecting page %p, %d bytes\n", GET_PAGE(__sig->si_addr), getpagesize());
  if (GET_PAGE(__sig->si_addr) == __sig->si_addr)
    mprotect(GET_PAGE(__sig->si_addr), getpagesize(), PROT_READ);
  else
    mprotect(GET_PAGE(__sig->si_addr), getpagesize(), PROT_READ | PROT_EXEC);

  return 0;
}

void lmdbio::db::init(MPI_Comm parent_comm, const char* fname, int batch_size,
    int reader_size) {
#ifdef BENCHMARK
  double start;
  init_time.init_var_time = 0.0;
  init_time.init_db_time = 0.0;
  init_time.assign_readers_open_db_time = 0.0;
  init_time.assign_readers_manage_comms_time = 0.0;
  init_time.assign_readers_open_db_barrier_time = 0.0;
  init_time.assign_readers_after_opening_db_barrier_time = 0.0;
  init_time.assign_readers_create_buffs_time = 0.0;
  iter_time.mpi_time = 0.0;
  iter_time.set_record_time = 0.0;
  iter_time.mdb_seek_time = 0.0;
  iter_time.access_time = 0.0;
  iter_time.local_barrier_time = 0.0;
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
  init_time.init_var_time = get_elapsed_time(start, MPI_Wtime());
  start = MPI_Wtime();
#endif

  this->reader_size = reader_size;
  assert(reader_size <= global_np);
  cout << "Reader size is set to " << reader_size << endl;

  this->local_reader_size = 0;

  assign_readers(fname, batch_size);

#ifdef BENCHMARK
  init_time.init_db_time = get_elapsed_time(start, MPI_Wtime());
#endif
}

/* assign one reader per node */
void lmdbio::db::assign_readers(const char* fname, int batch_size) {
  int size = 0;
  int is_rank_0 = 0;
  int is_reader_ = 0;
  int sublocal_id = 0;
#ifdef BENCHMARK
  double start = MPI_Wtime();
#endif
  local_rank = 0;
  local_np = 0;
  reader_id = 0;
  fetch_size = 0;

  /* communicator between processes within a node */
  local_comm = MPI_COMM_NULL;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, global_rank,
      MPI_INFO_NULL, &local_comm);
  assert(local_comm != MPI_COMM_NULL);
  MPI_Comm_size(local_comm, &local_np);
  MPI_Comm_rank(local_comm, &local_rank);

  /* get number of nodes */
  is_rank_0 = local_rank == 0 ? 1 : 0;
  MPI_Allreduce(&is_rank_0, &node_size, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  /* get a number of readers */
  reader_size = reader_size && reader_size >= node_size ? reader_size : node_size;

  /* get a number of readers within a node */
  local_reader_size = reader_size / node_size;
  assert(local_reader_size);

  //cout << "Local reader size is " << local_reader_size << endl;

  is_single_reader_per_node = local_reader_size == 1;

  /* communicator between the reader and processes */
  if (!is_single_reader_per_node) {
    sublocal_id = local_rank / (local_np / local_reader_size);
    sublocal_comm = MPI_COMM_NULL;
    MPI_Comm_split(local_comm, sublocal_id, local_rank, &sublocal_comm);
    assert(sublocal_comm != MPI_COMM_NULL);
    MPI_Comm_size(sublocal_comm, &sublocal_np);
    MPI_Comm_rank(sublocal_comm, &sublocal_rank);
    //cout << "Rank " << global_rank << " sublocal id " << sublocal_id << 
    //  " sublocal rank " << sublocal_rank << endl;
  }

  /* get number of readers */
  is_reader_ = is_reader() ? 1 : MPI_UNDEFINED;

  /* communicator between readers */
  reader_comm = MPI_COMM_NULL;
  MPI_Comm_split(MPI_COMM_WORLD, is_reader_, global_rank, &reader_comm);
  if (is_reader()) {
    assert(reader_comm != MPI_COMM_NULL);
    MPI_Comm_size(reader_comm, &reader_size);
    MPI_Comm_rank(reader_comm, &reader_id);
  }

  /* open database and set fetch size */
  if (is_reader(local_rank)) {
    char hostname[256];
    gethostname(hostname, 255);
    cout << "Number of readers " << reader_size << endl;
    cout << "Rank " << global_rank << " is a reader id " << reader_id << 
      " on host " << hostname << endl;

    fetch_size = batch_size / reader_size;
    assert(fetch_size);
  }
#ifdef BENCHMARK
  init_time.assign_readers_manage_comms_time = get_elapsed_time(start, MPI_Wtime());
  start = MPI_Wtime();
#endif

  MPI_Barrier(MPI_COMM_WORLD);

#ifdef BENCHMARK
  init_time.assign_readers_open_db_barrier_time = get_elapsed_time(start, MPI_Wtime());
  start = MPI_Wtime();
#endif
  if (is_reader(local_rank)) {
    open_db(fname);
  }
#ifdef BENCHMARK
  init_time.assign_readers_open_db_time = get_elapsed_time(start, MPI_Wtime());
  start = MPI_Wtime();
#endif

  MPI_Barrier(MPI_COMM_WORLD);

#ifdef BENCHMARK
  init_time.assign_readers_after_opening_db_barrier_time = get_elapsed_time(start, MPI_Wtime());
  start = MPI_Wtime();
#endif

  if (global_rank == 0) {
    /* get size */
    size = lmdb_value_size();
  }

  /* broadcast a size of data to allocate the shared buffer */
  MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  cout << "BCAST SIZE " << size << endl;

  this->win_size = subbatch_size * size * 2;
  cout << "WIN SIZE " << win_size << endl;

  /* allocate neccessary buffer */
  cout << "FETCH SIZE " << fetch_size << endl;
  if (is_reader(local_rank))
    batch_ptrs = new char*[fetch_size];
  if (dist_mode == MODE_SCATTERV) {
    if (is_reader(local_rank)) {
      int io_np = get_io_np();
      send_sizes = new int[fetch_size];
      send_displs = new int[io_np];
      send_counts = new int[io_np];
      batch_bytes = (char*) malloc(fetch_size * size * 2 * sizeof(char));
    }
    sizes = new int[subbatch_size];
    subbatch_bytes = (char*) malloc(win_size);
  }
  else if (dist_mode == MODE_SHMEM) {
    MPI_Comm io_comm = get_io_comm();
    MPI_Win_allocate_shared(subbatch_size * sizeof(int), sizeof(int),
        MPI_INFO_NULL, io_comm, &sizes, &size_win);
    MPI_Win_lock_all(MPI_MODE_NOCHECK, size_win);
    MPI_Win_allocate_shared(win_size, sizeof(char), MPI_INFO_NULL, io_comm,
        &subbatch_bytes, &batch_win);
    MPI_Win_lock_all(MPI_MODE_NOCHECK, batch_win);
    //printf("shared memory buffer is %p\n", subbatch_bytes);
  }

  /* allocate record's array */
  this->records = new (std::nothrow) record[subbatch_size];
#ifdef BENCHMARK
  init_time.assign_readers_create_buffs_time = get_elapsed_time(start, MPI_Wtime());
  start = MPI_Wtime();
#endif
}

/* open the database and initialize a position of a cursor */
void lmdbio::db::open_db(const char* fname) {
  int flags = MDB_RDONLY | MDB_NOTLS | MDB_NOLOCK;
  struct sigaction __sig;
  char addr_str[100];

  srand(time(NULL));

  char *e = getenv("ENABLE_MPROTECT");
  if (e && !strcmp(e, "1")) {
    __sig.sa_sigaction = (void (*) (int, siginfo_t *, void *))
      sigsegv_handler;
    __sig.sa_flags = SA_SIGINFO;
    sigaction(SIGSEGV, &__sig, 0);
  }

  mmap_addr = (size_t) getpagesize() * rand();
  snprintf(addr_str, 100, "MMAP_ADDRESS=%llu", mmap_addr);
  cout << addr_str << endl;
  putenv(addr_str);

  check_lmdb(mdb_env_create(&mdb_env_), "Created environment", false);
  check_lmdb(mdb_env_set_maxreaders(mdb_env_, reader_size), "Set maxreaders",
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
  int *read_sizes;
  int size = 0;
  int id = 0;
  int count = 0;
#ifdef BENCHMARK
  struct rusage rstart, rend;
  double ttime, utime, stime, sltime, start, end, start_;

  //MPI_Barrier(reader_comm);
  start = MPI_Wtime();
  getrusage(RUSAGE_SELF, &rstart);
  start_ = MPI_Wtime();
#endif
  //cout << "read batch start\n"; 
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
    for (int i = 0; i < fetch_size; i++) {
      size = lmdb_value_size();
      batch_ptrs[i] = (char*) lmdb_value_data();
      read_sizes[i] = size;
      total_byte_size += size;
      //cout << "reader " << reader_id << ": " << key() << endl;
      lmdb_next();
    }
  }

#ifdef BENCHMARK
  iter_time.access_time += get_elapsed_time(start_, MPI_Wtime());
  start_ = MPI_Wtime();
#endif

  /* determine if the data is larger than a buffer */
  assert(total_byte_size <= win_size * get_io_np());

  /* move a cursor to the next location */
  if (read_mode == MODE_STRIDE && global_np != 1)
    lmdb_next_fetch();

#ifdef BENCHMARK
  iter_time.mdb_seek_time += get_elapsed_time(start_, MPI_Wtime());
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
      if (i % subbatch_size == 0) {
        count = (i / subbatch_size) * win_size;
      }
      memcpy(subbatch_bytes + count, batch_ptrs[i], size);
      //printf("copying image[%d] to %p of size %d\n", i, subbatch_bytes + count, size);
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
  //cout << "read batch end\n"; 
}

void lmdbio::db::send_batch() {
  int count = 0;
  int size = 0;
#ifdef BENCHMARK
  double start;
  start = MPI_Wtime();
#endif
  MPI_Comm comm = get_io_comm();
  //cout << "send batch " << send_sizes[0] << endl;
  /* send data size */
  MPI_Scatter(send_sizes, subbatch_size, MPI_INT, sizes, subbatch_size, MPI_INT,
      0, comm);

  //cout << "scatter size completed" << endl; 
  /* get total size */
  for (int i = 0; i < subbatch_size; i++) {
    count += sizes[i];
  }

  //cout << "count size completed" << endl;
  /* send sub-batch */
  MPI_Scatterv(batch_bytes, send_counts, send_displs, MPI_BYTE,
      subbatch_bytes, count, MPI_BYTE, 0, comm);

  //cout << "scatterv batch completed" << endl;
#ifdef BENCHMARK
  iter_time.mpi_time += get_elapsed_time(start, MPI_Wtime());
#endif
}

/* set records */
void lmdbio::db::set_records() {
  int count = 0;
  int size = 0;
#ifdef BENCHMARK
  double start = MPI_Wtime();
#endif
  //cout << "set records start\n";
  if (dist_mode == MODE_SHMEM) {
    MPI_Win_sync(batch_win);
    MPI_Win_sync(size_win);

    MPI_Barrier(get_io_comm());

    MPI_Win_sync(batch_win);
    MPI_Win_sync(size_win);
  }
#ifdef BENCHMARK
  iter_time.local_barrier_time += get_elapsed_time(start, MPI_Wtime());
  start = MPI_Wtime();
#endif
  for (int i = 0; i < subbatch_size; i++) {
    //cout << "item " << i << endl;
    size = sizes[i];
    records[i].set_record(subbatch_bytes + count, size);
    //printf("set record %p, data %p, size %d\n", &records[i], records[i].data, size);
    count += size;
  }
#ifdef BENCHMARK
  iter_time.set_record_time += get_elapsed_time(start, MPI_Wtime());
#endif
  //cout << "set records end\n";
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

/* a process with local rank = 0 is a reader  */
bool lmdbio::db::is_reader(int local_rank) {
  if (is_single_reader_per_node)
    return local_rank == 0;
  else
    return sublocal_rank == 0;
}

bool lmdbio::db::is_reader() {
  return is_reader(local_rank);
}

int lmdbio::db::read_record_batch(void) 
{
  //MPI_Barrier(get_io_comm());
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

lmdbio::init_time_t lmdbio::db::get_init_time() {
  return init_time;
}

lmdbio::iter_time_t lmdbio::db::get_iter_time() {
  return iter_time;
}

lmdbio::io_stat lmdbio::db::get_read_stat() {
  return read_stat;
}

lmdbio::io_stat lmdbio::db::get_parse_stat() {
  return parse_stat;
}

MPI_Comm lmdbio::db::get_io_comm() {
  return is_single_reader_per_node ? local_comm : sublocal_comm;
}

int lmdbio::db::get_io_np() {
  return is_single_reader_per_node ? local_np : sublocal_np;
}


#endif
