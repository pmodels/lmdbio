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
#include <sys/mman.h>
#include <signal.h>
#include <math.h>
#include <climits>
#include <fcntl.h>

using std::cout;
using std::endl;
static int tmp = 0;
static uint64_t num_page_faults = 0;
char* lmdb_me_map;
char* lmdb_me_fmap;

#define GET_PAGE(x) ((char *) (((unsigned long long) (x)) & ~(PAGE_SIZE - 1)))
#define META_PAGE_NUM (2)
#define OPT_READ_CHUNK (4194304)

int sigsegv_handler(int dummy1, siginfo_t *__sig, void *dummy2)
{
  num_page_faults++;
  printf("lmdbio: got a SIGSEGV at %p, num_page_faults %llu\n",
      __sig->si_addr,
      num_page_faults);

  /* set the page to be accessible again */
  printf("lmdbio: unprotecting page %p, %d bytes\n",
      GET_PAGE(__sig->si_addr),
      getpagesize());
  if (GET_PAGE(__sig->si_addr) == __sig->si_addr)
    mprotect(GET_PAGE(__sig->si_addr), getpagesize(), PROT_READ);
  else
    mprotect(
        GET_PAGE(__sig->si_addr),
        getpagesize(),
        PROT_READ | PROT_EXEC);

  return 0;
}

int lmdb_fault_handler(int dummy1, siginfo_t *__sig, void *dummy2)
{
  char* fault_addr = GET_PAGE(__sig->si_addr);
  /*printf("lmdbio: FAULT at %p, for 4096 -------\n",
    fault_addr);*/
  size_t offset = (size_t) (fault_addr - lmdb_me_map);
  mprotect(fault_addr, getpagesize(), PROT_READ | PROT_WRITE);
  memcpy(fault_addr, lmdb_me_fmap + offset, getpagesize());
  return 0;
}

void lmdbio::db::init(
    MPI_Comm parent_comm,
    const char* fname,
    int batch_size,
    int reader_size,
    int batch_coalescing_size,
    int max_iter) {
#ifdef BENCHMARK
  double start, end;
  init_time.init_var_time = 0.0;
  init_time.init_db_time = 0.0;
  init_time.assign_readers_open_db_time = 0.0;
  init_time.assign_readers_manage_comms_time = 0.0;
  init_time.assign_readers_open_db_barrier_time = 0.0;
  init_time.assign_readers_after_opening_db_barrier_time = 0.0;
  init_time.assign_readers_create_buffs_time = 0.0;
  init_time.assign_readers_seq_seek_time = 0.0;
  init_time.assign_readers_adjust_ptrs_time = 0.0;
  init_time.seq_seek_compute_params_time = 0.0;
  init_time.seq_seek_create_datatype_time = 0.0;
  init_time.seq_seek_seq_read_time = 0.0;
  init_time.seq_seek_send_sizes_time = 0.0;
  init_time.seq_seek_send_batch_ptrs_time = 0.0;
  init_time.seq_seek_free_buffs_time = 0.0;
  iter_time.mpi_time = 0.0;
  iter_time.set_record_time = 0.0;
  iter_time.remap_time = 0.0;
  iter_time.mkstemp_time = 0.0;
  iter_time.unlink_time = 0.0;
  iter_time.write_time = 0.0;
  iter_time.mmap_time = 0.0;
  iter_time.close_time = 0.0;
  iter_time.unmap_time = 0.0;
  iter_time.load_meta_time = 0.0;
  iter_time.mprotect_time = 0.0;
  iter_time.prefetch_time = 0.0;
  iter_time.seek_time = 0.0;
  iter_time.mdb_seek_time = 0.0;
  iter_time.access_time = 0.0;
  iter_time.cursor_get_current_time = 0.0;
  iter_time.cursor_sz_recv_time = 0.0;
  iter_time.cursor_sz_send_time = 0.0;
  iter_time.cursor_recv_time = 0.0;
  iter_time.cursor_send_time = 0.0;
  iter_time.cursor_dsrl_time = 0.0;
  iter_time.cursor_srl_time = 0.0;
  iter_time.cursor_malloc_time = 0.0;
  iter_time.cursor_free_time = 0.0;
  iter_time.cursor_restoring_time = 0.0;
  iter_time.cursor_storing_time = 0.0;
  iter_time.barrier_time = 0.0;
  iter_time.mpi_io_time = 0.0;
  iter_time.compute_offset_time = 0.0;
  iter_time.adjust_offset_time = 0.0;
  iter_time.total_bytes_read = 0;
  iter_time.recv_notification_time = 0;
  iter_time.send_notification_time = 0;
  start = MPI_Wtime();
#endif

  //MPI_Comm_dup(parent_comm, &global_comm);

  /* initialize class attributes */
  global_np = 0; 
  global_rank = 0;
  iter = 0;
  prefetch_count = 0;
  num_read_batch = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &global_np);
  MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

  cout << "lmdbio: global rank " << global_rank << endl;
  cout << "lmdbio: global np " << global_np << endl;

  this->max_iter = max_iter;
  this->batch_size = batch_size;
  this->subbatch_size = batch_size / global_np;
  cout << "lmdbio: subbatch size " << this->subbatch_size << endl;
#ifdef BENCHMARK
  end = MPI_Wtime();
  init_time.init_var_time = get_elapsed_time(start, end);
  start = MPI_Wtime();
#endif

  this->reader_size = reader_size;
  assert(reader_size <= global_np);
  cout << "lmdbio: reader size is set to " << reader_size << endl;

  this->local_reader_size = 0;
  this->batch_coalescing_size = batch_coalescing_size;

  assign_readers(fname, batch_size);

#ifdef BENCHMARK
  end = MPI_Wtime();
  init_time.init_db_time = get_elapsed_time(start, end);
#endif
}

int lmdbio::db::round_up_power_of_two(int num) {
  int tmp = num - 1, count = 0;
  while (tmp > 0) {
    count++;
    tmp = tmp >> 1;
  }
  return 1 << count;
}

void lmdbio::db::init_read_params(int sample_size) {
  /* round sample size to a page unit */
  sample_size = (sample_size & (~(PAGE_SIZE - 1)))
    + (PAGE_SIZE * !!(sample_size % PAGE_SIZE));
  num_read_pages = (sample_size * fetch_size) / PAGE_SIZE;

  /* calculate batch_coalescing_size */
  batch_coalescing_size = batch_coalescing_size ? batch_coalescing_size :
    ceil((float) OPT_READ_CHUNK / (num_read_pages * getpagesize()));
  batch_coalescing_size = round_up_power_of_two(batch_coalescing_size);
  batch_coalescing_size = batch_coalescing_size > max_iter ? max_iter : batch_coalescing_size;
  assert(this->batch_coalescing_size);
  cout << "lmdbio: coalescing size " << this->batch_coalescing_size <<
    ", OPT_READ_CHUNK " << OPT_READ_CHUNK <<
    ", num_read_pages " << num_read_pages <<
    ", page size " << getpagesize() <<
    ", max coalescing size " << max_iter
    << endl;

  /* recalculate number of pages to read and fetch size */
  if (is_reader(local_rank)) {
    fetch_size *= batch_coalescing_size;
    cout << "lmdbio: fetch size: " << fetch_size << endl;
    printf("lmdbio: LMDB buffer address %p\n", lmdb_buffer);
    /* skip the first few pages as they are the meta pages */
    fflush(stdout);
  }
}

void lmdbio::db::lmdb_init_cursor() {
  int offset = 0;
  lmdb_seek_to_first();
#ifndef ICPADS
  /* shift the cursor */
  if (reader_id != 0)
    lmdb_seek_multiple(reader_id * fetch_size);
#endif
}

void lmdbio::db::lmdb_seq_seek() {
  MPI_Datatype size_type, batch_offset_type;
  MPI_Datatype size_vec_type, batch_offset_vec_type;
  int blockcount, blocklen, stride;
  int send_buff_size, recv_buff_size, single_fetch_size;
#ifdef BENCHMARK
  double start;
  start = MPI_Wtime();
#endif

  cout << "lmdbio: seq seek" << endl;

  single_fetch_size = fetch_size / batch_coalescing_size;
  blockcount = max_iter / batch_coalescing_size;
  blocklen = fetch_size;
  stride = blocklen * reader_size;
  recv_buff_size = single_fetch_size * max_iter;
  send_buff_size = single_fetch_size * reader_size * max_iter;

#ifdef BENCHMARK
  init_time.seq_seek_compute_params_time = get_elapsed_time(start, MPI_Wtime());
  start = MPI_Wtime();
#endif

  /* a derived datatype for sizes */
  MPI_Type_vector(blockcount, blocklen, stride, MPI_INT, &size_vec_type);
  MPI_Type_commit(&size_vec_type);
  MPI_Type_create_resized(
      size_vec_type,
      0,
      blocklen * sizeof(int),
      &size_type);
  MPI_Type_commit(&size_type);

  /* a derived datatype for batch ptrs */
  MPI_Type_vector(
      blockcount,
      blocklen * sizeof(MPI_Offset),
      stride * sizeof(MPI_Offset),
      MPI_BYTE,
      &batch_offset_vec_type);
  MPI_Type_commit(&batch_offset_vec_type);
  MPI_Type_create_resized(
      batch_offset_vec_type,
      0,
      blocklen * sizeof(MPI_Offset),
      &batch_offset_type);
  MPI_Type_commit(&batch_offset_type);

#ifdef BENCHMARK
  init_time.seq_seek_create_datatype_time =
    get_elapsed_time(start, MPI_Wtime());
  start = MPI_Wtime();
#endif

  /* seek through all the records */
  if (reader_id == 0) {
    assert(send_buff_size > 0);
    lmdb_seek_to_first();
    send_batch_offsets = new MPI_Offset[send_buff_size];
    send_sizes = new int[send_buff_size];
    for (int i = 0; i < send_buff_size; i++) {
      send_batch_offsets[i] = (char*) lmdb_value_data() - lmdb_buffer;
      send_sizes[i] = lmdb_value_size();
      /*printf("lmdbio: read item %d size %d at %p\n",
        i,
        send_sizes[i],
        send_batch_offsets[i]);*/
      lmdb_next();
    }
  }

#ifdef BENCHMARK
  init_time.seq_seek_seq_read_time = get_elapsed_time(start, MPI_Wtime());
  start = MPI_Wtime();
#endif

  /* distribute sizes */
  MPI_Scatter(
      send_sizes,
      1,
      size_type,
      sizes,
      recv_buff_size,
      MPI_INT,
      0,
      reader_comm);

#ifdef BENCHMARK
  init_time.seq_seek_send_sizes_time = get_elapsed_time(start, MPI_Wtime());
  start = MPI_Wtime();
#endif

  /* distribute batch ptrs */
  MPI_Scatter(
      send_batch_offsets,
      1,
      batch_offset_type,
      batch_offsets,
      recv_buff_size * sizeof(MPI_Offset),
      MPI_BYTE,
      0,
      reader_comm);

#ifdef BENCHMARK
  init_time.seq_seek_send_batch_ptrs_time
    = get_elapsed_time(start, MPI_Wtime());
  start = MPI_Wtime();
#endif

  /* free buffers and derived data types */
  if (reader_id == 0) {
    delete[] send_batch_offsets;
    delete[] send_sizes;
  }
  MPI_Type_free(&size_vec_type);
  MPI_Type_free(&size_type);
  MPI_Type_free(&batch_offset_vec_type);
  MPI_Type_free(&batch_offset_type);

#ifdef BENCHMARK
  init_time.seq_seek_free_buffs_time = get_elapsed_time(start, MPI_Wtime());
#endif
}

/* assign one reader per node */
void lmdbio::db::assign_readers(const char* fname, int batch_size) {
  int size = 0;
  string mmap_env;
  int is_rank_0 = 0;
  int is_reader_ = 0;
  int sublocal_id = 0;
  MPI_Aint size_win_size = 0;
  local_rank = 0;
  local_np = 0;
  reader_id = 0;
  fetch_size = 0;
#ifdef BENCHMARK
  double start;
  start = MPI_Wtime();
#endif

  /* communicator between processes within a node */
  local_comm = MPI_COMM_NULL;
  MPI_Comm_split_type(
      MPI_COMM_WORLD,
      MPI_COMM_TYPE_SHARED,
      global_rank,
      MPI_INFO_NULL,
      &local_comm);
  assert(local_comm != MPI_COMM_NULL);
  MPI_Comm_size(local_comm, &local_np);
  MPI_Comm_rank(local_comm, &local_rank);

  /* get number of nodes */
  is_rank_0 = local_rank == 0 ? 1 : 0;
  MPI_Allreduce(&is_rank_0, &node_size, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  /* get a number of readers */
  reader_size = reader_size && reader_size >= node_size ?
    reader_size : node_size;

  /* get a number of readers within a node */
  local_reader_size = reader_size / node_size;
  assert(local_reader_size);

  cout << "lmdbio: local reader size is " << local_reader_size << endl;

  is_single_reader_per_node = local_reader_size == 1;

  /* communicator between the reader and processes */
  if (!is_single_reader_per_node) {
    sublocal_id = local_rank / (local_np / local_reader_size);
    sublocal_comm = MPI_COMM_NULL;
    MPI_Comm_split(local_comm, sublocal_id, local_rank, &sublocal_comm);
    assert(sublocal_comm != MPI_COMM_NULL);
    MPI_Comm_size(sublocal_comm, &sublocal_np);
    MPI_Comm_rank(sublocal_comm, &sublocal_rank);
    //cout << "lmdbio: rank " << global_rank << " sublocal id " << sublocal_id << 
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
    cout << "lmdbio: number of readers " << reader_size << endl;
    cout << "lmdbio: rank " << global_rank <<
      " is a reader id " << reader_id <<
      " on host " << hostname << endl;
  }
#ifdef BENCHMARK
  init_time.assign_readers_manage_comms_time =
    get_elapsed_time(start, MPI_Wtime());
  start = MPI_Wtime();
#endif

  /* sync after openning the database */
  MPI_Barrier(MPI_COMM_WORLD);

#ifdef BENCHMARK
  init_time.assign_readers_open_db_barrier_time =
    get_elapsed_time(start, MPI_Wtime());
  start = MPI_Wtime();
#endif

  /* calculate fetch size */
  fetch_size = batch_size / reader_size;
  assert(fetch_size);

  /* open database file */
  if (is_reader(local_rank)) {
    open_db(fname);
  }

#ifdef BENCHMARK
  init_time.assign_readers_open_db_time =
    get_elapsed_time(start, MPI_Wtime());
  start = MPI_Wtime();
#endif

  MPI_Barrier(MPI_COMM_WORLD);

#ifdef BENCHMARK
  init_time.assign_readers_after_opening_db_barrier_time =
    get_elapsed_time(start, MPI_Wtime());
  start = MPI_Wtime();
#endif

  /* getting data record size */
  if (prov_info_mode == prov_info_mode_enum::ENABLE) {
    size = prov_info.max_data_size;
  }
  else {
    if (global_rank == 0) {
      /* get size */
      size = lmdb_value_size();
    }

    /* broadcast a size of data to allocate the shared buffer */
    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }

  /* init number of read pages, batch_coalescing_size, and fetch size */
  init_read_params(size);

  /* calculate win size - 2x larger than the estimated size */
  this->win_size =
    (MPI_Aint) subbatch_size *
    (MPI_Aint) batch_coalescing_size *
    (MPI_Aint) size *
    (MPI_Aint) 2 *
    (MPI_Aint) sizeof(char);
  assert(win_size > 0);

  if (prov_info_mode == prov_info_mode_enum::ENABLE) {
    size_win_size = subbatch_size * batch_coalescing_size;
  }
  else {
    size_win_size = subbatch_size * max_iter;
  }
  assert(size_win_size > 0);

  /* allocate neccessary buffer */
  if (dist_mode == dist_mode_enum::SHMEM) {
    MPI_Comm io_comm = get_io_comm();
    /* allocate a shared buffer for sizes */
    if (prov_info_mode != prov_info_mode_enum::ENABLE) {
      MPI_Win_allocate_shared(
          size_win_size * sizeof(int),
          sizeof(int),
          MPI_INFO_NULL,
          io_comm,
          &sizes,
          &size_win);
      MPI_Win_lock_all(MPI_MODE_NOCHECK, size_win);
    }
    /* allocate a shared offset buffer */
    MPI_Win_allocate_shared(
        size_win_size * sizeof(MPI_Offset),
        sizeof(MPI_Offset),
        MPI_INFO_NULL,
        io_comm,
        &batch_offsets,
        &batch_offset_win);
    MPI_Win_lock_all(MPI_MODE_NOCHECK, batch_offset_win);
    /* allocate a shared buffer for samples */
    MPI_Win_allocate_shared(
        win_size,
        sizeof(char),
        MPI_INFO_NULL,
        io_comm,
        &subbatch_bytes,
        &batch_win);
    MPI_Win_lock_all(MPI_MODE_NOCHECK, batch_win);
  }

  /* allocate record's array */
  this->records = new (std::nothrow) record[subbatch_size];
#ifdef BENCHMARK
  init_time.assign_readers_create_buffs_time =
    get_elapsed_time(start, MPI_Wtime());
  start = MPI_Wtime();
#endif

  /* get batch ptrs and sizes */
  if (is_reader() && prov_info_mode != prov_info_mode_enum::ENABLE)
    lmdb_seq_seek();

#ifdef BENCHMARK
  init_time.assign_readers_seq_seek_time = get_elapsed_time(start, MPI_Wtime());
  start = MPI_Wtime();
#endif

  cout << "lmdbio: rank " << local_rank << ", reset size ptrs" << endl;

  /* reset size ptr */
  if (prov_info_mode == prov_info_mode_enum::ENABLE) {
    batch_offsets_addr = batch_offsets;
  }
  else {
    sizes += ((subbatch_size * batch_coalescing_size) - size_win_size) * local_rank;
    batch_offsets += ((subbatch_size * batch_coalescing_size) - size_win_size) *
      local_rank;
  }
  /* reset subbatch bytes ptr to zero */
  subbatch_bytes -= (win_size / sizeof(char)) * local_rank;

#ifdef BENCHMARK
  init_time.assign_readers_adjust_ptrs_time =
    get_elapsed_time(start, MPI_Wtime());
#endif
}


/* open the database and initialize a position of a cursor */
void lmdbio::db::open_db(const char* fname) {
  int flags = MDB_RDONLY | MDB_NOTLS | MDB_NOLOCK;
  size_t addr;
  char addr_str[100]; // enough to hold all numbers up to 64-bits
  int max_try = 200;
  int rc = 0;
  int success = 0;
  struct sigaction __sig;
  MDB_envinfo stat;

#ifdef DIRECTIO
  if (global_rank == 0 && prov_info_mode != prov_info_mode_enum::ENABLE) {
#endif
    srand(time(NULL));
    check_lmdb(mdb_env_create(&mdb_env_), "Created environment", false);
    check_lmdb(
        mdb_env_set_maxreaders(mdb_env_, reader_size),
        "Set maxreaders",
        false);
#ifdef DIRECTIO
  }
#endif

  char *e = getenv("ENABLE_MPROTECT");
  if (e && !strcmp(e, "1")) {
    __sig.sa_sigaction =
      (void (*) (int, siginfo_t *, void *)) sigsegv_handler;
    __sig.sa_flags = SA_SIGINFO;
    sigaction(SIGSEGV, &__sig, 0);
  }

#ifdef ICPADS
  /* random an address and fix mmap's address */
  for (int i = 0; i < max_try; i++) {
    if (global_rank == 0) {
      addr = (size_t) getpagesize() * rand();
      snprintf(addr_str, 100, "MMAP_ADDRESS=%llu", addr);
      cout << addr_str << endl;
      putenv(addr_str);
      //cout << "lmdbio: mapping file " << fname << endl;
      rc = mdb_env_open(mdb_env_, fname, flags, 0664);
      //cout << "lmdbio: reader " << reader_id << " error code " << rc << endl;
      if (rc != 0) {
        cout << "lmdbio: mmap failed; trying again\n";
        continue;
      }
    }
    MPI_Bcast(addr_str, 100, MPI_CHAR, 0, reader_comm);
    if (global_rank != 0) {
      putenv(addr_str);
      cout << "lmdbio: trying to mmap with address " << addr_str << endl;
      rc = mdb_env_open(mdb_env_, fname, flags, 0664);
      //cout << "lmdbio: reader " << reader_id << " error code " << rc << endl;
    }
    MPI_Allreduce(&rc, &success, 1, MPI_INT, MPI_MAX, reader_comm);
    if (success == 0) {
      cout << "lmdbio: all processes got the same address" << addr <<
        endl;
      break;
    }
    if (rc == 0)
      mdb_env_close(mdb_env_);
    rc = success;
  }
  if (rc != 0) {
    cout << "lmdbio: cannot open db" << endl;
    exit(1);
  }
  /* set lmdb buffer */
  strtok(addr_str, "=");
  addr = atoll(strtok(NULL, "="));
  lmdb_buffer = (char*) addr;
  lmdb_me_map = lmdb_buffer;
#elif DIRECTIO
  if (global_rank == 0 && prov_info_mode != prov_info_mode_enum::ENABLE) {
    rc = mdb_env_open(mdb_env_, fname, flags, 0664);
    lmdb_buffer = mdb_get_me_map(mdb_env_);
  }
  cout << "lmdbio: DIRECT IO mode" << endl;

  char filename[1000];
  snprintf(filename, 1000, "%s/data.mdb", fname);

  fd = open(filename, O_RDONLY);
#else
  rc = mdb_env_open(mdb_env_, fname, flags, 0664);
  //cout << "lmdbio: reader " << reader_id << " error code " << rc << endl;
#endif

#ifdef DIRECTIO
  if (global_rank == 0 && prov_info_mode != prov_info_mode_enum::ENABLE) {
#endif
    check_lmdb(
        mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn),
        "Begun transaction",
        false);
    check_lmdb(
        mdb_dbi_open(mdb_txn, NULL, 0, &mdb_dbi_),
        "Opened database",
        false);
    check_lmdb(
        mdb_cursor_open(mdb_txn, mdb_dbi_, &mdb_cursor),
        "Opened cursor",
        false);
    lmdb_init_cursor();
#ifdef DIRECTIO
  }
#endif

  if (e && !strcmp(e, "1")) {
    /* protect the buffer against read accesses */
    mdb_env_info(mdb_env_, &stat);
    mprotect(lmdb_buffer, (size_t) stat.me_mapsize, PROT_NONE);
  }
}

/* compute data offsets based on provided provenance info */
void lmdbio::db::compute_data_offsets(
    long start_key,
    long end_key,
    off_t *start_offset,
    ssize_t *bytes) {

  assert(prev_key <= start_key && start_key <= end_key);

  long i = 0;
  int j = 0, k = 0;
  int counter_len = 0;
  int limit = 0;
  int commit_freenode_count = 0;
  int freelist_count = 0;
  int new_node_count = 0;
  int page_size = getpagesize();
  int page_header_size = mdb_get_pagehdrsz();
  bool started = false;
  MPI_Offset offset;

  for (i = prev_key; i <= end_key; i++) {
    data_start_page_no = data_end_page_no + 1;
    data_end_page_no = data_start_page_no + prov_info.data_num_pages;
    if (i == start_key) {
      *start_offset = data_start_page_no;
      k = 0;
      started = true;
    }

    /* store batch offset */
    if (started) {
      batch_offsets[k] = (data_start_page_no - *start_offset) * page_size;
      k++;
    }

    /* commit */
    if (prov_info.commit_iter && i % prov_info.commit_iter == 0 && i != 0) {
      /* compute number of free nodes to commit */
      if (txnid == 1) {
        commit_freenode_count = 0;
      }
      else if (txnid == 2) {
        commit_freenode_count = old_depth;
        freenode_count[0] = commit_freenode_count;
      }
      else {
        commit_freenode_count = old_depth + 1;
        freenode_count[1] = commit_freenode_count;
      }

      /* compute number of pages used to store free nodes info */
      if (i > prov_info.commit_iter)
        freelist_count =
          ceil((double) ((commit_freenode_count + 1) * sizeof(size_t)) /
              page_size);


      /* compute number of new added nodes based on the number of free nodes */
      new_node_count = depth;
      if (txnid > 2) {
        new_node_count = depth - freenode_count[0];
        new_node_count = new_node_count < 0 ? 0 : new_node_count;
      }


      /* append new pages to the end of the database file */
      data_end_page_no += new_node_count + freelist_count;

      txnid++;

      /* get number of remainning free nodes */
      txn_freenode_count = 0;
      if (txnid > 3) {
        txn_freenode_count = freenode_count[0] - depth;
        freenode_count[0] = freenode_count[1];
      }

      old_depth = depth;
    }

    /* add new nodes */
    counter_len = node_count.size();
    for (j = 0; j < counter_len; j++) {
      limit = j == 0 ? prov_info.leaf_num_keys : prov_info.branch_num_keys;
      node_count[j]++;
      if (node_count[j] > limit) {
        if (txn_freenode_count)
          txn_freenode_count--;
        else
          data_end_page_no++;
        node_count[j] = 1;
        if (j == counter_len - 1) {
          node_count.push_back(1);
          if (txn_freenode_count)
            txn_freenode_count--;
          else
            data_end_page_no++;
          depth++;
        }
      }
      else
        break;
    }
  }

  *bytes = (data_end_page_no - *start_offset + 1) * page_size;
  *bytes -= page_header_size;
  *start_offset *= page_size;
  *start_offset += page_header_size;
  prev_key = end_key + 1;
}

void lmdbio::db::read_all() {
  if (is_reader()) {
    int bytes, rc, len;
    ssize_t target_bytes, remaining, rs;
    char *buff, err[MPI_MAX_ERROR_STRING + 1];
    MPI_Status status;
    MPI_Offset *offsets;
    off_t start_offset, offset;
    int num_groups, group_no;
    const int group_size = this->stagger_size;
    int is_done = 0;
#ifdef BENCHMARK
    struct rusage rstart, rend;
    double ttime, utime, stime, sltime, start, end, start_, mpi_io_time;

    start_ = MPI_Wtime();
#endif
    /* wait for the previous group to be done */
    if (group_size && reader_size > group_size) {
      num_groups = reader_size / group_size;
      group_no = reader_id / group_size;
      cout << "lmdbio: reader " << reader_id <<
        ", group size " << group_size <<
        ", num group " << num_groups <<
        ", group no " << group_no << endl;
      if (group_no != 0) {
        cout << "lmdbio: reader " << reader_id <<
          ", wait for " << (reader_id - group_size) <<
          " to finish reading" << endl;
        MPI_Recv(
            &is_done,
            1,
            MPI_INT,
            reader_id - group_size,
            0,
            reader_comm,
            MPI_STATUS_IGNORE);
      }
    }
#ifdef BENCHMARK
    iter_time.recv_notification_time += get_elapsed_time(start_, MPI_Wtime());
    start = MPI_Wtime();
    getrusage(RUSAGE_SELF, &rstart);
    start_ = MPI_Wtime();
#endif
    len = 0;
    /* if provenance info is provided, calculate offsets on the fly;
     * else get offsets from the array */
    if (prov_info_mode == prov_info_mode_enum::ENABLE) {
      long start_key =
        (fetch_size * reader_size * (iter/ batch_coalescing_size)) +
        (fetch_size * reader_id);
      long end_key = start_key + fetch_size - 1; /* needed!! */
      compute_data_offsets(start_key, end_key, &start_offset, &target_bytes);
    }
    else {
      start_offset = batch_offsets[0];
      target_bytes =
        batch_offsets[fetch_size - 1] -
        start_offset +
        sizes[fetch_size - 1];
    }
    if (target_bytes <= 0) {
      cout << "lmdbio: invalid target bytes " << target_bytes <<
        endl;
    }
    assert(target_bytes > 0);
#ifdef BENCHMARK
    iter_time.total_bytes_read += target_bytes;
    iter_time.compute_offset_time += get_elapsed_time(start_, MPI_Wtime());
#endif
    if ((ssize_t) (win_size * local_np) < target_bytes) {
      cout << "lmdbio: rank " << reader_id <<
        ", bytes overflow, win size " << ((ssize_t) win_size * local_np) <<
        ", target bytes " << target_bytes << endl;
    }
    assert((ssize_t) (win_size * local_np) >= target_bytes);

    /* read data (single_fetch_size * batch_coalescing_size) from pointers */
    if (dist_mode == dist_mode_enum::SHMEM) {
#ifdef BENCHMARK
      start_ = MPI_Wtime();
#endif
      /* fixed target bytes */
      buff = subbatch_bytes;
      remaining = target_bytes;
      offset = start_offset;
      while (remaining) {
        rs = pread(fd, buff, remaining, offset);
        assert(rs > 0);
        offset += rs;
        buff += rs;
        remaining -= rs;
        assert(remaining >= 0);
      }
      assert(remaining == 0);
#ifdef BENCHMARK
      mpi_io_time = get_elapsed_time(start_, MPI_Wtime());
      iter_time.mpi_io_time += mpi_io_time;
      start_ = MPI_Wtime();
#endif
      if (prov_info_mode != prov_info_mode_enum::ENABLE) {
        start_offset = batch_offsets[0];
        for (int i = 0; i < fetch_size; i++) {
          batch_offsets[i] -= start_offset;
          /*cout << "lmdbio: item " << i <<
            ", subbatch offset " << batch_offsets[i] << endl;*/
        }
      }
    }

    cout << "lmdbio: done parsing batch" << endl;

#ifdef BENCHMARK
    iter_time.adjust_offset_time += get_elapsed_time(start_, MPI_Wtime());
    getrusage(RUSAGE_SELF, &rend);
    end = MPI_Wtime();
    ttime = get_elapsed_time(start, end);
    utime = get_utime(rstart, rend);
    stime = get_stime(rstart, rend);
    sltime = get_sltime(ttime, utime, stime);
    parse_stat.add_stat(
        get_ctx_switches(rstart, rend),
        get_inv_ctx_switches(rstart, rend),
        ttime,
        utime,
        stime,
        sltime);
    start_ = MPI_Wtime();
#endif
    /* notify the next group that the read is done */
    if (stagger_size && reader_size > group_size) {
      if (group_no != num_groups - 1) {
        is_done = 1;
        cout << "lmdbio: reader " << reader_id <<
          ", notify " << (reader_id + group_size) <<
          " that its read has finished reading" << endl;
        MPI_Send(
            &is_done,
            1,
            MPI_INT,
            reader_id + group_size,
            0,
            reader_comm);
      }
    }
#ifdef BENCHMARK
    iter_time.send_notification_time += get_elapsed_time(start_, MPI_Wtime());
#endif
  }
#ifdef BENCHMARK
  double start = MPI_Wtime();
#endif
  MPI_Barrier(MPI_COMM_WORLD);
#ifdef BENCHMARK
  iter_time.barrier_time += get_elapsed_time(start, MPI_Wtime());
  start = MPI_Wtime();
#endif
  /* local barrier */
  if (dist_mode == dist_mode_enum::SHMEM) {
    MPI_Win_sync(batch_win);
    if (prov_info_mode != prov_info_mode_enum::ENABLE)
      MPI_Win_sync(size_win);
    MPI_Win_sync(batch_offset_win);
    MPI_Barrier(get_io_comm());
    MPI_Win_sync(batch_win);
    if (prov_info_mode != prov_info_mode_enum::ENABLE)
      MPI_Win_sync(size_win);
    MPI_Win_sync(batch_offset_win);
  }
#ifdef BENCHMARK
  iter_time.local_barrier_time += get_elapsed_time(start, MPI_Wtime());
#endif
}

void lmdbio::db::send_batch() {
  int count = 0;
  int size = 0;
#ifdef BENCHMARK
  double start;
  start = MPI_Wtime();
#endif
  MPI_Comm comm = get_io_comm();
  //cout << "lmdbio: send batch " << send_sizes[0] << endl;
  /* send data size */
  MPI_Scatter(
      send_sizes,
      subbatch_size,
      MPI_INT,
      sizes,
      subbatch_size,
      MPI_INT,
      0,
      comm);

  //cout << "lmdbio: scatter size completed" << endl; 
  /* get total size */
  for (int i = 0; i < subbatch_size; i++) {
    count += sizes[i];
  }

  //cout << "lmdbio: count size completed" << endl;
  /* send sub-batch */
  MPI_Scatterv(
      batch_bytes,
      send_counts,
      send_displs,
      MPI_BYTE,
      subbatch_bytes,
      count,
      MPI_BYTE,
      0,
      comm);

  //cout << "lmdbio: scatterv batch completed" << endl;
#ifdef BENCHMARK
  iter_time.mpi_time += get_elapsed_time(start, MPI_Wtime());
#endif
}

/* set records */
void lmdbio::db::set_records() {
  int size;
  MPI_Offset offset;
#ifdef BENCHMARK
  double start = MPI_Wtime();
#endif
  for (int i = 0; i < subbatch_size; i++) {
    size = prov_info_mode == prov_info_mode_enum::ENABLE ?
      prov_info.max_data_size : sizes[i];
    records[i].set_record(subbatch_bytes + batch_offsets[i], size);
  }
#ifdef BENCHMARK
  iter_time.set_record_time += get_elapsed_time(start, MPI_Wtime());
#endif
}

void lmdbio::db::set_stagger_size(int stagger_size) {
  this->stagger_size = stagger_size;
}

void lmdbio::db::set_prov_info(prov_info_t prov_info) {
  this->prov_info.commit_iter = prov_info.commit_iter;
  this->prov_info.branch_num_keys = prov_info.branch_num_keys;
  this->prov_info.leaf_num_keys = prov_info.leaf_num_keys;
  this->prov_info.data_num_pages = prov_info.data_num_pages - 1;
  this->prov_info.first_key = prov_info.first_key;
  this->prov_info.first_leaf_page_no = prov_info.first_leaf_page_no;
  this->prov_info.overflow = prov_info.overflow;
  this->prov_info.max_data_size = prov_info.max_data_size;

  /* init prov params */
  prev_key = prov_info.first_key;
  data_start_page_no = prov_info.first_leaf_page_no;
  data_end_page_no = data_start_page_no;
  depth = old_depth = 1;
  txnid = 1;
  txn_freenode_count = 0;
  node_count.push_back(0);
}

void lmdbio::db::set_mode(
    lmdbio::dist_mode_enum dist_mode,
    lmdbio::read_mode_enum read_mode,
    lmdbio::prov_info_mode_enum prov_info_mode) {
  if (dist_mode == dist_mode_enum::SHMEM)
    cout << "lmdbio: set dist mode to SHMEM" << endl;
  if (read_mode == read_mode_enum::STRIDE)
    cout << "lmdbio: set read mode to STRIDE" << endl;
  else if (read_mode == read_mode_enum::CONT)
    cout << "lmdbio: set read mode to CONT" << endl;
  if (prov_info_mode == prov_info_mode_enum::ENABLE)
    cout << "lmdbio: set prov info to MODE_PROV_INFO_ENABLE" << endl;
  else if (prov_info_mode == prov_info_mode_enum::DISABLE)
    cout << "lmdbio: set prov info to MODE_PROV_INFO_DISABLED" << endl;

  this->dist_mode = dist_mode;
  this->read_mode = read_mode;
  this->prov_info_mode = prov_info_mode;
}

void lmdbio::db::lmdb_touch_pages() {
  /*cout << "lmdbio: touching data from page " << start_pg <<
    " to " << (start_pg + read_pages) << endl;*/
  for (size_t i = start_pg; i < start_pg + read_pages; i++) {
    //cout << "lmdbio: touching page " << i << endl;
    //for (size_t j = 0; j < PAGE_SIZE; j++)
    tmp += lmdb_buffer[PAGE_SIZE * i];
  }
  //cout << "lmdbio: done touching pages" << endl;
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

void lmdbio::db::update_buffer_offsets() {
  /* update size offset */
  if (prov_info_mode == prov_info_mode_enum::ENABLE) {
    if (is_read_iter())
      batch_offsets = batch_offsets_addr;
    else
      batch_offsets += subbatch_size;
  }
  else {
    int size_offset = (iter == 0) ?
      0 : (batch_coalescing_size == 1 || is_read_iter() ?
          subbatch_size * ((batch_coalescing_size * (local_np - 1)) + 1) :
          subbatch_size);
    sizes += size_offset;
    batch_offsets += size_offset;
  }
}

/* read one batch of data and set records */
void lmdbio::db::read_record_batch(void) {
  update_buffer_offsets();
  if (is_read_iter())
    read_all();
  set_records();
  iter++;
}

bool lmdbio::db::is_read_iter() {
  return iter % batch_coalescing_size == 0;
}

/* read one or more batches and return number of data records read */
const int lmdbio::db::read_bulk(
    int bulk_read_num_batches,
    const int** bulk_sizes,
    const void** bulk_bytes,
    const long long int** bulk_offsets) {
  update_buffer_offsets();
  *bulk_sizes = sizes;
  *bulk_bytes = (void*)(subbatch_bytes);
  *bulk_offsets = batch_offsets;
  if (!num_read_batch) {
    read_all();
    num_read_batch = batch_coalescing_size;
  }
  num_read_batch -= bulk_read_num_batches;
  iter += bulk_read_num_batches;
  return bulk_read_num_batches * subbatch_size;
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


int lmdbio::db::get_io_np() {
  return is_single_reader_per_node ? local_np : sublocal_np;
}
#endif

MPI_Comm lmdbio::db::get_io_comm() {
  return is_single_reader_per_node ? local_comm : sublocal_comm;
}

