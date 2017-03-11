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
  // initialize class attributes
  current_batch = -1;
  has_diff_data_size = false;
  has_comm = true;
  rsize_len = 0;
  dist_mode = "alltoallv";

  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  this->batch_size = batch_size;
  this->sbatch_size = get_sbatch_size(rank);
  has_diff_batch_size = !is_identical_sbatch_size();
  if (dist_mode == "no_mpi") {
    this->readers = np;
    this->fetch_batch_num = 1;
  }
  else {
    compute_num_readers();
    // fix number of prefetch batch to 1
    this->fetch_batch_num = 1;
  }
  if (this->readers == np && this->fetch_batch_num == 1) {
    has_comm = false;
  }

  /*cout << "Readers: " << readers;
  cout << "Batch num: " << fetch_batch_num;
  cout << "Has diff batch size " << has_diff_batch_size;
  cout << "Myrank " << rank << " ParallelDataReader is created";*/

  this->total_images = this->max_iter * this->batch_size;
  compute_fetch_size(false);

  // open database
  if (is_reader(rank)) {
    this->batch_ptrs = (char**) malloc(sizeof(char*) * this->fetch_size);
    open_db();
    this->datum_byte_size = lmdb_value_size();
  }
  has_diff_batch_size = !is_identical_sbatch_size();
  if (dist_mode == "alltoallv") {
    sbatch_sendcounts = new int[np];
    sbatch_senddispls = new int[np];
    sbatch_recvcounts = new int[np];
    sbatch_recvdispls = new int[np];
  }
}

// replace this with the model
void lmdbio::db::compute_num_readers() {
  this->readers = 1;
}

void lmdbio::db::compute_fetch_size(bool invalid) {
  int fetch_images = this->batch_size * this->fetch_batch_num; 
  if (phase == TEST_PHASE) {
    this->total_images = this->max_iter * this->batch_size;
  }
  //cout << "Total images: " << this->total_images;
  if (this->total_images >= fetch_images) {
    //cout << "Fetch one or more batch";
    this->fetch_size = fetch_images / readers;
    is_full_batch = true;
  }
  else {
    //cout << "Fetch less than one batch";
    this->fetch_size = this->total_images / readers;
    this->fetch_batch_num = this->total_images / this->batch_size;
    is_full_batch = false;
  }
  fetch_images = this->fetch_size * readers; 
  //adjust_readers(fetch_images, invalid);
  this->fetch_size = fetch_images / readers;
  this->fetch_batch_num = fetch_images / this->batch_size;
  this->total_images -= fetch_images; 
  /*if (param_.phase() == caffe::TRAIN) {
    cout << "Rank " << rank << " New Readers: " << readers;
    cout << "Rank " << rank << " New Batch num: " << fetch_batch_num;
    cout << "Rank " << rank << " New Fetch size: " << this->fetch_size;
    cout << "Rank " << rank << " New Fetch images: " << fetch_images
      << " Remaining images: " << this->total_images;
  }*/
}

// open the database and initialize a position of a cursor
void lmdbio::db::open_db() {
  int flags = MDB_RDONLY | MDB_NOTLS | MDB_NOLOCK;
  check_lmdb(mdb_env_create(&mdb_env_), "Created environment", false);
  check_lmdb(mdb_env_set_maxreaders(mdb_env_, np), "Set maxreaders", false);
  check_lmdb(mdb_env_open(mdb_env_, fname, flags, 0664),
      "Opened environment", false);
  printf("Rank %d source %s\n", rank, fname);
  check_lmdb(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn),
      "Begun transaction", false);
  check_lmdb(mdb_dbi_open(mdb_txn, NULL, 0, &mdb_dbi_),
      "Opened database", false);
  check_lmdb(mdb_cursor_open(mdb_txn, mdb_dbi_, &mdb_cursor),
      "Opened cursor", false);
  //mdb_env_set_mapsize(mdb_env_, 1024 * 1024 * 1024);
  lmdb_init_cursor();
  this->cursor = cursor;
}

void lmdbio::db::rsize_alltoallv(
    int* s_rsize, int* r_rsize, int* sendcounts, int* senddispls, 
    int* recvcounts, int* recvdispls) {
  int senddispl = 0;
  int recvdispl = 0;
  int sendcount = this->rsize_len; 
  int receiver_size = 0;
  int send_batch_num = this->fetch_batch_num / this->readers;
  int soffset = 0;
  int roffset = 0;
  int count = 0;
  int offset = 0;

  // one batch is read by more than one processes
  if (readers > fetch_batch_num) {
    int reader_per_batch = batch_size / fetch_size;
    receiver_size = fetch_size / sbatch_size;
    //printf("Receiver size %d\n", receiver_size);
    send_batch_num = 1;
    for (int i = 0; i < np; i++) {
      sendcounts[i] = 0;
      recvcounts[i] = 0;
      if (is_reader(rank) && 
          recv_group(i, receiver_size) == read_group(rank, reader_per_batch)) {
        sendcounts[i] = sendcount;
      }
      if (is_reader(i) && 
          recv_group(rank, receiver_size) == read_group(i, reader_per_batch)) {
        recvcounts[i] = sendcount;
      }
      senddispls[i] = senddispl;
      recvdispls[i] = recvdispl;
      senddispl += sendcounts[i];
      recvdispl += recvcounts[i];
      /*printf("Myrank %d item %d sendcounts %d\n", rank, i, sendcounts[i]);
      printf("Myrank %d item %d senddispls %d\n", rank, i, senddispls[i]);
      printf("Myrank %d item %d recvcounts %d\n", rank, i, recvcounts[i]);
      printf("Myrank %d item %d recvdispls %d\n", rank, i, recvdispls[i]);*/
    }
  }
  // each reader reads at least one batch and sends sbatch to every process
  else {
    int offset = sendcount * (send_batch_num - 1);
    //printf("offset %d\n", offset);
    // for each batch
    receiver_size = np;
    for (int i = 0; i < np; i++) {
      sendcounts[i] = 0;
      recvcounts[i] = 0;
      if (is_reader(rank)) {
        sendcounts[i] = sendcount;
      }
      if (is_reader(i)) {
        recvcounts[i] = sendcount;
        if (i != 0)
          recvdispl += offset;
      }
      senddispls[i] = senddispl;
      recvdispls[i] = recvdispl;
      senddispl += sendcounts[i];
      recvdispl += recvcounts[i];
    }
  }
  //printf("Send batch num %d\n", send_batch_num);
  for (int i = 0; i < send_batch_num; i++) {
    soffset = i * sendcount * (np / readers);
    roffset = i * sendcount;
    //printf("soffset %d roffset %d\n", soffset, roffset);
    MPI_Alltoallv(s_rsize + soffset, sendcounts, senddispls, MPI_INT, 
        r_rsize + roffset, recvcounts, recvdispls, MPI_INT, MPI_COMM_WORLD);
  }
  int r_rsize_len = rsize_len * send_batch_num;
  /*for (int i = 0; i < r_rsize_len; i++) {
    printf("Rank %d Rrsize-alltoallv item %d = %d\n", rank,  i, r_rsize[i]);
  }*/
}


void lmdbio::db::send_batch(char* batch_bytes, char** sbatch_bytes, 
    int* s_rsize, int** r_rsize) {
  int count = this->sbatch_size * this->datum_byte_size;
  int* sop;
  int* rop;
  int send_batch_num = fetch_batch_num < readers ? 1 : fetch_batch_num / readers;
  int r_rsize_len = 0;
  int recv_size = 0;
  //cout << "Rank " << rank << " send batch";
  if (!has_diff_data_size) {
    MPI_Bcast(&this->datum_byte_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    //cout << "Rank " << rank << " bcast size " << this->datum_byte_size;
  }
  else {
    int r_rsize_len = rsize_len * send_batch_num;
    *r_rsize = (int*) malloc (r_rsize_len * sizeof(int));
    rsize_alltoallv(s_rsize, *r_rsize, sbatch_sendcounts, 
        sbatch_senddispls, sbatch_recvcounts, sbatch_recvdispls);
    /*for (int i = 0; i < r_rsize_len; i++) {
      printf("Rank %d Rrsize item %d = %d\n", rank,  i, (*r_rsize)[i]);
    }*/
  }

  // exchange batch between processes
  if (has_diff_data_size) {
    r_rsize_len = rsize_len * send_batch_num;
    for (int i = rsize_len - 1; i < r_rsize_len; i += rsize_len) {
      recv_size += (*r_rsize)[i];
    }
  }
  else {
    recv_size = (fetch_size * readers * this->datum_byte_size) / np;
  }
  *sbatch_bytes = (char*) malloc (recv_size * sizeof(char));

  if (dist_mode == "alltoallv") {
    dist_alltoallv(batch_bytes, *sbatch_bytes, 
        sbatch_sendcounts, sbatch_senddispls, 
        sbatch_recvcounts, sbatch_recvdispls, 
        s_rsize, *r_rsize);
  }
  else if (dist_mode == "scatter") {
    dist_scatter(batch_bytes, *sbatch_bytes);
  }

}

void lmdbio::db::read_batch() {
  int size = 0;
  int psize = 0;
  int rsize = 0;
  int s_psize = 0;
  int s_rsize = 0;
  int diff_count = 0;
  int max_diff_count = 0;
  int count = 0;
  has_diff_data_size = false;
  total_byte_size = 0;
  rsize_vec.clear();
  cout << "Rank " << rank << " Read batch" << endl;
  cout << "Rank " << rank << " Fetch size " << this->fetch_size << endl;
  for (int i = 0; i < this->fetch_size; i++) {
    // get pointers
    this->batch_ptrs[i] = (char*) lmdb_value_data();
    size = lmdb_value_size();
    total_byte_size += size;
    rsize = size - psize;
    s_rsize = size - s_psize;
    if (rsize != 0) {
      if (i != 0) {
        has_diff_data_size = true;
      }
      rsize_vec.push_back(i);
      rsize_vec.push_back(rsize);
    }
    if (s_rsize != 0) {
      diff_count += 2;
    }
    if (i / sbatch_size > count) {
      max_diff_count = std::max(max_diff_count, diff_count);
      diff_count = 0;
      s_psize = 0;
      count++;
    }
    psize = size;
    cout << "Rank " << rank << " read item " << i << " size " << size << endl;
    //this->cursor->Next();
    lmdb_next();
    //validate_cursor();
  }
  this->datum_byte_size = lmdb_value_size();
  max_diff_count = std::max(max_diff_count, diff_count);
  this->rsize_len = max_diff_count;
  //cout << "Rank " << rank << " max_diff_count " << max_diff_count; 
  // move a cursor to the next batch
  if (readers != 1) {
    //this->cursor->NextBatch(readers, this->fetch_size);
    lmdb_next_fetch();
  }
  //validate_cursor();
}

void lmdbio::db::check_diff_batch() {
  int sop[2];
  int rop[2];
  sop[0] = has_diff_data_size ? 1 : 0;
  sop[1] = rsize_len;
  MPI_Allreduce(sop, rop, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  has_diff_data_size = rop[0] == 1;
  rsize_len = rop[1] + 4;
  cout << "Rank " << rank << " check diff batch rsize_len " << rsize_len << endl;
}

void lmdbio::db::serialize_data(char** batch_bytes, int** s_rsize) {
  int offset = 0;
  int count = -1;
  int size = 0;
  int psize = 0;
  int rsize = 0;
  int tsize = 0;
  int r_idx = 0;
  int s_r_idx = 0;
  int sbatch_size = get_sbatch_size(0);
  int send_batch_num = fetch_batch_num < readers ? 1 : fetch_batch_num / readers;
  int s_rsize_len = (np / readers) * rsize_len * send_batch_num;
  int rsize_size = rsize_vec.size();
  int* s_rsize_;
  cout << "Rank " << rank << " Serialze data" << endl;
  cout << "Rank " << rank << " has diff data size " << has_diff_data_size << endl;
  cout << "Rank " << rank << " send_batch_num " << send_batch_num << endl;
  cout << "Rank " << rank << " s_rsize_len " << s_rsize_len << endl;
  cout << "Rank " << rank << " rsize_len " << rsize_len << endl;
  cout << "Rank " << rank << " sbatch_size " << sbatch_size << endl;

  *batch_bytes = (char*) malloc (sizeof(char) * total_byte_size);
  if (has_diff_data_size)
    s_rsize_ = (int*) malloc (s_rsize_len * sizeof(int));
  for (int i = 0; i < this->fetch_size; i++) {
    if (r_idx < rsize_size && rsize_vec[r_idx] == i) {
      size += rsize_vec[r_idx + 1];
      r_idx += 2;
    }
    cout << "Rank " << rank << " serialize item " << i << " size " << size << endl;
    if (has_diff_data_size) {
      rsize = size - psize;
      if (i / sbatch_size > count) {
        if (s_r_idx != 0) {
          s_rsize_[s_r_idx++] = -1;
          s_r_idx = rsize_len * (count + 1);
          s_rsize_[s_r_idx - 1] = tsize;
          tsize = 0;
        }
        s_rsize_[s_r_idx] = 0;
        s_rsize_[s_r_idx + 1] = size;
        s_r_idx += 2;
        count++;
      }
      else if (rsize != 0) {
        s_rsize_[s_r_idx] = i % sbatch_size;
        s_rsize_[s_r_idx + 1] = rsize;
        s_r_idx += 2;
      }
      psize = size;
      tsize += size;
    }
    //cout << "Rank " << rank << " serialize item " << i << " s_r_idx " << s_r_idx;
    memcpy(*batch_bytes + offset, this->batch_ptrs[i], size);
    offset += size;
    // compute displacements and counts
  }
  //cout << "Final s_r_idx " << s_r_idx;
  if (has_diff_data_size) {
    s_rsize_[s_r_idx] = -1;
    s_rsize_[s_rsize_len - 1] = tsize;
    *s_rsize = s_rsize_;
  }

  /*for (int i = 0; i < s_rsize_len; i++) {
    cout << "S_RSIZE item " << i << " " << s_rsize_[i]; 
  }*/
}

int lmdbio::db::get_sbatch_size(int sbatch_id) {
  int remainder = this->batch_size % np;
  return (this->batch_size / np) + (sbatch_id < remainder ? 1 : 0);
}

bool lmdbio::db::is_identical_sbatch_size() {
  return this->batch_size % np == 0;
}

bool lmdbio::db::is_reader(int proc) {
  return (proc % (np / readers)) == 0;
}

int lmdbio::db::recv_group(int proc, int receiver_size) {
  return proc / receiver_size;
}

int lmdbio::db::reader_id(int proc) {
  return proc / (np / readers);
}

int lmdbio::db::read_group(int proc, int reader_per_batch) {
  return reader_id(proc) % reader_per_batch;
}

void lmdbio::db::dist_scatter(char* sbuf, char* rbuf) {
  int sendcount = sbatch_size * this->datum_byte_size;
  //printf("Recvcount %d\n", recvcount);
  //printf("Sendcount %d\n", sendcount);
  int soffset;
  int roffset;
  int count = 0;
  //printf("Myrank %d read each data %d\n", rank, (fetch_batch_num / readers));
  for (int i = 0; i < np; i += (np / readers)) {
    //printf("Myrank %d Scatter from %d\n", rank, i );
    for (int j = 0; j < (fetch_batch_num / readers); j++) {
      soffset = j * batch_size * this->datum_byte_size; // each batch
      roffset = count * sendcount; // each sbatch
      /*printf("Myrank %d soffset %d\n", rank, soffset);
      printf("Myrank %d soffset %d\n", rank, soffset);
      printf("Myrank %d sendcount %d\n", rank, sendcount);*/
      MPI_Scatter(sbuf + soffset, sendcount, MPI_BYTE,
          rbuf + roffset, sendcount, MPI_BYTE, i, MPI_COMM_WORLD);
      count++;
    }
  }
}

void lmdbio::db::dist_alltoallv(char* sbuf, char* rbuf, 
    int* sendcounts, int* senddispls, int* recvcounts, int* recvdispls, 
    int* s_rsize, int* r_rsize) {
  int senddispl = 0;
  int recvdispl = 0;
  int sendcount = sbatch_size * this->datum_byte_size; 
  //cout << "Rank " << rank << " Datam byte size " << this->datum_byte_size;
  int receiver_size = 0;
  int send_batch_num = readers > fetch_batch_num ? 1 
    : fetch_batch_num / readers;
  int soffset = 0;
  int roffset = 0;
  int count = 0;
  int offset = 0;
  int s_rcount = rsize_len - 1;
  int r_rcount = rsize_len - 1;
  //printf("Rank %d dist alltoallv\n", rank);
  //printf("Send batch num %d\n", send_batch_num);
  soffset = 0;
  roffset = 0;
  for (int i = 0; i < send_batch_num; i++) {
    senddispl = 0;
    recvdispl = 0;
    // one batch is read by more than one processes
    if (readers > this->fetch_batch_num) {
      //printf("Rank %d one batch\n", rank);
      int reader_per_batch = batch_size / fetch_size;
      receiver_size = fetch_size / sbatch_size;
      //printf("Receiver size %d\n", receiver_size);
      for (int i = 0; i < np; i++) {
        sendcounts[i] = 0;
        recvcounts[i] = 0;
        if (is_reader(rank) && 
            recv_group(i, receiver_size) == read_group(rank, reader_per_batch)) {
          if (has_diff_data_size) {
            sendcount = s_rsize[s_rcount];
            s_rcount += rsize_len;
          }
          sendcounts[i] += sendcount;
        }
        if (is_reader(i) && 
            recv_group(rank, receiver_size) == read_group(i, reader_per_batch)) {
          if (has_diff_data_size) {
            sendcount = r_rsize[r_rcount];
            r_rcount += rsize_len;
          }
          recvcounts[i] += sendcount; 
        }
        senddispls[i] = senddispl;
        recvdispls[i] = recvdispl;
        senddispl += sendcounts[i];
        recvdispl += recvcounts[i];
        /*printf("Myrank %d item %d sendcounts %d\n", rank, i, sendcounts[i]);
        printf("Myrank %d item %d senddispls %d\n", rank, i, senddispls[i]);
        printf("Myrank %d item %d recvcounts %d\n", rank, i, recvcounts[i]);
        printf("Myrank %d item %d recvdispls %d\n", rank, i, recvdispls[i]);*/
      }
    }
    // each reader reads at least one batch and sends sbatch to every process
    else {
      //printf("Rank %d multiple batch\n", rank);
      int offset = sendcount * (send_batch_num - 1);
      //printf("offset %d\n", offset);
      // for each batch
      receiver_size = np;
      for (int i = 0; i < np; i++) {
        sendcounts[i] = 0;
        recvcounts[i] = 0;
        if (is_reader(rank)) {
          if (has_diff_data_size) {
            sendcount = s_rsize[s_rcount];
            s_rcount += rsize_len;
          }
          sendcounts[i] += sendcount;
        }
        if (is_reader(i)) {
          if (has_diff_data_size) {
            sendcount = r_rsize[r_rcount];
            r_rcount += rsize_len;
          }
          recvcounts[i] += sendcount;
          if (i != 0) {
            if (has_diff_data_size) {
              offset = 0;
              for (int j = 0; j < send_batch_num; j++) {
                offset += r_rsize[r_rcount];
                r_rcount += rsize_len;
              }
            }
            recvdispl += offset;
          }
        }
        senddispls[i] = senddispl;
        recvdispls[i] = recvdispl;
        senddispl += sendcounts[i];
        recvdispl += recvcounts[i];
        /*printf("Myrank %d item %d sendcounts %d\n", rank, i, sendcounts[i]);
        printf("Myrank %d item %d senddispls %d\n", rank, i, senddispls[i]);
        printf("Myrank %d item %d recvcounts %d\n", rank, i, recvcounts[i]);
        printf("Myrank %d item %d recvdispls %d\n", rank, i, recvdispls[i]);*/
      }
    }

    //printf("soffset %d roffset %d\n", soffset, roffset);
    MPI_Alltoallv(sbuf + soffset, sendcounts, senddispls, MPI_BYTE, 
        rbuf + roffset, recvcounts, recvdispls, MPI_BYTE, MPI_COMM_WORLD);
    if (has_diff_data_size) {
      soffset += senddispl;
      roffset += r_rsize[(i * rsize_len) + (rsize_len - 1)];
    }
    else {
      soffset = (i + 1) * batch_size * this->datum_byte_size;
      roffset = (i + 1) * sendcount;
    }
    //printf("Rank %d soffset %d\n", rank, soffset);
    //printf("Rank %d roffset %d\n", rank, roffset);
  }
}

void lmdbio::db::parse_sbatch_bytes(char* sbatch_bytes, int* r_rsize) {
  bool is_end = false;
  int count = 0;
  int r_idx = current_batch * rsize_len;
  int send_batch_num = fetch_batch_num < readers ? 1 
    : fetch_batch_num / readers;
  int size = has_diff_data_size ? 0 : this->datum_byte_size;
  cout << "Rank " << rank << " parse_sbatch_bytes" << endl;
  if (has_comm) {
    //int count_size = sbatch_recvcount * current_batch;
    int count_size = 0;
    int rsize_lim = rsize_len * (current_batch + 1);
    if (has_diff_data_size) {
      int idx = rsize_len - 1;
      for (int i = 0; i < current_batch; i++) {
        count_size += r_rsize[rsize_len - 1];
        idx += rsize_len;
      }
    }
    else {
      count_size = this->datum_byte_size * this->sbatch_size * current_batch;
    }


  }
  num_records = this->sbatch_size;
  records = (record *) operator new[] (num_records * sizeof(record));

  if (has_comm) {
    int count_size = 0;
    int rsize_lim = rsize_len * (current_batch + 1);
    if (has_diff_data_size) {
      int idx = rsize_len - 1;
      for (int i = 0; i < current_batch; i++) {
        count_size += r_rsize[rsize_len - 1];
        idx += rsize_len;
      }
    }
    else {
      count_size = this->datum_byte_size * this->sbatch_size * current_batch;
    }
    for (int i = 0; i < this->sbatch_size; i++) {
      if (has_diff_data_size) {
        if (!is_end && r_idx < rsize_lim
            && r_rsize[r_idx] == i % sbatch_size) {
          size += r_rsize[r_idx + 1];
          r_idx += 2;
          is_end = r_rsize[r_idx] == -1;
          //printf("count %d \n", count);
        }
      }
      records[i].set_record(sbatch_bytes + count_size, size);
      count_size += size;
    }
  }
  else {
    is_end = false;
    rsize_len = rsize_vec.size();
    for (int i = 0; i < this->sbatch_size; i++) {
      if (has_diff_data_size) {
        if (!is_end && r_idx < rsize_len
            && rsize_vec[r_idx] == i % sbatch_size) {
          size += rsize_vec[r_idx + 1];
          r_idx += 2;
          is_end = rsize_vec[r_idx] == -1;
          cout << "count " << count << endl;
        }
      }
      records[i].set_record(batch_ptrs[i], size);
    }
  }
}

int lmdbio::db::read_record_batch(void) 
{
  cout << "Parallel data reader max iter " << max_iter << endl;
  char* sbatch_bytes;
  char* batch_bytes;
  int* s_rsize;
  int* r_rsize;
  this->has_diff_data_size = false;
  this->rsize_len = 0;
  // each process reads one batch
  if (this->fetch_size != 0) {
    cout << "Current batch " << current_batch << endl;
    if (current_batch == -1 || ++current_batch >= this->fetch_batch_num) {
      // compute fetch size for the next data loading 
      current_batch = 0;
      if (is_reader(rank)) {
        read_batch();
      }
      cout << "Rank " << rank << " has communication " << has_comm << endl;
      if (has_comm) {
        check_diff_batch();
        if (is_reader(rank)) {
          serialize_data(&batch_bytes, &s_rsize);
        }
        // send buffer: batch_bytes, receive buffer: sbatch_bytes
        // send rsize: s_rsize, receive rsize: r_rsize
        send_batch(batch_bytes, &sbatch_bytes, s_rsize, &r_rsize);
        if (is_reader(rank)) {
          delete batch_bytes;
          if (has_diff_data_size)
            delete s_rsize;
        }
      }
      compute_fetch_size(true);
    }
    parse_sbatch_bytes(sbatch_bytes, r_rsize);
    cout << "Rank " << rank << " successfully parse batch" << endl;
    if (has_comm) {
      delete sbatch_bytes;
      if (has_diff_data_size)
        delete r_rsize;
    }
  }
  
  cout << "Rank " << rank << " reaches the last line of read_record_batch" << endl;
  return 0;
}
