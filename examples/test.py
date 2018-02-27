import lmdbio
import numpy

# init params
fname = "/lcrc/project/radix/pumma/dl_benchmark/data/cifar10_alexnet_data/cifar10_alexnet_train_lmdb"
batch_size = 64
max_iter = 10

print(">>> instantiate db")
db = lmdbio.db()

print(">>> db.set_mode(...)")
db.set_mode(lmdbio.DIST_MODE.SHMEM, lmdbio.READ_MODE.STRIDE,
    lmdbio.PROV_INFO_MODE.DISABLE)

print(">>> db.init(...)")
db.init(fname, batch_size, max_iter=max_iter)

print(">>> db.set_stagger_size(...)")
db.set_stagger_size(0)

print(">>> db.get_batch_size(): ", db.get_batch_size())

print(">>> db.get_num_records(): ", db.get_num_records())

for i in range(0, max_iter):
  print(">>> db.read_record_batch()")
  db.read_record_batch()

  for j in range(0, db.get_num_records()):
    print(">>> db.get_record(", j, ")")
    record = db.get_record(j)

    print(">>> db.get_record(", j, ").get_record_size(): ", record.get_record_size())

    print(">>> db.get_record(", j, ").get_record()")
    print(record.get_record())
