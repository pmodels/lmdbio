/* 
 * Python binding of LMDBIO using Pybind11.
 *
 * See https://github.com/pybind/pybind11 for more info of Pybind11.
 *
 */

#include "lmdbio.h"

namespace py = pybind11;

using namespace py::literals;

PYBIND11_MODULE(lmdbio, m) {
  /* lmdbio::record binding */
  py::class_<lmdbio::record>(m, "record", py::buffer_protocol())
    .def(py::init<>())
    .def("get_record_size", &lmdbio::record::get_record_size)
    .def("get_record", &lmdbio::record::py_get_record);

  /* enum bindings */
  py::enum_<lmdbio::dist_mode_enum>(m, "DIST_MODE")
    .value("SCATTERV", lmdbio::dist_mode_enum::SCATTERV)
    .value("SHMEM", lmdbio::dist_mode_enum::SHMEM)
    .export_values();

  py::enum_<lmdbio::read_mode_enum>(m, "READ_MODE")
    .value("STRIDE", lmdbio::read_mode_enum::STRIDE)
    .value("CONT", lmdbio::read_mode_enum::CONT)
    .export_values();

  py::enum_<lmdbio::prov_info_mode_enum>(m, "PROV_INFO_MODE")
    .value("ENABLE", lmdbio::prov_info_mode_enum::ENABLE)
    .value("DISABLE", lmdbio::prov_info_mode_enum::DISABLE)
    .export_values();

  /* lmdbio::prov_info_t binding */
  py::class_<lmdbio::prov_info_t>(m, "prov_info")
    .def(py::init<>())
    .def_readwrite("commit_iter", &lmdbio::prov_info_t::commit_iter)
    .def_readwrite("branch_num_keys", &lmdbio::prov_info_t::branch_num_keys)
    .def_readwrite("leaf_num_keys", &lmdbio::prov_info_t::leaf_num_keys)
    .def_readwrite("data_num_pages", &lmdbio::prov_info_t::data_num_pages)
    .def_readwrite("first_key", &lmdbio::prov_info_t::first_key)
    .def_readwrite("first_leaf_page_no",
        &lmdbio::prov_info_t::first_leaf_page_no)
    .def_readwrite("overflow", &lmdbio::prov_info_t::overflow)
    .def_readwrite("max_data_size", &lmdbio::prov_info_t::max_data_size);

  /* lmdbio::db binding */
  py::class_<lmdbio::db>(m, "db")
    .def(py::init<>())
    .def("init", &lmdbio::db::py_init, "fname"_a, "batch_size"_a,
        "reader_size"_a=0, "prefetch"_a=0, "max_iter"_a=1)
    .def("set_mode", &lmdbio::db::set_mode)
    .def("set_prov_info", &lmdbio::db::py_set_prov_info)
    .def("set_stagger_size", &lmdbio::db::set_stagger_size)
    .def("read_record_batch", &lmdbio::db::read_record_batch)
    .def("get_batch_size", &lmdbio::db::get_batch_size)
    .def("get_num_records", &lmdbio::db::get_num_records)
    .def("get_record", &lmdbio::db::py_get_record);
}

/* Python binding */
py::array_t<char> lmdbio::record::py_get_record() const {
  return py::array(py::buffer_info(
        data,
        sizeof(char),
        py::format_descriptor<char>::format(),
        1,
        { record_size },
        { sizeof(char) }
        ));
}

void lmdbio::db::py_init(const char *fname, int batch_size,
    int reader_size, int prefetch, int max_iter) {
  int init_flag = 0, comm_rank, comm_size;

  std::cout << ">>> lmdbio: py_init" << std::endl;

  MPI_Initialized(&init_flag);
  if (!init_flag) {
    MPI_Init(NULL, NULL);
    std::cout << "lmdbio: initialize MPI" << std::endl;
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  
  /* print param settings */
  std::cout << "lmdbio: file name: " << fname << std::endl;
  std::cout << "lmdbio: batch size: " << batch_size << std::endl;
  std::cout << "lmdbio: reader size: " << reader_size << std::endl;
  std::cout << "lmdbio: prefetch: " << prefetch << std::endl;
  std::cout << "lmdbio: max iter: " << max_iter << std::endl;
  std::cout << "lmdbio: num processes: " << comm_size << std::endl;
  std::cout << "lmdbio: rank: " << comm_rank << std::endl;
  init(MPI_COMM_WORLD, fname, batch_size, reader_size, prefetch, max_iter);
}

void lmdbio::db::py_set_prov_info(py::object prov_info) {
  auto prov_info_ = prov_info.cast<prov_info_t *>();

  std::cout << ">>> lmdbio: py_set_prov_info" << std::endl;

  set_prov_info(*prov_info_);
}

py::object lmdbio::db::py_get_record(int i) {
  std:: cout << ">>> lmdbio: py_get_record" << std::endl;

  return py::cast(&records[i]);
}
