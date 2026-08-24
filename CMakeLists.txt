cmake_minimum_required(VERSION 3.18)
project(MedSOM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# The public reproduction build supports only the self-contained .sbcsr corpus
# path (load_sbcsr in kernel.cu). The legacy .bin/MeSH loader — the only consumer
# of Boost::serialization and pugixml — has been removed, so the build is Boost-free.
add_executable(medsom
    Experiment/kernel.cu
)

target_include_directories(medsom PRIVATE Experiment)
target_link_libraries(medsom PRIVATE cusparse)
target_compile_options(medsom PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
)
