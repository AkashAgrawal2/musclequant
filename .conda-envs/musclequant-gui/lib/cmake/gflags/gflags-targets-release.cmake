#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "gflags::gflags_shared" for configuration "Release"
set_property(TARGET gflags::gflags_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(gflags::gflags_shared PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libgflags.2.2.2.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libgflags.2.2.dylib"
  )

list(APPEND _cmake_import_check_targets gflags::gflags_shared )
list(APPEND _cmake_import_check_files_for_gflags::gflags_shared "${_IMPORT_PREFIX}/lib/libgflags.2.2.2.dylib" )

# Import target "gflags::gflags_nothreads_shared" for configuration "Release"
set_property(TARGET gflags::gflags_nothreads_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(gflags::gflags_nothreads_shared PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libgflags_nothreads.2.2.2.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libgflags_nothreads.2.2.dylib"
  )

list(APPEND _cmake_import_check_targets gflags::gflags_nothreads_shared )
list(APPEND _cmake_import_check_files_for_gflags::gflags_nothreads_shared "${_IMPORT_PREFIX}/lib/libgflags_nothreads.2.2.2.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
