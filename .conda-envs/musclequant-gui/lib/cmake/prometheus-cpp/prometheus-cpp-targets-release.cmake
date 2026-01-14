#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "prometheus-cpp::core" for configuration "Release"
set_property(TARGET prometheus-cpp::core APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(prometheus-cpp::core PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libprometheus-cpp-core.1.2.4.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libprometheus-cpp-core.1.2.dylib"
  )

list(APPEND _cmake_import_check_targets prometheus-cpp::core )
list(APPEND _cmake_import_check_files_for_prometheus-cpp::core "${_IMPORT_PREFIX}/lib/libprometheus-cpp-core.1.2.4.dylib" )

# Import target "prometheus-cpp::pull" for configuration "Release"
set_property(TARGET prometheus-cpp::pull APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(prometheus-cpp::pull PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libprometheus-cpp-pull.1.2.4.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libprometheus-cpp-pull.1.2.dylib"
  )

list(APPEND _cmake_import_check_targets prometheus-cpp::pull )
list(APPEND _cmake_import_check_files_for_prometheus-cpp::pull "${_IMPORT_PREFIX}/lib/libprometheus-cpp-pull.1.2.4.dylib" )

# Import target "prometheus-cpp::push" for configuration "Release"
set_property(TARGET prometheus-cpp::push APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(prometheus-cpp::push PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libprometheus-cpp-push.1.2.4.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libprometheus-cpp-push.1.2.dylib"
  )

list(APPEND _cmake_import_check_targets prometheus-cpp::push )
list(APPEND _cmake_import_check_files_for_prometheus-cpp::push "${_IMPORT_PREFIX}/lib/libprometheus-cpp-push.1.2.4.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
