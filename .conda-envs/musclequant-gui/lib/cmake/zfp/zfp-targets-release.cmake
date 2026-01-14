#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "zfp::zfp" for configuration "Release"
set_property(TARGET zfp::zfp APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(zfp::zfp PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libzfp.1.0.1.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libzfp.1.dylib"
  )

list(APPEND _cmake_import_check_targets zfp::zfp )
list(APPEND _cmake_import_check_files_for_zfp::zfp "${_IMPORT_PREFIX}/lib/libzfp.1.0.1.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
