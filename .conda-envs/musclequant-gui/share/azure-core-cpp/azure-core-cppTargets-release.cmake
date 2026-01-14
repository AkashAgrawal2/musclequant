#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Azure::azure-core" for configuration "Release"
set_property(TARGET Azure::azure-core APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(Azure::azure-core PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libazure-core.1.16.0.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libazure-core.1.16.0.dylib"
  )

list(APPEND _cmake_import_check_targets Azure::azure-core )
list(APPEND _cmake_import_check_files_for_Azure::azure-core "${_IMPORT_PREFIX}/lib/libazure-core.1.16.0.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
