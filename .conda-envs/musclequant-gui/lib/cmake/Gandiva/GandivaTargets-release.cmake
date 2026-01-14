#----------------------------------------------------------------
# Generated CMake target import file for configuration "RELEASE".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Gandiva::gandiva_shared" for configuration "RELEASE"
set_property(TARGET Gandiva::gandiva_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(Gandiva::gandiva_shared PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "re2::re2"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libgandiva.1500.2.0.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libgandiva.1500.dylib"
  )

list(APPEND _cmake_import_check_targets Gandiva::gandiva_shared )
list(APPEND _cmake_import_check_files_for_Gandiva::gandiva_shared "${_IMPORT_PREFIX}/lib/libgandiva.1500.2.0.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
