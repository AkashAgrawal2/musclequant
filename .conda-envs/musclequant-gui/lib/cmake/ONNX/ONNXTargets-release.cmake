#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "ONNX::onnx" for configuration "Release"
set_property(TARGET ONNX::onnx APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ONNX::onnx PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libonnx.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libonnx.dylib"
  )

list(APPEND _cmake_import_check_targets ONNX::onnx )
list(APPEND _cmake_import_check_files_for_ONNX::onnx "${_IMPORT_PREFIX}/lib/libonnx.dylib" )

# Import target "ONNX::onnx_proto" for configuration "Release"
set_property(TARGET ONNX::onnx_proto APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ONNX::onnx_proto PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "absl::base;absl::cord;absl::die_if_null;absl::hash;absl::log_initialize;absl::log_severity;absl::status;absl::statusor;absl::strings;absl::synchronization;absl::time"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libonnx_proto.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libonnx_proto.dylib"
  )

list(APPEND _cmake_import_check_targets ONNX::onnx_proto )
list(APPEND _cmake_import_check_files_for_ONNX::onnx_proto "${_IMPORT_PREFIX}/lib/libonnx_proto.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
