#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "opentelemetry-cpp::proto" for configuration "Release"
set_property(TARGET opentelemetry-cpp::proto APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opentelemetry-cpp::proto PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libopentelemetry_proto.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libopentelemetry_proto.dylib"
  )

list(APPEND _cmake_import_check_targets opentelemetry-cpp::proto )
list(APPEND _cmake_import_check_files_for_opentelemetry-cpp::proto "${_IMPORT_PREFIX}/lib/libopentelemetry_proto.dylib" )

# Import target "opentelemetry-cpp::proto_grpc" for configuration "Release"
set_property(TARGET opentelemetry-cpp::proto_grpc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opentelemetry-cpp::proto_grpc PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "absl::synchronization"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libopentelemetry_proto_grpc.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libopentelemetry_proto_grpc.dylib"
  )

list(APPEND _cmake_import_check_targets opentelemetry-cpp::proto_grpc )
list(APPEND _cmake_import_check_files_for_opentelemetry-cpp::proto_grpc "${_IMPORT_PREFIX}/lib/libopentelemetry_proto_grpc.dylib" )

# Import target "opentelemetry-cpp::common" for configuration "Release"
set_property(TARGET opentelemetry-cpp::common APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opentelemetry-cpp::common PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libopentelemetry_common.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libopentelemetry_common.dylib"
  )

list(APPEND _cmake_import_check_targets opentelemetry-cpp::common )
list(APPEND _cmake_import_check_files_for_opentelemetry-cpp::common "${_IMPORT_PREFIX}/lib/libopentelemetry_common.dylib" )

# Import target "opentelemetry-cpp::trace" for configuration "Release"
set_property(TARGET opentelemetry-cpp::trace APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opentelemetry-cpp::trace PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libopentelemetry_trace.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libopentelemetry_trace.dylib"
  )

list(APPEND _cmake_import_check_targets opentelemetry-cpp::trace )
list(APPEND _cmake_import_check_files_for_opentelemetry-cpp::trace "${_IMPORT_PREFIX}/lib/libopentelemetry_trace.dylib" )

# Import target "opentelemetry-cpp::metrics" for configuration "Release"
set_property(TARGET opentelemetry-cpp::metrics APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opentelemetry-cpp::metrics PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libopentelemetry_metrics.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libopentelemetry_metrics.dylib"
  )

list(APPEND _cmake_import_check_targets opentelemetry-cpp::metrics )
list(APPEND _cmake_import_check_files_for_opentelemetry-cpp::metrics "${_IMPORT_PREFIX}/lib/libopentelemetry_metrics.dylib" )

# Import target "opentelemetry-cpp::logs" for configuration "Release"
set_property(TARGET opentelemetry-cpp::logs APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opentelemetry-cpp::logs PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libopentelemetry_logs.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libopentelemetry_logs.dylib"
  )

list(APPEND _cmake_import_check_targets opentelemetry-cpp::logs )
list(APPEND _cmake_import_check_files_for_opentelemetry-cpp::logs "${_IMPORT_PREFIX}/lib/libopentelemetry_logs.dylib" )

# Import target "opentelemetry-cpp::version" for configuration "Release"
set_property(TARGET opentelemetry-cpp::version APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opentelemetry-cpp::version PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libopentelemetry_version.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libopentelemetry_version.dylib"
  )

list(APPEND _cmake_import_check_targets opentelemetry-cpp::version )
list(APPEND _cmake_import_check_files_for_opentelemetry-cpp::version "${_IMPORT_PREFIX}/lib/libopentelemetry_version.dylib" )

# Import target "opentelemetry-cpp::resources" for configuration "Release"
set_property(TARGET opentelemetry-cpp::resources APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opentelemetry-cpp::resources PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libopentelemetry_resources.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libopentelemetry_resources.dylib"
  )

list(APPEND _cmake_import_check_targets opentelemetry-cpp::resources )
list(APPEND _cmake_import_check_files_for_opentelemetry-cpp::resources "${_IMPORT_PREFIX}/lib/libopentelemetry_resources.dylib" )

# Import target "opentelemetry-cpp::http_client_curl" for configuration "Release"
set_property(TARGET opentelemetry-cpp::http_client_curl APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opentelemetry-cpp::http_client_curl PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libopentelemetry_http_client_curl.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libopentelemetry_http_client_curl.dylib"
  )

list(APPEND _cmake_import_check_targets opentelemetry-cpp::http_client_curl )
list(APPEND _cmake_import_check_files_for_opentelemetry-cpp::http_client_curl "${_IMPORT_PREFIX}/lib/libopentelemetry_http_client_curl.dylib" )

# Import target "opentelemetry-cpp::otlp_recordable" for configuration "Release"
set_property(TARGET opentelemetry-cpp::otlp_recordable APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opentelemetry-cpp::otlp_recordable PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libopentelemetry_otlp_recordable.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libopentelemetry_otlp_recordable.dylib"
  )

list(APPEND _cmake_import_check_targets opentelemetry-cpp::otlp_recordable )
list(APPEND _cmake_import_check_files_for_opentelemetry-cpp::otlp_recordable "${_IMPORT_PREFIX}/lib/libopentelemetry_otlp_recordable.dylib" )

# Import target "opentelemetry-cpp::otlp_grpc_client" for configuration "Release"
set_property(TARGET opentelemetry-cpp::otlp_grpc_client APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opentelemetry-cpp::otlp_grpc_client PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "gRPC::grpc++"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libopentelemetry_exporter_otlp_grpc_client.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libopentelemetry_exporter_otlp_grpc_client.dylib"
  )

list(APPEND _cmake_import_check_targets opentelemetry-cpp::otlp_grpc_client )
list(APPEND _cmake_import_check_files_for_opentelemetry-cpp::otlp_grpc_client "${_IMPORT_PREFIX}/lib/libopentelemetry_exporter_otlp_grpc_client.dylib" )

# Import target "opentelemetry-cpp::otlp_grpc_exporter" for configuration "Release"
set_property(TARGET opentelemetry-cpp::otlp_grpc_exporter APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opentelemetry-cpp::otlp_grpc_exporter PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "gRPC::grpc++"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libopentelemetry_exporter_otlp_grpc.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libopentelemetry_exporter_otlp_grpc.dylib"
  )

list(APPEND _cmake_import_check_targets opentelemetry-cpp::otlp_grpc_exporter )
list(APPEND _cmake_import_check_files_for_opentelemetry-cpp::otlp_grpc_exporter "${_IMPORT_PREFIX}/lib/libopentelemetry_exporter_otlp_grpc.dylib" )

# Import target "opentelemetry-cpp::otlp_grpc_log_record_exporter" for configuration "Release"
set_property(TARGET opentelemetry-cpp::otlp_grpc_log_record_exporter APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opentelemetry-cpp::otlp_grpc_log_record_exporter PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "gRPC::grpc++"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libopentelemetry_exporter_otlp_grpc_log.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libopentelemetry_exporter_otlp_grpc_log.dylib"
  )

list(APPEND _cmake_import_check_targets opentelemetry-cpp::otlp_grpc_log_record_exporter )
list(APPEND _cmake_import_check_files_for_opentelemetry-cpp::otlp_grpc_log_record_exporter "${_IMPORT_PREFIX}/lib/libopentelemetry_exporter_otlp_grpc_log.dylib" )

# Import target "opentelemetry-cpp::otlp_grpc_metrics_exporter" for configuration "Release"
set_property(TARGET opentelemetry-cpp::otlp_grpc_metrics_exporter APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opentelemetry-cpp::otlp_grpc_metrics_exporter PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "gRPC::grpc++;absl::strings"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libopentelemetry_exporter_otlp_grpc_metrics.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libopentelemetry_exporter_otlp_grpc_metrics.dylib"
  )

list(APPEND _cmake_import_check_targets opentelemetry-cpp::otlp_grpc_metrics_exporter )
list(APPEND _cmake_import_check_files_for_opentelemetry-cpp::otlp_grpc_metrics_exporter "${_IMPORT_PREFIX}/lib/libopentelemetry_exporter_otlp_grpc_metrics.dylib" )

# Import target "opentelemetry-cpp::otlp_http_client" for configuration "Release"
set_property(TARGET opentelemetry-cpp::otlp_http_client APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opentelemetry-cpp::otlp_http_client PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "opentelemetry-cpp::proto;opentelemetry-cpp::http_client_curl"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libopentelemetry_exporter_otlp_http_client.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libopentelemetry_exporter_otlp_http_client.dylib"
  )

list(APPEND _cmake_import_check_targets opentelemetry-cpp::otlp_http_client )
list(APPEND _cmake_import_check_files_for_opentelemetry-cpp::otlp_http_client "${_IMPORT_PREFIX}/lib/libopentelemetry_exporter_otlp_http_client.dylib" )

# Import target "opentelemetry-cpp::otlp_http_exporter" for configuration "Release"
set_property(TARGET opentelemetry-cpp::otlp_http_exporter APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opentelemetry-cpp::otlp_http_exporter PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libopentelemetry_exporter_otlp_http.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libopentelemetry_exporter_otlp_http.dylib"
  )

list(APPEND _cmake_import_check_targets opentelemetry-cpp::otlp_http_exporter )
list(APPEND _cmake_import_check_files_for_opentelemetry-cpp::otlp_http_exporter "${_IMPORT_PREFIX}/lib/libopentelemetry_exporter_otlp_http.dylib" )

# Import target "opentelemetry-cpp::otlp_http_log_record_exporter" for configuration "Release"
set_property(TARGET opentelemetry-cpp::otlp_http_log_record_exporter APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opentelemetry-cpp::otlp_http_log_record_exporter PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libopentelemetry_exporter_otlp_http_log.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libopentelemetry_exporter_otlp_http_log.dylib"
  )

list(APPEND _cmake_import_check_targets opentelemetry-cpp::otlp_http_log_record_exporter )
list(APPEND _cmake_import_check_files_for_opentelemetry-cpp::otlp_http_log_record_exporter "${_IMPORT_PREFIX}/lib/libopentelemetry_exporter_otlp_http_log.dylib" )

# Import target "opentelemetry-cpp::otlp_http_metric_exporter" for configuration "Release"
set_property(TARGET opentelemetry-cpp::otlp_http_metric_exporter APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opentelemetry-cpp::otlp_http_metric_exporter PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libopentelemetry_exporter_otlp_http_metric.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libopentelemetry_exporter_otlp_http_metric.dylib"
  )

list(APPEND _cmake_import_check_targets opentelemetry-cpp::otlp_http_metric_exporter )
list(APPEND _cmake_import_check_files_for_opentelemetry-cpp::otlp_http_metric_exporter "${_IMPORT_PREFIX}/lib/libopentelemetry_exporter_otlp_http_metric.dylib" )

# Import target "opentelemetry-cpp::ostream_span_exporter" for configuration "Release"
set_property(TARGET opentelemetry-cpp::ostream_span_exporter APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opentelemetry-cpp::ostream_span_exporter PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libopentelemetry_exporter_ostream_span.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libopentelemetry_exporter_ostream_span.dylib"
  )

list(APPEND _cmake_import_check_targets opentelemetry-cpp::ostream_span_exporter )
list(APPEND _cmake_import_check_files_for_opentelemetry-cpp::ostream_span_exporter "${_IMPORT_PREFIX}/lib/libopentelemetry_exporter_ostream_span.dylib" )

# Import target "opentelemetry-cpp::ostream_metrics_exporter" for configuration "Release"
set_property(TARGET opentelemetry-cpp::ostream_metrics_exporter APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opentelemetry-cpp::ostream_metrics_exporter PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libopentelemetry_exporter_ostream_metrics.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libopentelemetry_exporter_ostream_metrics.dylib"
  )

list(APPEND _cmake_import_check_targets opentelemetry-cpp::ostream_metrics_exporter )
list(APPEND _cmake_import_check_files_for_opentelemetry-cpp::ostream_metrics_exporter "${_IMPORT_PREFIX}/lib/libopentelemetry_exporter_ostream_metrics.dylib" )

# Import target "opentelemetry-cpp::ostream_log_record_exporter" for configuration "Release"
set_property(TARGET opentelemetry-cpp::ostream_log_record_exporter APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opentelemetry-cpp::ostream_log_record_exporter PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libopentelemetry_exporter_ostream_logs.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libopentelemetry_exporter_ostream_logs.dylib"
  )

list(APPEND _cmake_import_check_targets opentelemetry-cpp::ostream_log_record_exporter )
list(APPEND _cmake_import_check_files_for_opentelemetry-cpp::ostream_log_record_exporter "${_IMPORT_PREFIX}/lib/libopentelemetry_exporter_ostream_logs.dylib" )

# Import target "opentelemetry-cpp::in_memory_span_exporter" for configuration "Release"
set_property(TARGET opentelemetry-cpp::in_memory_span_exporter APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opentelemetry-cpp::in_memory_span_exporter PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libopentelemetry_exporter_in_memory.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libopentelemetry_exporter_in_memory.dylib"
  )

list(APPEND _cmake_import_check_targets opentelemetry-cpp::in_memory_span_exporter )
list(APPEND _cmake_import_check_files_for_opentelemetry-cpp::in_memory_span_exporter "${_IMPORT_PREFIX}/lib/libopentelemetry_exporter_in_memory.dylib" )

# Import target "opentelemetry-cpp::prometheus_exporter" for configuration "Release"
set_property(TARGET opentelemetry-cpp::prometheus_exporter APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opentelemetry-cpp::prometheus_exporter PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libopentelemetry_exporter_prometheus.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libopentelemetry_exporter_prometheus.dylib"
  )

list(APPEND _cmake_import_check_targets opentelemetry-cpp::prometheus_exporter )
list(APPEND _cmake_import_check_files_for_opentelemetry-cpp::prometheus_exporter "${_IMPORT_PREFIX}/lib/libopentelemetry_exporter_prometheus.dylib" )

# Import target "opentelemetry-cpp::opentelemetry_exporter_zipkin_trace" for configuration "Release"
set_property(TARGET opentelemetry-cpp::opentelemetry_exporter_zipkin_trace APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opentelemetry-cpp::opentelemetry_exporter_zipkin_trace PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libopentelemetry_exporter_zipkin_trace.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libopentelemetry_exporter_zipkin_trace.dylib"
  )

list(APPEND _cmake_import_check_targets opentelemetry-cpp::opentelemetry_exporter_zipkin_trace )
list(APPEND _cmake_import_check_files_for_opentelemetry-cpp::opentelemetry_exporter_zipkin_trace "${_IMPORT_PREFIX}/lib/libopentelemetry_exporter_zipkin_trace.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
