# Gin sources
set(GIN_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/gin_host.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/gin_host_proxy.cc
)

if(ENABLE_ROCSHMEM OR ENABLE_ROCSHMEM_GIN)
    list(APPEND GIN_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/gin_host_rocshmem_common.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/gin_anvil_ipc_table_host.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/gin_plugin_rocshmem_gda.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/gin_plugin_anvil_sdma.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/gin_rocshmem_gda_factory.cc)
endif()

# Add gin sources to parent scope
set(GIN_SOURCES ${GIN_SOURCES} PARENT_SCOPE)
