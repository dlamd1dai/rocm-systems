/* Load-time stub for rocSHMEM host symbols referenced by librccl.so.
 * The GIN rocSHMEM-GDA unit test (GinRocshmemGdaTemplateTest) exercises only
 * in-TU device-side QueuePair stubs; the real rocSHMEM host library is not
 * built into this image, so librccl.so is left with undefined host symbols.
 * These stubs exist solely to satisfy the dynamic loader; none are invoked by
 * the fixtures test. Built as C so identifiers map 1:1 to the mangled names. */

/* Data objects (global variables). Oversized to tolerate any incidental read. */
char _ZN8rocshmem3ibvE[4096];
char _ZN8rocshmem6envvar3gda13traffic_classE[4096];
char _ZN8rocshmem6envvar3gda7sq_sizeE[4096];
char _ZN8rocshmem6envvar9log_flagsE[4096];

/* Function symbols: no-op definitions (never called by the fixtures test). */
void _ZN8rocshmem10IBVWrapper10dealloc_pdEP6ibv_pd(void) {}
void _ZN8rocshmem10IBVWrapper10destroy_cqEP6ibv_cq(void) {}
void _ZN8rocshmem10IBVWrapper10destroy_qpEP6ibv_qp(void) {}
void _ZN8rocshmem10IBVWrapper10query_portEP11ibv_contexthP13ibv_port_attr(void) {}
void _ZN8rocshmem10IBVWrapper11cq_ex_to_cqEP9ibv_cq_ex(void) {}
void _ZN8rocshmem10IBVWrapper11open_deviceEP10ibv_device(void) {}
void _ZN8rocshmem10IBVWrapper12close_deviceEP11ibv_context(void) {}
void _ZN8rocshmem10IBVWrapper12create_qp_exEP11ibv_contextP19ibv_qp_init_attr_ex(void) {}
void _ZN8rocshmem10IBVWrapper12query_deviceEP11ibv_contextP15ibv_device_attr(void) {}
void _ZN8rocshmem10IBVWrapper15get_device_listEPi(void) {}
void _ZN8rocshmem10IBVWrapper15get_device_nameEP10ibv_device(void) {}
void _ZN8rocshmem10IBVWrapper15query_gid_tableEP11ibv_contextP13ibv_gid_entrymj(void) {}
void _ZN8rocshmem10IBVWrapper16free_device_listEPP10ibv_device(void) {}
void _ZN8rocshmem10IBVWrapper19alloc_parent_domainEP11ibv_contextP27ibv_parent_domain_init_attr(void) {}
void _ZN8rocshmem10IBVWrapper19is_dmabuf_supportedEv(void) {}
void _ZN8rocshmem10IBVWrapper6reg_mrEP6ibv_pdPvmiPNS_12HIPAllocatorE(void) {}
void _ZN8rocshmem10IBVWrapper8alloc_pdEP11ibv_context(void) {}
void _ZN8rocshmem10IBVWrapper9modify_qpEP6ibv_qpP11ibv_qp_attri(void) {}
void _ZN8rocshmem12mlx5_devx_qp4dumpEi(void) {}
void _ZN8rocshmem14mlx5dv_funcs_t10destroy_qpERNS_12mlx5_devx_qpE(void) {}
void _ZN8rocshmem14mlx5dv_funcs_t9create_qpERNS_12mlx5_devx_qpEP11ibv_contextP6ibv_pdt(void) {}
void _ZN8rocshmem14mlx5dv_funcs_t9modify_qpERNS_12mlx5_devx_qpEP11ibv_qp_attrij(void) {}
void _ZN8rocshmem15MemoryAllocator10deallocateEPv(void) {}
void _ZN8rocshmem15MemoryAllocator8allocateEPPvm(void) {}
void _ZN8rocshmem15MemoryAllocatorC2EPF10hipError_tPPvmjEPFS1_S2_Ej(void) {}
void _ZN8rocshmem18GetClosestNicToGpuEiPKcPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(void) {}
void _ZN8rocshmem9QueuePairC1EP6ibv_pdi(void) {}
void rocshmem_gin_init_constmem(void) {}
