// Weak OSS7 toggle for device bitcode / rccl-tests LTO when librccl strong symbol
// is not pulled into the final device link.
namespace gin_anvil {
namespace sdma {

__device__ __attribute__((weak)) int gin_anvil_sdma_oss7_enabled = 1;

}  // namespace sdma
}  // namespace gin_anvil
