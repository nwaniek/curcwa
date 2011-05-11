#ifndef __GLOBAL_H__858EF715_649D_4C4A_BF4E_C8674B7862D0
#define __GLOBAL_H__858EF715_649D_4C4A_BF4E_C8674B7862D0

/*
 * disable the following define if you want to store dev::q, dev::kt and dev::kr
 * in col-order fashion
 */
#define ROW_MAJOR 1

/*
 * disable the following to switch inveps/epsdiag from shared memory to
 * non-shared memory version
 */
#define INVEPS_EPSDIAG_SHM 1

/*
 * disable to use non-shared memory version for prepare_evm_tm in secular.cu
 */
#define SECULAR_PREPARE_EVM_SHM 1

/*
 * disable to use non-shared memort version for result computation in secular.cu
 */
#define SECULAR_COMPUTE_RESULTS_SHM 1




#endif /* __GLOBAL_H__858EF715_649D_4C4A_BF4E_C8674B7862D0 */

