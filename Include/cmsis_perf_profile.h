/**
 * @file cmsis_perf_profile.h
 * @author Carlos Morales
 * @brief Collection of performance profiling utilities
 * @version 0.1
 * @date 2025-6-13
 *
 * @copyright Copyright (c) 2025
 */
#include <stdint.h>
#include <assert.h>
// This includes the platform-dependent core_<cmxx>.h, defined in build_and_run_tests.sh 
// and passed to the compiler through CMake to avoid a long list of ifdef includes.
#include CORE_INCLUDE  

#ifndef _CMSIS_PERF_PROFILE_H_
#define _CMSIS_PERF_PROFILE_H_

#ifdef __cplusplus
extern "C" {
#endif

// Defines macros for the hardWare Performance Measurement Unit (PMU) 
// only if the target CPU supports it.

#define PERF_DWT_CHARACTERIZE(op_func_call)     \
    do {                                        \
        perf_dwt_init();                        \
        perf_dwt_start();                       \
        op_func_call;                           \
        perf_dwt_stop();                        \
        perf_dwt_capture(&op_dwt_perf_count);   \
    } while (0)

#if defined (__PMU_PRESENT) && (__PMU_PRESENT == 1U)
#define PERF_PMU_CHARACTERIZE(op_func_call)       \
    do {                                          \
        pmu_characterize_setup(&op_perf_pmu_cfg); \
        op_func_call;                             \
        perf_pmu_disable();                       \
        perf_pmu_get_counters(&op_perf_pmu_cfg);  \
    } while (0)
#else
#define PERF_PMU_CHARACTERIZE(op_func_call) \
    do { } while (0)
#endif

#if defined (__PMU_PRESENT) && (__PMU_PRESENT == 1U)

#define PERF_PMU_EVENT_COUNTER_SIZE_16 0
#define PERF_PMU_EVENT_COUNTER_SIZE_32 1
#define PERF_PMU_STATUS_SUCCESS 0
#define PERF_PMU_STATUS_INVALID_HANDLE 1
#define PERF_PMU_INVALID_CONFIG 3
#define PERF_PMU_STATUS_INIT_FAILED 4

// Bitmask to enable all PMU counters at the hardware level
#define VALID_PMU_COUNTERS  (PMU_CNTENSET_CNT0_ENABLE_Msk | \
                             PMU_CNTENSET_CNT1_ENABLE_Msk | \
                             PMU_CNTENSET_CNT2_ENABLE_Msk | \
                             PMU_CNTENSET_CNT3_ENABLE_Msk | \
                             PMU_CNTENSET_CNT4_ENABLE_Msk | \
                             PMU_CNTENSET_CNT5_ENABLE_Msk | \
                             PMU_CNTENSET_CNT6_ENABLE_Msk | \
                             PMU_CNTENSET_CNT7_ENABLE_Msk | \
                             PMU_CNTENSET_CCNTR_ENABLE_Msk)

#define VALID_PMU_OVSCLRS   (PMU_OVSCLR_CNT0_STATUS_Msk | \
                             PMU_OVSCLR_CNT1_STATUS_Msk | \
                             PMU_OVSCLR_CNT2_STATUS_Msk | \
                             PMU_OVSCLR_CNT3_STATUS_Msk | \
                             PMU_OVSCLR_CNT4_STATUS_Msk | \
                             PMU_OVSCLR_CNT5_STATUS_Msk | \
                             PMU_OVSCLR_CNT6_STATUS_Msk | \
                             PMU_OVSCLR_CNT7_STATUS_Msk | \
                             PMU_OVSCLR_CYCCNT_STATUS_Msk)

#endif // #if defined (__PMU_PRESENT) && (__PMU_PRESENT == 1U)

// Struct to capture hardware Data Watchpoint Tracker counts
typedef struct {
    uint32_t cycCnt;
    uint32_t cpiCnt;
} perf_ctx_t;

static perf_ctx_t op_dwt_perf_count;

#if defined (__PMU_PRESENT) && (__PMU_PRESENT == 1U)

// Struct representing data for a single Performance Measurement Unit event
typedef struct {
    bool enabled;
    uint32_t eventId;
    uint32_t counterSize;
} perf_pmu_event_t;

// Struct to hold PMU event count for a single event
typedef struct {
    bool added;            /*!< Used to track if the event has been added to 
                                the perf_pmu_util_config_t */
    uint32_t mapIndex;     /// pmu_map index for this event
    uint32_t counterValue; /// Value read from the PMU
} perf_pmu_counter_t;

// Struct to hold general PMU configuration, including all PMU events being tracked and their counts
typedef struct {
    perf_pmu_event_t events[8];
    perf_pmu_counter_t counter[8];
} perf_pmu_config_t;

// Struct to hold PMU event types
typedef struct {
    uint32_t eventType[__PMU_NUM_EVENTCNT];
    uint32_t counters;
} perf_pmu_util_config_t;

// A single PMU map entry
typedef struct {
    uint32_t eventId;
    const char regname[50];
    const char description[120];
} perf_pmu_map_entry;

static perf_pmu_util_config_t perf_pmu_util_config;
static bool perf_pmu_initialized = false;
static bool perf_pmu_profiling = false;
static uint32_t perf_pmu_config_index[8];
static perf_pmu_config_t op_perf_pmu_cfg;

// Map containing 4 PMU events to be captured
const perf_pmu_map_entry perf_pmu_map[] = {
    { ARM_PMU_INST_RETIRED, "ARM_PMU_INST_RETIRED", "Instruction architecturally executed" },
    { ARM_PMU_MVE_FP_RETIRED, "ARM_PMU_MVE_FP_RETIRED", "MVE floating-point instruction architecturally executed" },
    { ARM_PMU_MVE_INST_RETIRED, "ARM_PMU_MVE_INST_RETIRED", "MVE instruction architecturally executed" },
    { ARM_PMU_MVE_INT_MAC_RETIRED, "ARM_PMU_MVE_INT_MAC_RETIRED", 
    "MVE multiply or multiply-accumulate instruction architecturally executed" }
}; 

#define PERF_PMU_MAP_SIZE (sizeof(perf_pmu_map) / sizeof(perf_pmu_map[0]))
#endif

/**
 * @brief Enables Debug and ITM Trace registers for the hardware Data Watchpoint Tracker (DWT).
 * 
 * @details
 *    1. Supported by all CPUs
 *
 */
__STATIC_INLINE void perf_dwt_regs_enable(void)
{
    //
    // Perform general debug/tracing configuration.
    //
#ifndef DCB
    CoreDebug->DEMCR = CoreDebug_DEMCR_TRCENA_Msk;
#else
    DCB->DEMCR |= DCB_DEMCR_TRCENA_Msk;
#endif

    //
    // Set the enable bits in the ITM Trace Privilege Register and the
    // ITM Trace Enable Register to enable trace data output.
    //
    ITM->TPR = 0x0000000F;
    ITM->TER = 0xFFFFFFFF;

    //
    // Write the fields in the ITM Trace Control Register.
    //
    ITM->TCR =
        _VAL2FLD(ITM_TCR_TRACEBUSID, 0x15)      |
        _VAL2FLD(ITM_TCR_GTSFREQ, 1)            |
        _VAL2FLD(ITM_TCR_TSPRESCALE, 1)         |
#ifndef ITM_TCR_STALLENA
        _VAL2FLD(CoreDebug_DHCSR_C_SNAPSTALL, 0)|
#else
        _VAL2FLD(ITM_TCR_STALLENA, 0)           |
#endif
        _VAL2FLD(ITM_TCR_SWOENA, 1)             |
        _VAL2FLD(ITM_TCR_DWTENA, 0)             |
        _VAL2FLD(ITM_TCR_SYNCENA, 0)            |
        _VAL2FLD(ITM_TCR_TSENA, 0)              |
        _VAL2FLD(ITM_TCR_ITMENA, 1);
}

/**
 * @brief Reset the hardware DWT performance counters
 *
 */
__STATIC_INLINE void perf_dwt_reset(void)
{
    DWT->CYCCNT = 0;
    DWT->CPICNT = 0;
    DWT->EXCCNT = 0;
    DWT->SLEEPCNT = 0;
    DWT->LSUCNT = 0;
    DWT->FOLDCNT = 0;
}

/**
 * @brief Configure the hardware performance capture unit
 *
 */
__STATIC_INLINE void perf_dwt_init(void)
{
    op_dwt_perf_count.cycCnt = 0;
    op_dwt_perf_count.cpiCnt = 0;
    DWT->CTRL = 0;
    perf_dwt_reset();
}

/**
 * @brief Start the hardware performance capture unit
 *
 */
__STATIC_INLINE void perf_dwt_start(void)
{
    perf_dwt_regs_enable();
    
    DWT->CTRL = _VAL2FLD(DWT_CTRL_CYCCNTENA, 1) | _VAL2FLD(DWT_CTRL_CPIEVTENA, 1) |
                _VAL2FLD(DWT_CTRL_EXCEVTENA, 1) | _VAL2FLD(DWT_CTRL_SLEEPEVTENA, 1) |
                _VAL2FLD(DWT_CTRL_LSUEVTENA, 1) | _VAL2FLD(DWT_CTRL_FOLDEVTENA, 1) |
                _VAL2FLD(DWT_CTRL_CYCEVTENA, 1);
}

/**
 * @brief Stop the hardware performance capture unit
 *
 */
__STATIC_INLINE void perf_dwt_stop(void) 
{ 
    DWT->CTRL = 0; 
}

/**
 * @brief Capture the current value of the hardware performance
 * counters in the provided struct
 *
 * @param c struct with members to hold snapshot of hardware counters
 */
__STATIC_INLINE void perf_dwt_capture(perf_ctx_t *c)
{
    c->cycCnt = DWT->CYCCNT;
    c->cpiCnt = DWT->CPICNT;
}

/**
 * @brief Perform subtraction on two unsigned 32 bit integers accounting for wraparound.
 *
 */
__STATIC_INLINE uint32_t perf_delta_byte_counter_wrap(uint32_t e, uint32_t s) 
{
    uint32_t retval = (e < s) ? (e + 256 - s) : e - s;
    return retval;
}

/**
 * @brief Calculate the delta between two hardware performance snapshots
 *
 * @param s starting snapshot struct
 * @param e ending snapshot struct
 * @param d Calculated delta
 */
__STATIC_INLINE void perf_dwt_delta(perf_ctx_t *s, perf_ctx_t *e, perf_ctx_t *d)
{
    d->cycCnt = e->cycCnt - s->cycCnt; // 32 bits, probably won't wrap
    d->cpiCnt = perf_delta_byte_counter_wrap(e->cpiCnt, s->cpiCnt);
}

/**
 * @brief Print a hardware performance counter snapshot with JSON format
 *
 * @param c Performance snapshot to be printed
 */
__STATIC_INLINE void perf_dwt_print(perf_ctx_t *c)
{
    printf("\"DWT_CYCCNT\":%lu, ", c->cycCnt);
}

// Defines functions for the hardware Performance Measurement Unit (PMU) 
// only if the target CPU supports it.
#if defined (__PMU_PRESENT) && (__PMU_PRESENT == 1U)

/**
 * @brief Enables registers necessary for the PMU.
 * 
 * @details
 *    1. Supported by Cortex-M23 and later
 *
 */
__STATIC_INLINE void perf_pmu_enable(void)
{
    ARM_PMU_Enable();
    ARM_PMU_CYCCNT_Reset();
    ARM_PMU_EVCNTR_ALL_Reset();
    ARM_PMU_Set_CNTR_OVS(VALID_PMU_OVSCLRS);
    PMU->CTRL |= PMU_CTRL_CYCCNT_RESET_Msk;
    PMU->CTRL |= PMU_CTRL_EVENTCNT_RESET_Msk;
    PMU->OVSCLR = (PMU_OVSCLR_CNT0_STATUS_Msk | 
                   PMU_OVSCLR_CNT1_STATUS_Msk | 
                   PMU_OVSCLR_CNT2_STATUS_Msk | 
                   PMU_OVSCLR_CNT3_STATUS_Msk | 
                   PMU_OVSCLR_CNT4_STATUS_Msk | 
                   PMU_OVSCLR_CNT5_STATUS_Msk | 
                   PMU_OVSCLR_CNT6_STATUS_Msk | 
                   PMU_OVSCLR_CNT7_STATUS_Msk | 
                   PMU_OVSCLR_CYCCNT_STATUS_Msk);   
}

/**
 * @brief Disables registers necessary for the hardware Performance Measurement Unit (PMU). 
 * Called to stop profiling or right before a reset.
 * 
 * @details
 *    1. Supported by Cortex-M23 and later
 *
 */
__STATIC_INLINE void perf_pmu_disable(void)
{
    ARM_PMU_Disable();
}

/**
 * @brief Initializes a PMU event object with the given event ID and counter size.
 * 
 * @param event pointer to the PMU event being created
 * @param eventId the ID of the PMU event being created
 * @param counterSize counter size for the PMU event being created
 *
 */
__STATIC_INLINE void perf_pmu_event_create(perf_pmu_event_t *event, uint32_t eventId, 
                                           uint32_t counterSize)
{
    event->enabled = true;
    event->eventId = eventId;
    event->counterSize = counterSize;
}

/**
 * @brief Enable specific counters of the PMU based on the value passed to the function 
 * 
 * @param countersEnable an integer that will encode which counters of the PMU should be enabled
 * 
 * @details
 *    1. Supported by Cortex-M23 and later
 *
 */
__STATIC_INLINE uint32_t perf_pmu_cntr_enable(uint32_t countersEnable) 
{
    if (countersEnable & (~VALID_PMU_COUNTERS)) {
        return -1;
    }
    //
    // PMU counters increment if the appropriate bit in PMU_CNTENSET register is set.
    //
    ARM_PMU_CNTR_Enable(countersEnable);
    return 0;
}

/**
 * @brief Disable specific counters of the PMU based on the value passed to the function 
 * 
 * @param countersEnable an integer that will encode which counters of the PMU should be disabled
 * 
 * @details
 *    1. Supported by Cortex-M23 and later
 *
 */
__STATIC_INLINE uint32_t perf_pmu_cntr_disable(uint32_t countersDisable)
{
    if (countersDisable & (~VALID_PMU_COUNTERS)) {
        return -1;
    }

    ARM_PMU_CNTR_Disable(countersDisable);
    return 0;
}

/**
 * @brief Get the PMU map index of a given event.
 * 
 * @param eventId the event to get
 * 
 * @details
 *    1. Supported by Cortex-M23 and later
 *
 */
__STATIC_INLINE int32_t perf_pmu_get_map_index(uint32_t eventId) 
{
    // iterate over the length of pmu_map until we find the eventId
    for (uint32_t i = 0; i < PERF_PMU_MAP_SIZE; i++) {
        // printf("Checking %d for 0x%x match\n", i, eventId);
        if (perf_pmu_map[i].eventId == eventId) {
            return i;
        }
    }
    return -1;
}

/**
 * @brief Initialize the PMU based on the specific events to be counted.
 * 
 * @param pPMUConfig a configuration parameter that specifies which counters to initialize and profile
 * 
 * @details
 *    1. Supported by Cortex-M23 and later
 *
 */
__STATIC_INLINE uint32_t perf_pmu_util_init(perf_pmu_util_config_t *pPMUConfig)
{
    uint32_t evtcnt;
    uint32_t ui32IntMsk = 0;

    if (pPMUConfig == NULL) {
        return 0;
    }

    // Configure for general debug i.e. DBG power, tracing, TPIU, etc.
    // am_hal_debug_enable();
    DCB->DEMCR |= DCB_DEMCR_TRCENA_Msk;

    // Enable Trace
    DCB->DEMCR |= DCB_DEMCR_MON_EN_Msk;

    // Enable Low Overhead Loops
    SCB->CCR |= SCB_CCR_LOB_Msk;

    // Configure Event Counters Registers
    for (evtcnt = 0; evtcnt < __PMU_NUM_EVENTCNT; evtcnt++) {
        //
        // Set event type for expected event counters.
        //
        if (pPMUConfig->counters & (1U << evtcnt)) {
            ARM_PMU_Set_EVTYPER(evtcnt, pPMUConfig->eventType[evtcnt]);
        }
    }

    // Reset Cycle Counter and Event Counters
    ARM_PMU_CYCCNT_Reset();
    ARM_PMU_EVCNTR_ALL_Reset();

    // Clears overflow status of event counter register.
    ARM_PMU_Set_CNTR_OVS(VALID_PMU_OVSCLRS);
    for (evtcnt = 0; evtcnt < __PMU_NUM_EVENTCNT; evtcnt++) {
        if (evtcnt == __PMU_NUM_EVENTCNT - 1) {
            ui32IntMsk |= (1UL << evtcnt);
        } else if ((pPMUConfig->eventType[evtcnt] == ARM_PMU_CHAIN) || 
                   (pPMUConfig->eventType[evtcnt + 1] != ARM_PMU_CHAIN)) {
            ui32IntMsk |= (1UL << evtcnt);
        }
    }
    ui32IntMsk |= PMU_INTENSET_CCYCNT_ENABLE_Msk;
    ARM_PMU_Set_CNTR_IRQ_Enable(ui32IntMsk);

    // Enable Cycle Counter and Event Counters
    perf_pmu_cntr_enable(pPMUConfig->counters);

    return 1;
} // perf_pmu_util_init()

/**
 * @brief Initialize the PMU based on the overall configuration setting. 
 *        Calls helper method perf_pmu_util_init.
 * 
 * @param cfg a configuration parameter that specifies which counters to initialize and profile
 * 
 * @details
 *    1. Supported by Cortex-M23 and later
 *
 */
__STATIC_INLINE uint32_t perf_pmu_init(perf_pmu_config_t *cfg) 
{
    uint8_t totalCounters = 0;
    uint32_t counterMask = PMU_CNTENSET_CCNTR_ENABLE_Msk;

    assert(cfg != NULL);

    // Init counter state and perf_pmu_config
    for (int i = 0; i < 8; i++) {
        cfg->counter[i].counterValue = 0;
        cfg->counter[i].added = false;
        if (cfg->events[i].enabled) {
            cfg->counter[i].mapIndex = perf_pmu_get_map_index(cfg->events[i].eventId);
        }
        perf_pmu_util_config.eventType[i] = 0xffff;
        perf_pmu_util_config.counters = 0;
    }

    // Create a am_pmu_util_config_t to pass to the am_pmu_util_init function
    // Pack the 32 bit counters to the front of the array
    totalCounters = 0;
    for (int i = 0; i < 8; i++) {
        if ((cfg->events[i].enabled) && (!cfg->counter[i].added) 
           && (cfg->events[i].counterSize == PERF_PMU_EVENT_COUNTER_SIZE_32)) {
            counterMask |= (1 << totalCounters);
            perf_pmu_config_index[totalCounters] = i; // map the pmu counter to the cf counter index
            perf_pmu_util_config.eventType[totalCounters++] = cfg->events[i].eventId;
            counterMask |= (1 << totalCounters);
            perf_pmu_util_config.eventType[totalCounters++] = ARM_PMU_CHAIN;
            cfg->counter[i].added = true;
        }
    }

    // Now add in the 16 bit counters
    for (int i = 0; i < 8; i++) {
        if ((cfg->events[i].enabled) && (!cfg->counter[i].added) 
           && (cfg->events[i].counterSize == PERF_PMU_EVENT_COUNTER_SIZE_16)) {
            counterMask |= (1 << totalCounters);
            // map the pmu counter to the cf counter index
            perf_pmu_config_index[totalCounters] = i; 
            perf_pmu_util_config.eventType[totalCounters++] = cfg->events[i].eventId;
            cfg->counter[i].added = true;
        }
    }

    perf_pmu_util_config.counters = counterMask;

    perf_pmu_util_init(&perf_pmu_util_config);
    perf_pmu_initialized = true;
    perf_pmu_profiling = true;

    return PERF_PMU_STATUS_SUCCESS;
}

/**
 * @brief Capture the current value of the PMU event counters in the provided PMU config struct
 *
 * @param cfg struct with members to hold snapshot of PMU event counters
 * 
 * @details
 *    1. Supported by Cortex-M23 and later
 */
__STATIC_INLINE uint32_t perf_pmu_get_counters(perf_pmu_config_t *cfg) 
{
    // Adapted from am_pmu_util_get_profiling
    uint32_t value;

    if (cfg == NULL) {
        return PERF_PMU_STATUS_INVALID_HANDLE;
    }
    if (!perf_pmu_initialized) {
        return PERF_PMU_STATUS_INIT_FAILED;
    }

    perf_pmu_cntr_disable(perf_pmu_util_config.counters);

    // Update cfg->counter for each enabled counter
    for (int pmu_index = 0; pmu_index < 8; pmu_index++) {
        if (perf_pmu_util_config.counters & (1 << pmu_index)) {
            // Handle chain counters
            uint32_t cfgIndex = perf_pmu_config_index[pmu_index];
            if (cfg->events[cfgIndex].counterSize == PERF_PMU_EVENT_COUNTER_SIZE_32) {
                uint32_t baseValue = ARM_PMU_Get_EVCNTR(pmu_index);
                uint32_t chainValue = ARM_PMU_Get_EVCNTR(pmu_index + 1);
                cfg->counter[cfgIndex].counterValue = baseValue + (chainValue << 16);
                pmu_index++;
            } else {
                value = ARM_PMU_Get_EVCNTR(pmu_index);
                cfg->counter[cfgIndex].counterValue = value;
            } 
        }
    }

    ARM_PMU_CYCCNT_Reset();
    ARM_PMU_EVCNTR_ALL_Reset();
    ARM_PMU_Set_CNTR_OVS(VALID_PMU_OVSCLRS);

    perf_pmu_cntr_enable(perf_pmu_util_config.counters);
    return PERF_PMU_STATUS_SUCCESS;
}

/**
 * @brief Print a PMU event counter snapshot with JSON format
 *
 * @param cfg PMU config containing event snapshot to be printed
 * 
 * @details
 *    1. Supported by Cortex-M23 and later
 */
__STATIC_INLINE uint32_t perf_pmu_print_counters(perf_pmu_config_t *cfg) 
{
    if (cfg == NULL) {
        return PERF_PMU_STATUS_INVALID_HANDLE;
    }

    printf("{");
    for (int i = 0; i < 4; i++) {
        if (cfg->events[i].enabled) {
            uint32_t mapIndex = cfg->counter[i].mapIndex;
            printf("\"%s\":%lu", perf_pmu_map[mapIndex].regname, cfg->counter[i].counterValue);
            if (i != 3) {
                printf(", ");
            }
        }
    }
    printf("}");

    return PERF_PMU_STATUS_SUCCESS;
}

/**
 * @brief Enable and set up PMU characterization for each PMU even tin the map
 *
 * @param cfg PMU config to capture event counts
 * 
 * @details
 *    1. Supported by Cortex-M23 and later
 */
__STATIC_INLINE void pmu_characterize_setup(perf_pmu_config_t *cfg)
{
    perf_pmu_enable();

    // Walk through every event counter in PMU map
    uint32_t map_index = 0;
    // We can do 4 32b counters at a time
    perf_pmu_event_create(&(cfg->events[0]), perf_pmu_map[map_index].eventId, 
                          PERF_PMU_EVENT_COUNTER_SIZE_32);
    perf_pmu_event_create(&(cfg->events[1]), perf_pmu_map[(++map_index)%PERF_PMU_MAP_SIZE].eventId, 
                          PERF_PMU_EVENT_COUNTER_SIZE_32);
    perf_pmu_event_create(&(cfg->events[2]), perf_pmu_map[(++map_index)%PERF_PMU_MAP_SIZE].eventId, 
                          PERF_PMU_EVENT_COUNTER_SIZE_32);
    perf_pmu_event_create(&(cfg->events[3]), perf_pmu_map[(++map_index)%PERF_PMU_MAP_SIZE].eventId, 
                          PERF_PMU_EVENT_COUNTER_SIZE_32);

    perf_pmu_init(cfg);
}

#endif

/**
 * @brief Print dimension string of a tensor as numbers with 'x' between given an array of dimensions. JSON formatted.
 *
 * @param dims Array of tensor dimensions
 * @param dim_len Length of the array dims
 */
__STATIC_INLINE void print_dimension_string(const int32_t dims[], const int32_t dims_len)
{
    if (dims == NULL) {
        return;
    }

    for (int i = 0; i < dims_len - 1; i++) {
        printf("%ld", dims[i]);
        printf("x");
    }
    printf("%ld", dims[dims_len - 1]);
    printf("\", ");
}

/**
 * @brief Print complete Performance summary for a given operation
 *
 * @param op_name Name of the function being profiled
 * @param dims Array holding the dimensions passed to the operation
 * @param dims_len Length of dims
 * @param est_macs Calculated estimated macs value to print
 */
__STATIC_INLINE void perf_print_all(const char *op_name, const int32_t dims[], const int32_t dims_len, const int32_t rhs_dims[], const int32_t rhs_dims_len)
{
    printf("{");

    printf("\"date\":\"%s\", ", __DATE__ " " __TIME__);

    printf("\"op\":\"%s\", ", op_name);

    if (rhs_dims == NULL)
    {
        if (dims != NULL)
        {
            printf("\"shape\":\"");
            print_dimension_string(dims, dims_len);
        }
    }
    else
    {
        printf("\"lhs_shape\":\"");
        print_dimension_string(dims, dims_len);
        printf("\"rhs_shape\":\"");
        print_dimension_string(rhs_dims, rhs_dims_len);
    }

    // printf("\"est_mac\":%ld, ", est_macs);
    printf("\"DWT_CYCCNT\":%lu", op_dwt_perf_count.cycCnt);

    #if defined (__PMU_PRESENT) && (__PMU_PRESENT == 1U)
        printf(", \"PMU\":");
        perf_pmu_print_counters(&op_perf_pmu_cfg);
    #endif

    printf("}");
}

#ifdef __cplusplus
}
#endif

#endif // _CMSIS_PERF_PROFILE_H_