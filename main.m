#import <Metal/Metal.h>
#include <stdbool.h>  // For bool, true, false

// Must exactly match the shader.
const uint32_t TEST_SIZE = 65535;
const uint32_t BLOCK_DIM = 32;
const uint32_t ERROR_TYPE_MESSAGE = 1u;
const uint32_t ERROR_TYPE_SHUFFLE_READY = 2u;
const uint32_t ERROR_TYPE_SHUFFLE_INC = 3u;
const uint32_t ERROR_TYPE_SGSIZE = 4u;
const uint32_t FLAG_NOT_READY = 0u;
const uint32_t FLAG_READY = 0x40000000u;
const uint32_t FLAG_INCLUSIVE = 0x80000000u;
const uint32_t VALUE_MASK = 0xFFFFu;

static bool SetupPipelineStates(id<MTLDevice> device, id<MTLComputePipelineState>* outInitPSO,
                                id<MTLComputePipelineState>* outStressPSO, NSError** errorPtr) {
    NSURL* initUrl = [NSURL fileURLWithPath:@"initShader.metallib"];
    id<MTLLibrary> initLibrary = [device newLibraryWithURL:initUrl error:errorPtr];
    if (initLibrary == nil) {
        NSLog(@"Failed to load the init library: %@.", (*errorPtr).localizedDescription);
        return false;
    }
    NSURL* stressUrl = [NSURL fileURLWithPath:@"stressShader.metallib"];
    id<MTLLibrary> stressLibrary = [device newLibraryWithURL:stressUrl error:errorPtr];
    if (stressLibrary == nil) {
        NSLog(@"Failed to load the stress library: %@.", (*errorPtr).localizedDescription);
        return false;
    }

    id<MTLFunction> initEntry = [initLibrary newFunctionWithName:@"init"];
    if (initEntry == nil) {
        NSLog(@"Failed to find the init entrypoint function.");
        return false;
    }
    id<MTLFunction> stressEntry = [stressLibrary newFunctionWithName:@"stress"];
    if (stressEntry == nil) {
        NSLog(@"Failed to find the stress entrypoint function.");
        return false;
    }

    *outInitPSO = [device newComputePipelineStateWithFunction:initEntry error:errorPtr];
    if (*outInitPSO == nil) {
        NSLog(@"Failed to create init pipeline state object, error %@.",
              (*errorPtr).localizedDescription);
        return false;
    }
    *outStressPSO = [device newComputePipelineStateWithFunction:stressEntry error:errorPtr];
    if (*outStressPSO == nil) {
        NSLog(@"Failed to create stress pipeline state object, error %@.",
              (*errorPtr).localizedDescription);
        return false;
    }
    return true;
}

static bool CreateMetalBuffers(id<MTLDevice> device, id<MTLBuffer>* outTransferBuffer,
                               id<MTLBuffer>* outScanBuffer, id<MTLBuffer>* outScanBumpBuffer,
                               id<MTLBuffer>* outErrorsBuffer) {
    *outTransferBuffer = [device newBufferWithLength:(TEST_SIZE * 4 * sizeof(uint32_t))
                                             options:MTLResourceStorageModeShared];
    *outScanBuffer = [device newBufferWithLength:(TEST_SIZE * 2 * sizeof(uint32_t))
                                         options:MTLResourceStorageModePrivate];
    *outScanBumpBuffer =
        [device newBufferWithLength:(sizeof(uint32_t)) options:MTLResourceStorageModePrivate];
    *outErrorsBuffer = [device newBufferWithLength:(TEST_SIZE * 4 * sizeof(uint32_t))
                                           options:MTLResourceStorageModePrivate];

    if (!*outTransferBuffer || !*outScanBuffer || !*outScanBumpBuffer || !*outErrorsBuffer) {
        NSLog(@"Failed to create one or more Metal buffers.");
        return false;
    }
    return true;
}

static bool DispatchKernels(id<MTLCommandQueue> commandQueue, id<MTLComputePipelineState> initPSO,
                            id<MTLComputePipelineState> stressPSO, id<MTLBuffer> scanBumpBuffer,
                            id<MTLBuffer> scanBuffer, id<MTLBuffer> errorsBuffer,
                            MTLSize initGridDim, MTLSize initBlockDim, MTLSize stressGridDim,
                            MTLSize stressBlockDim) {
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    if (commandBuffer == nil) {
        NSLog(@"Failed to create the command buffer for dispatch.");
        return false;
    }

    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    if (computeEncoder == nil) {
        NSLog(@"Failed to create the command encoder for dispatch.");
        return false;
    }

    [computeEncoder setComputePipelineState:initPSO];
    [computeEncoder setBuffer:scanBumpBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:scanBuffer offset:0 atIndex:1];
    [computeEncoder setBuffer:errorsBuffer offset:0 atIndex:2];
    [computeEncoder dispatchThreadgroups:initGridDim threadsPerThreadgroup:initBlockDim];

    [computeEncoder setComputePipelineState:stressPSO];
    [computeEncoder setBuffer:scanBumpBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:scanBuffer offset:0 atIndex:1];
    [computeEncoder setBuffer:errorsBuffer offset:0 atIndex:2];
    [computeEncoder dispatchThreadgroups:stressGridDim threadsPerThreadgroup:stressBlockDim];

    [computeEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    return true;
}

// Sanity checks the scan
static bool ValidateScanBuffer(id<MTLCommandQueue> commandQueue, id<MTLBuffer> scanBuffer,
                               id<MTLBuffer> transferBuffer) {
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    if (commandBuffer == nil) {
        NSLog(@"Failed to create the command buffer for scan buffer validation.");
        return false;
    }

    id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
    if (blitEncoder == nil) {
        NSLog(@"Failed to create the blit encoder for scan buffer validation.");
        return false;
    }

    [blitEncoder copyFromBuffer:scanBuffer
                   sourceOffset:0
                       toBuffer:transferBuffer
              destinationOffset:0
                           size:TEST_SIZE * 2 * sizeof(uint32_t)];
    [blitEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    uint32_t* scan = transferBuffer.contents;
    for (uint32_t k = 0; k < TEST_SIZE; ++k) {
        uint32_t index = k * 2;
        uint32_t rejoinedVal = (scan[index] & 0xffff) | (scan[index + 1] << 16);
        if (rejoinedVal != 1024 * (k + 1)) {
            NSLog(@"Test failed: got %u at %u\n", rejoinedVal, k);
            return false;
        }
    }
    return true;
}

bool CheckError(uint32_t errCode, uint32_t got, uint32_t tile_id, uint32_t tid) {
    if (!errCode) {
        return true;
    }

    if (errCode == ERROR_TYPE_MESSAGE) {
        uint32_t val_content_for_ready_state = (1024u >> (tid * 16u)) & VALUE_MASK;
        uint32_t expected_full_value_for_ready_state = val_content_for_ready_state | FLAG_READY;

        printf(
            "Message Passing type error at tile %u, thread %u: GOT 0x%08X.\n"
            "  Expected patterns include:\n"
            "    1. 0x%08X (NOT_READY)\n"
            "    2. 0x%08X (READY state: value 0x%04X combined with READY flag for this thread)\n"
            "    3. (value & 0x%04X) | 0x%08X (INCLUSIVE state: some value derived from lookback "
            "combined with INCLUSIVE flag for this thread)\n",
            tile_id, tid, got, FLAG_NOT_READY, expected_full_value_for_ready_state,
            val_content_for_ready_state, VALUE_MASK, FLAG_INCLUSIVE);
        return false;
    } else if (errCode == ERROR_TYPE_SHUFFLE_READY) {
        printf(
            "Shuffle Ready error at tile %u, thread %u: GOT 0x%08X (this was 'prev_red' from the "
            "shader during a READY phase).\n"
            "  The expected value for 'prev_red' depends on the specific lookback step (tile_id - "
            "lookback_id) * 1024u.\n",
            tile_id, tid, got);
        return false;
    } else if (errCode == ERROR_TYPE_SHUFFLE_INC) {
        printf("Shuffle Inclusive error at tile %u, thread %u: GOT 0x%08X (this was 'prev_red' "
               "from the "
               "shader during an INCLUSIVE phase).\n"
               "  The expected value for 'prev_red' should be tile_id * 1024u.\n",
               tile_id, tid, got);
        return false;
    } else if (errCode == ERROR_TYPE_SGSIZE) {
        printf("Subgroup Size Mismatch error: Expected BLOCK_DIM (%u), but shader reported sgSize "
               "%u.\n"
               "  Error logged at effective coordinates: tile %u, thread %u (often 0,0 for this "
               "type of global check).\n",
               BLOCK_DIM, got, tile_id, tid);
        return false;
    } else {
        printf("Unknown error code %u detected at tile %u, thread %u: GOT 0x%08X.\n", errCode,
               tile_id, tid, got);
        return false;
    }
}

bool ValidateErrorBuffer(id<MTLCommandQueue> commandQueue, id<MTLBuffer> errorsBuffer,
                         id<MTLBuffer> transferBuffer) {
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    if (commandBuffer == nil) {
        NSLog(@"Failed to create the command buffer for error buffer check.");
        return false;
    }

    id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
    if (blitEncoder == nil) {
        NSLog(@"Failed to create the blit encoder for error buffer check.");
        return false;
    }

    [blitEncoder copyFromBuffer:errorsBuffer
                   sourceOffset:0
                       toBuffer:transferBuffer
              destinationOffset:0
                           size:TEST_SIZE * 4 * sizeof(uint32_t)];
    [blitEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    uint32_t* error_data_ptr = transferBuffer.contents;
    for (uint32_t tile_id = 0; tile_id < TEST_SIZE; ++tile_id) {
        uint32_t index = tile_id * 4;  // Base index for errType (uint2[2]) for this tile_id
        bool passed = true;

        // First thread's error data
        uint32_t errCode0 = error_data_ptr[index + 0];
        uint32_t got_val0 = error_data_ptr[index + 1];
        if (errCode0 != 0) {
            if (!CheckError(errCode0, got_val0, tile_id, 0)) {
                passed = false;
            }
        }

        // Second thread's error data
        uint32_t errCode1 = error_data_ptr[index + 2];
        uint32_t got_val1 = error_data_ptr[index + 3];
        if (errCode1 != 0) {
            if (!CheckError(errCode1, got_val1, tile_id, 1)) {
                passed = false;
            }
        }

        if (!passed) {
            return false;
        }
    }
    return true;
}

void run(uint32_t batchSize) {
    NSError* error = nil;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        NSLog(@"Failed to get default Metal device.");
        return;
    }

    id<MTLComputePipelineState> initPSO = nil;
    id<MTLComputePipelineState> stressPSO = nil;
    if (!SetupPipelineStates(device, &initPSO, &stressPSO, &error)) {
        return;
    }

    id<MTLBuffer> transferBuffer = nil;
    id<MTLBuffer> scanBuffer = nil;
    id<MTLBuffer> scanBumpBuffer = nil;
    id<MTLBuffer> errorsBuffer = nil;
    if (!CreateMetalBuffers(device, &transferBuffer, &scanBuffer, &scanBumpBuffer, &errorsBuffer)) {
        return;
    }

    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    if (commandQueue == nil) {
        NSLog(@"Failed to create the command queue.");
        return;
    }

    MTLSize initGridDim = MTLSizeMake(256, 1, 1);
    MTLSize initBlockDim = MTLSizeMake(256, 1, 1);

    MTLSize stressGridDim = MTLSizeMake(TEST_SIZE, 1, 1);
    MTLSize stressBlockDim = MTLSizeMake(32, 1, 1);  // Exactly equal to simdgroup size

    uint testsPassed = 0;
    for (uint32_t i = 0; i < batchSize; ++i) {
        if (!DispatchKernels(commandQueue, initPSO, stressPSO, scanBumpBuffer, scanBuffer,
                             errorsBuffer, initGridDim, initBlockDim, stressGridDim,
                             stressBlockDim)) {
            NSLog(@"Batch %u: Failed to dispatch kernels.", i + 1);
            return;
        }

        bool validScan = ValidateScanBuffer(commandQueue, scanBuffer, transferBuffer);
        if (!validScan) {
            NSLog(@"Batch %u: Scan buffer validation FAILED.", i + 1);
        }

        bool validErr = ValidateErrorBuffer(commandQueue, errorsBuffer, transferBuffer);
        if (!validErr) {
            NSLog(@"Batch %u: Error buffer check FAILED (errors found and printed).", i + 1);
        }

        testsPassed += validScan && validErr ? 1 : 0;
    }

    printf("%u / %u ", testsPassed, batchSize);
    printf(testsPassed == batchSize ? "ALL TESTS PASSED\n" : "TESTS FAILED\n");
}

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        if (argc != 2) {
            NSLog(@"Usage: %s <batchSize>", argv[0]);
            NSLog(@"batchSize must be a non-negative integer less than 65536.");
            return 1;
        }

        char* endptr;
        errno = 0;
        long batch_val = strtol(argv[1], &endptr, 10);

        if (errno != 0 || endptr == argv[1] || *endptr != '\0' || batch_val < 0 ||
            batch_val >= 65536) {
            NSLog(@"Usage: %s <batchSize>", argv[0]);
            NSLog(@"batchSize must be a non-negative integer less than 65536.");
            return 1;
        }
        run((uint32_t)batch_val);
        NSLog(@"All batches completed.");
    }
    return 0;
}