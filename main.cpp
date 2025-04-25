#include <dawn/webgpu_cpp.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <string>
#include <vector>

struct GPUContext {
    wgpu::Instance instance;
    wgpu::Device device;
    wgpu::Queue queue;
};

struct ComputeShader {
    wgpu::BindGroup bindGroup;
    wgpu::ComputePipeline computePipeline;
    std::string label;
};

struct Shaders {
    ComputeShader init;
    ComputeShader stress;
};

struct GPUBuffers {
    wgpu::Buffer info;
    wgpu::Buffer scanBump;
    wgpu::Buffer scan;
    wgpu::Buffer readback;
    wgpu::Buffer err;
};

void GetGPUContext(GPUContext* context) {
    wgpu::InstanceDescriptor instanceDescriptor{};
    instanceDescriptor.features.timedWaitAnyEnable = true;
    wgpu::Instance instance = wgpu::CreateInstance(&instanceDescriptor);
    if (instance == nullptr) {
        std::cerr << "Instance creation failed!\n";
    }

    wgpu::RequestAdapterOptions options = {};
    options.powerPreference = wgpu::PowerPreference::HighPerformance;
    options.backendType = wgpu::BackendType::Undefined;  // specify as needed

    wgpu::Adapter adapter;
    std::promise<void> adaptPromise;
    instance.RequestAdapter(
        &options, wgpu::CallbackMode::AllowProcessEvents,
        [&](wgpu::RequestAdapterStatus status, wgpu::Adapter adapt, wgpu::StringView) {
            if (status == wgpu::RequestAdapterStatus::Success) {
                adapter = adapt;
            } else {
                std::cerr << "Failed to get adapter" << std::endl;
            }
            adaptPromise.set_value();
        });
    std::future<void> adaptFuture = adaptPromise.get_future();
    while (adaptFuture.wait_for(std::chrono::nanoseconds(100)) == std::future_status::timeout) {
        instance.ProcessEvents();
    }

    wgpu::AdapterInfo info{};
    adapter.GetInfo(&info);
    std::cout << "VendorID: " << std::hex << info.vendorID << std::dec << std::endl;
    std::cout << "Vendor: " << std::string(info.vendor.data, info.vendor.length) << std::endl;
    std::cout << "Architecture: " << std::string(info.architecture.data, info.architecture.length)
              << std::endl;
    std::cout << "DeviceID: " << std::hex << info.deviceID << std::dec << std::endl;
    std::cout << "Name: " << std::string(info.device.data, info.device.length) << std::endl;
    std::cout << "Driver description: "
              << std::string(info.description.data, info.description.length) << std::endl;

    std::vector<wgpu::FeatureName> reqFeatures = {
        wgpu::FeatureName::Subgroups,
    };

    auto errorCallback = [](const wgpu::Device& device, wgpu::ErrorType type,
                            wgpu::StringView message) {
        std::cerr << "Error: " << std::string(message.data, message.length) << std::endl;
    };

    wgpu::DeviceDescriptor devDescriptor{};
    devDescriptor.requiredFeatures = reqFeatures.data();
    devDescriptor.requiredFeatureCount = static_cast<uint32_t>(reqFeatures.size());
    devDescriptor.SetUncapturedErrorCallback(errorCallback);

    wgpu::Device device;
    std::promise<void> devPromise;
    adapter.RequestDevice(
        &devDescriptor, wgpu::CallbackMode::AllowProcessEvents,
        [&](wgpu::RequestDeviceStatus status, wgpu::Device dev, wgpu::StringView) {
            if (status == wgpu::RequestDeviceStatus::Success) {
                device = dev;
            } else {
                std::cerr << "Failed to get device" << std::endl;
            }
            devPromise.set_value();
        });
    std::future<void> devFuture = devPromise.get_future();
    while (devFuture.wait_for(std::chrono::nanoseconds(100)) == std::future_status::timeout) {
        instance.ProcessEvents();
    }
    wgpu::Queue queue = device.GetQueue();

    (*context).instance = instance;
    (*context).device = device;
    (*context).queue = queue;
}

void GetGPUBuffers(const wgpu::Device& device, GPUBuffers* buffs, uint32_t size) {
    wgpu::BufferDescriptor infoDesc = {};
    infoDesc.label = "Info";
    infoDesc.size = sizeof(uint32_t) * 4;
    infoDesc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    wgpu::Buffer info = device.CreateBuffer(&infoDesc);

    wgpu::BufferDescriptor scanBumpDesc = {};
    scanBumpDesc.label = "Scan Atomic Bump";
    scanBumpDesc.size = sizeof(uint32_t);
    scanBumpDesc.usage = wgpu::BufferUsage::Storage;
    wgpu::Buffer scanBump = device.CreateBuffer(&scanBumpDesc);

    wgpu::BufferDescriptor scanDesc = {};
    scanDesc.label = "Scan";
    scanDesc.size = sizeof(uint32_t) * size * 2;
    scanDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    wgpu::Buffer scan = device.CreateBuffer(&scanDesc);

    wgpu::BufferDescriptor readbackDesc = {};
    readbackDesc.label = "Main Readback";
    readbackDesc.size = sizeof(uint32_t) * size * 4;
    readbackDesc.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    wgpu::Buffer readback = device.CreateBuffer(&readbackDesc);

    wgpu::BufferDescriptor errDesc = {};
    errDesc.label = "Error";
    errDesc.size = sizeof(uint32_t) * size * 4;
    errDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    wgpu::Buffer err = device.CreateBuffer(&errDesc);

    (*buffs).info = info;
    (*buffs).scanBump = scanBump;
    (*buffs).scan = scan;
    (*buffs).readback = readback;
    (*buffs).err = err;
}

// For simplicity we will use the same brind group and layout for all kernels
void GetComputeShaderPipeline(const wgpu::Device& device, const GPUBuffers& buffs,
                              ComputeShader* cs, const char* entryPoint,
                              const wgpu::ShaderModule& module, const std::string& csLabel) {
    auto makeLabel = [&](const std::string& suffix) -> std::string { return csLabel + suffix; };

    wgpu::BindGroupLayoutEntry bglInfo = {};
    bglInfo.binding = 0;
    bglInfo.visibility = wgpu::ShaderStage::Compute;
    bglInfo.buffer.type = wgpu::BufferBindingType::Uniform;

    wgpu::BindGroupLayoutEntry bglScanBump = {};
    bglScanBump.binding = 1;
    bglScanBump.visibility = wgpu::ShaderStage::Compute;
    bglScanBump.buffer.type = wgpu::BufferBindingType::Storage;

    wgpu::BindGroupLayoutEntry bglScan = {};
    bglScan.binding = 2;
    bglScan.visibility = wgpu::ShaderStage::Compute;
    bglScan.buffer.type = wgpu::BufferBindingType::Storage;

    wgpu::BindGroupLayoutEntry bglErr = {};
    bglErr.binding = 3;
    bglErr.visibility = wgpu::ShaderStage::Compute;
    bglErr.buffer.type = wgpu::BufferBindingType::Storage;

    std::vector<wgpu::BindGroupLayoutEntry> bglEntries{bglInfo, bglScanBump, bglScan, bglErr};

    wgpu::BindGroupLayoutDescriptor bglDesc = {};
    bglDesc.label = makeLabel("Bind Group Layout").c_str();
    bglDesc.entries = bglEntries.data();
    bglDesc.entryCount = static_cast<uint32_t>(bglEntries.size());
    wgpu::BindGroupLayout bgl = device.CreateBindGroupLayout(&bglDesc);

    wgpu::BindGroupEntry bgInfo = {};
    bgInfo.binding = 0;
    bgInfo.buffer = buffs.info;
    bgInfo.size = buffs.info.GetSize();

    wgpu::BindGroupEntry bgScanBump = {};
    bgScanBump.binding = 1;
    bgScanBump.buffer = buffs.scanBump;
    bgScanBump.size = buffs.scanBump.GetSize();

    wgpu::BindGroupEntry bgScan = {};
    bgScan.binding = 2;
    bgScan.buffer = buffs.scan;
    bgScan.size = buffs.scan.GetSize();

    wgpu::BindGroupEntry bgErr = {};
    bgErr.binding = 3;
    bgErr.buffer = buffs.err;
    bgErr.size = buffs.err.GetSize();

    std::vector<wgpu::BindGroupEntry> bgEntries{bgInfo, bgScan, bgScanBump, bgErr};

    wgpu::BindGroupDescriptor bindGroupDesc = {};
    bindGroupDesc.entries = bgEntries.data();
    bindGroupDesc.entryCount = static_cast<uint32_t>(bgEntries.size());
    bindGroupDesc.layout = bgl;
    wgpu::BindGroup bindGroup = device.CreateBindGroup(&bindGroupDesc);

    wgpu::PipelineLayoutDescriptor pipeLayoutDesc = {};
    pipeLayoutDesc.label = makeLabel("Pipeline Layout").c_str();
    pipeLayoutDesc.bindGroupLayoutCount = 1;
    pipeLayoutDesc.bindGroupLayouts = &bgl;
    wgpu::PipelineLayout pipeLayout = device.CreatePipelineLayout(&pipeLayoutDesc);

    wgpu::ComputeState compState = {};
    compState.entryPoint = entryPoint;
    compState.module = module;

    wgpu::ComputePipelineDescriptor compPipeDesc = {};
    compPipeDesc.label = makeLabel("Compute Pipeline").c_str();
    compPipeDesc.layout = pipeLayout;
    compPipeDesc.compute = compState;
    wgpu::ComputePipeline compPipeline = device.CreateComputePipeline(&compPipeDesc);

    (*cs).bindGroup = bindGroup;
    (*cs).computePipeline = compPipeline;
    (*cs).label = csLabel;
}

std::string ReadWGSL(const std::string& path, const std::vector<std::string>& pseudoArgs) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return "";
    }

    std::stringstream buffer;
    for (size_t i = 0; i < pseudoArgs.size(); ++i) {
        buffer << pseudoArgs[i] << "\n";
    }
    buffer << file.rdbuf();
    file.close();
    return buffer.str();
}

void CreateShaderFromSource(const GPUContext& gpu, const GPUBuffers& buffs, ComputeShader* cs,
                            const char* entryPoint, const std::string& path,
                            const std::string& csLabel,
                            const std::vector<std::string>& pseudoArgs) {
    wgpu::ShaderSourceWGSL wgslSource = {};
    std::string source = ReadWGSL(path, pseudoArgs);
    wgslSource.code = source.c_str();
    wgpu::ShaderModuleDescriptor desc = {};
    desc.nextInChain = &wgslSource;
    wgpu::ShaderModule mod = gpu.device.CreateShaderModule(&desc);
    std::promise<void> promise;
    mod.GetCompilationInfo(
        wgpu::CallbackMode::AllowProcessEvents,
        [&](wgpu::CompilationInfoRequestStatus status, wgpu::CompilationInfo const* info) {
            for (size_t i = 0; i < info->messageCount; ++i) {
                const wgpu::CompilationMessage& message = info->messages[i];
                if (message.type == wgpu::CompilationMessageType::Error) {
                    std::cerr << "Shader compilation error: "
                              << std::string(message.message.data, message.message.length)
                              << std::endl;
                } else if (message.type == wgpu::CompilationMessageType::Warning) {
                    std::cerr << "Shader compilation warning: "
                              << std::string(message.message.data, message.message.length)
                              << std::endl;
                }
            }
            promise.set_value();
        });
    std::future<void> future = promise.get_future();
    while (future.wait_for(std::chrono::nanoseconds(100)) == std::future_status::timeout) {
        gpu.instance.ProcessEvents();
    }
    GetComputeShaderPipeline(gpu.device, buffs, cs, entryPoint, mod, csLabel);
}

void GetAllShaders(const GPUContext& gpu, const GPUBuffers& buffs, Shaders* shaders) {
    std::vector<std::string> empty;
    CreateShaderFromSource(gpu, buffs, &shaders->init, "main", "Shaders/init.wgsl", "Init", empty);
    CreateShaderFromSource(gpu, buffs, &shaders->stress, "main", "Shaders/stress.wgsl", "Stress",
                           empty);
}

void SetComputePass(const ComputeShader& cs, wgpu::CommandEncoder* comEncoder, uint32_t workTiles) {
    wgpu::ComputePassDescriptor comDesc = {};
    comDesc.label = cs.label.c_str();
    wgpu::ComputePassEncoder pass = (*comEncoder).BeginComputePass(&comDesc);
    pass.SetPipeline(cs.computePipeline);
    pass.SetBindGroup(0, cs.bindGroup);
    pass.DispatchWorkgroups(workTiles, 1, 1);
    pass.End();
}

void QueueSync(const GPUContext& gpu) {
    std::promise<void> promise;
    gpu.queue.OnSubmittedWorkDone(wgpu::CallbackMode::AllowProcessEvents,
                                  [&](wgpu::QueueWorkDoneStatus status) {
                                      if (status != wgpu::QueueWorkDoneStatus::Success) {
                                          std::cerr << "Queue submission failed" << std::endl;
                                      }
                                      promise.set_value();
                                  });
    std::future<void> future = promise.get_future();
    while (future.wait_for(std::chrono::nanoseconds(100)) == std::future_status::timeout) {
        gpu.instance.ProcessEvents();
    }
}

void CopyBufferSync(const GPUContext& gpu, wgpu::Buffer* srcReadback, wgpu::Buffer* dstReadback,
                    uint64_t sourceOffsetBytes, uint64_t copySizeBytes) {
    wgpu::CommandEncoderDescriptor comEncDesc = {};
    comEncDesc.label = "Copy Command Encoder";
    wgpu::CommandEncoder comEncoder = gpu.device.CreateCommandEncoder(&comEncDesc);
    comEncoder.CopyBufferToBuffer(*srcReadback, sourceOffsetBytes, *dstReadback, 0ULL,
                                  copySizeBytes);
    wgpu::CommandBuffer comBuffer = comEncoder.Finish();
    gpu.queue.Submit(1, &comBuffer);
    QueueSync(gpu);
}

template <typename T>
void ReadbackSync(const GPUContext& gpu, wgpu::Buffer* dstReadback, std::vector<T>* readOut,
                  uint64_t readbackSizeBytes) {
    std::promise<void> promise;
    dstReadback->MapAsync(
        wgpu::MapMode::Read, 0, readbackSizeBytes, wgpu::CallbackMode::AllowProcessEvents,
        [&](wgpu::MapAsyncStatus status, wgpu::StringView) {
            if (status == wgpu::MapAsyncStatus::Success) {
                const void* data = dstReadback->GetConstMappedRange(0, readbackSizeBytes);
                std::memcpy(readOut->data(), data, readbackSizeBytes);
                dstReadback->Unmap();
            } else {
                std::cerr << "Bad readback" << std::endl;
            }
            promise.set_value();
        });

    std::future<void> future = promise.get_future();
    while (future.wait_for(std::chrono::nanoseconds(100)) == std::future_status::timeout) {
        gpu.instance.ProcessEvents();
    }
}

template <typename T>
void CopyAndReadbackSync(const GPUContext& gpu, wgpu::Buffer* srcReadback,
                         wgpu::Buffer* dstReadback, std::vector<T>* readOut, uint32_t sourceOffset,
                         uint32_t readbackSize) {
    CopyBufferSync(gpu, srcReadback, dstReadback, sourceOffset * sizeof(T),
                   readbackSize * sizeof(T));
    ReadbackSync(gpu, dstReadback, readOut, readbackSize * sizeof(T));
}

void InitializeUniforms(const GPUContext& gpu, GPUBuffers* buffs, uint32_t size) {
    wgpu::CommandEncoderDescriptor comEncDesc = {};
    comEncDesc.label = "Initialize Uniforms Command Encoder";
    wgpu::CommandEncoder comEncoder = gpu.device.CreateCommandEncoder(&comEncDesc);
    std::vector<uint32_t> info{size, 0, 0, 0};
    gpu.queue.WriteBuffer(buffs->info, 0ULL, info.data(), info.size() * sizeof(uint32_t));
    wgpu::CommandBuffer comBuffer = comEncoder.Finish();
    gpu.queue.Submit(0, &comBuffer);
    QueueSync(gpu);
}

const uint32_t FLAG_NOT_READY = 0u;
const uint32_t FLAG_READY = 0x40000000u;
const uint32_t FLAG_INCLUSIVE = 0x80000000u;
const uint32_t VALUE_MASK = 0xFFFFu;
const uint32_t SPLIT_THREADS = 2u;
const uint32_t ERROR_TYPE_MESSAGE = 1u;
const uint32_t ERROR_TYPE_SHUFFLE = 2u;

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
    } else if (errCode == ERROR_TYPE_SHUFFLE) {
        printf(
            "Shuffle error at tile %u, thread %u: GOT 0x%08X (this was 'prev_red' from the "
            "shader).\n"
            "  The expected value for 'prev_red' depends on the specific lookback step and scan "
            "phase.\n"
            "  It should typically be of the form (tile_id - N) * 1024u (where N is related to "
            "lookback_id or 1 for inclusive end).\n",
            tile_id, tid, got);
        return false;
    } else {
        printf("Unknown error code %u detected at tile %u, thread %u: GOT 0x%08X.\n", errCode,
               tile_id, tid, got);
        return false;
    }
}

bool Validate(const GPUContext& gpu, GPUBuffers* buffs, uint32_t size) {
    if (size == 0) {
        return true;
    }
    std::vector<uint32_t> readOut(size * SPLIT_THREADS * 2);
    CopyAndReadbackSync(gpu, &buffs->err, &buffs->readback, &readOut, 0, readOut.size());
    for (uint32_t tile_id = 0; tile_id < size; ++tile_id) {
        uint32_t t = tile_id * SPLIT_THREADS * 2;
        for (uint32_t tid = 0; tid < SPLIT_THREADS; ++tid) {
            uint32_t errCode = readOut[t + tid * 2];
            uint32_t got_val = readOut[t + tid * 2 + 1];
            if (!CheckError(errCode, got_val, tile_id, tid)) {
                return false;
            }
        }
    }
    return true;
}

template <uint32_t lineLength>
void ReadbackAndPrintSync(const GPUContext& gpu, GPUBuffers* buffs, uint32_t readbackSize) {
    std::vector<uint32_t> readOut(readbackSize * 2);
    CopyAndReadbackSync(gpu, &buffs->scan, &buffs->readback, &readOut, 0, readbackSize * 2);
    for (uint32_t i = 0; i < (readbackSize + lineLength - 1) / lineLength; ++i) {
        for (uint32_t k = 0; k < lineLength; ++k) {
            uint32_t index = i * lineLength + k;
            if (index < readbackSize) {
                uint32_t rejoin = readOut[index * 2] & 65535 | readOut[index * 2 + 1] << 16;
                std::cout << rejoin / 1024 << ", ";
            }
        }
        std::cout << std::endl;
    }
}

void RunTest(uint32_t size, uint32_t batchSize, GPUBuffers& buffers, const GPUContext& gpu,
             const Shaders& shaders) {
    uint32_t testsPassed = 0;
    for (uint32_t i = 0; i < batchSize; ++i) {
        wgpu::CommandEncoderDescriptor comEncDesc = {};
        comEncDesc.label = "Command Encoder";
        wgpu::CommandEncoder comEncoder = gpu.device.CreateCommandEncoder(&comEncDesc);

        SetComputePass(shaders.init, &comEncoder, 256);
        SetComputePass(shaders.stress, &comEncoder, size);

        wgpu::CommandBuffer comBuffer = comEncoder.Finish();
        gpu.queue.Submit(1, &comBuffer);
        QueueSync(gpu);
        testsPassed += Validate(gpu, &buffers, size) ? 1 : 0;
        // ReadbackAndPrintSync<10>(gpu, &buffers, size);
    }

    printf("%u / %u", testsPassed, batchSize);
    printf(testsPassed == batchSize ? " ALL TESTS PASSED\n" : " TEST FAILED");
}

bool parse(const char* arg_str, uint32_t exclMax, uint32_t& o) {
    char* endptr;
    errno = 0;
    unsigned long t = std::strtoul(arg_str, &endptr, 10);
    if (errno == ERANGE || endptr == arg_str || *endptr != '\0' || t > exclMax) {
        return false;
    }
    o = static_cast<uint32_t>(t);
    return true;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <test size> <number of tests to run>" << std::endl;
        return EXIT_FAILURE;
    }

    uint32_t size;
    if (!parse(argv[1], 65535, size)) {
        std::cerr << "Expected uint32_t less than 65536 for maximum test size." << std::endl;
        return EXIT_FAILURE;
    }

    uint32_t batchSize;
    if (!parse(argv[2], 1023, batchSize)) {
        std::cerr << "Expected uint32_t less than 1024 for number of tests to run." << std::endl;
        return EXIT_FAILURE;
    }

    GPUContext gpu;
    GPUBuffers buffs;
    Shaders shaders;
    GetGPUContext(&gpu);
    GetGPUBuffers(gpu.device, &buffs, size);
    GetAllShaders(gpu, buffs, &shaders);
    InitializeUniforms(gpu, &buffs, size);
    RunTest(size, batchSize, buffs, gpu, shaders);

    return EXIT_SUCCESS;
}
