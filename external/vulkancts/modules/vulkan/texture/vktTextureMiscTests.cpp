/*------------------------------------------------------------------------
 * Vulkan Conformance Tests
 * ------------------------
 *
 * Copyright (c) 2025 The Khronos Group Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *//*!
* \file
* \brief Texture misc tests.
*//*--------------------------------------------------------------------*/

#include "vktTextureMiscTests.hpp"
#include "vktTestCase.hpp"
#include "vktTestCaseDefs.hpp"
#include "vktTestGroupUtil.hpp"
#include "vktTextureTestUtil.hpp"

#include "vkBarrierUtil.hpp"
#include "vkBufferWithMemory.hpp"
#include "vkBuilderUtil.hpp"
#include "vkCmdUtil.hpp"
#include "vkDefs.hpp"
#include "vkImageUtil.hpp"

#include "tcuTexture.hpp"
#include "tcuVectorType.hpp"
#include "tcuVectorUtil.hpp"
#include "tcuImageCompare.hpp"
#include "tcuTextureUtil.hpp"

#include "deDefs.h"
#include "deMath.h"
#include "deRandom.hpp"
#include "deStringUtil.hpp"

#include <cstdint>
#include <vector>

namespace vkt
{
namespace texture
{
namespace
{

using namespace vk;
using namespace tcu;

const uint32_t kNumRegions      = 3u;
const uint32_t kTexelsPerRegion = 4u;

enum class TexelBufferType
{
    TEXEL_BUFF_UNIFORM = 0,
    TEXEL_BUFF_STORAGE,
};

struct MaxTextureElementsTestParams
{
    TexelBufferType bufferType;
    VkFormat format;
};

class MaxTextureElementsTestInstance : public TestInstance
{
public:
    MaxTextureElementsTestInstance(Context &ctx, MaxTextureElementsTestParams params);
    virtual ~MaxTextureElementsTestInstance(void);
    TestStatus iterate(void);

private:
    void initBuffer(const BufferWithMemory &colorBuffer, const uint32_t size, const VkFormat format);

private:
    MaxTextureElementsTestParams m_params;
};

MaxTextureElementsTestInstance::MaxTextureElementsTestInstance(Context &ctx, MaxTextureElementsTestParams params)
    : TestInstance(ctx)
    , m_params(params)
{
}

MaxTextureElementsTestInstance::~MaxTextureElementsTestInstance(void)
{
}
class MaxTextureElementsTest : public TestCase
{
public:
    MaxTextureElementsTest(TestContext &ctx, const std::string &name, MaxTextureElementsTestParams params);
    virtual ~MaxTextureElementsTest(void);
    virtual std::string getRequiredCapabilitiesId() const
    {
        return typeid(MaxTextureElementsTest).name();
    }
    virtual void initDeviceCapabilities(DevCaps &caps);
    virtual void checkSupport(Context &context) const;
    virtual void initPrograms(SourceCollections &programCollection) const;
    virtual TestInstance *createInstance(Context &context) const;

private:
    MaxTextureElementsTestParams m_params;
};

MaxTextureElementsTest::MaxTextureElementsTest(TestContext &ctx, const std::string &name,
                                               MaxTextureElementsTestParams params)
    : TestCase(ctx, name.c_str())
    , m_params(params)
{
}

MaxTextureElementsTest::~MaxTextureElementsTest(void)
{
}

void MaxTextureElementsTest::initDeviceCapabilities(DevCaps &caps)
{
    caps.addExtension("VK_EXT_robustness2");
    caps.addExtension("VK_EXT_shader_image_atomic_int64");
    caps.addFeature(&VkPhysicalDeviceRobustness2FeaturesEXT::robustBufferAccess2);
    caps.addFeature(&VkPhysicalDeviceFeatures::robustBufferAccess);
    caps.addFeature(&VkPhysicalDeviceFeatures::shaderInt64);
    caps.addFeature(&VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT::shaderImageInt64Atomics);
}

void MaxTextureElementsTest::checkSupport(Context &context) const
{
    const auto &vki           = context.getInstanceInterface();
    const auto physicalDevice = context.getPhysicalDevice();

    const VkFormatProperties formatProperties = getPhysicalDeviceFormatProperties(vki, physicalDevice, m_params.format);

    if ((m_params.bufferType == TexelBufferType::TEXEL_BUFF_UNIFORM) &&
        !(formatProperties.bufferFeatures & VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT))
        TCU_THROW(NotSupportedError, "Format not supported for uniform texel buffers");

    if ((m_params.bufferType == TexelBufferType::TEXEL_BUFF_STORAGE) &&
        !(formatProperties.bufferFeatures & VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT))
        TCU_THROW(NotSupportedError, "Format not supported for storage texel buffers");

    {
        if (!context.isDeviceFunctionalitySupported("VK_KHR_robustness2") &&
            !context.isDeviceFunctionalitySupported("VK_EXT_robustness2"))

            TCU_THROW(NotSupportedError, "VK_KHR_robustness2 and VK_EXT_robustness2 are not supported");

        VkPhysicalDeviceRobustness2FeaturesEXT robustness2Features = initVulkanStructure();
        VkPhysicalDeviceFeatures2 features2                        = initVulkanStructure(&robustness2Features);

        context.getInstanceInterface().getPhysicalDeviceFeatures2(context.getPhysicalDevice(), &features2);

        if (robustness2Features.robustBufferAccess2 == false)
            TCU_THROW(NotSupportedError, "robustBufferAccess2 not supported");
    }

    if (is64BitIntegerFormat(m_params.format))
    {
        context.requireDeviceFunctionality("VK_EXT_shader_image_atomic_int64");
        if (!context.getDeviceFeatures().shaderInt64 || !context.getShaderAtomicInt64Features().shaderBufferInt64Atomics)
            TCU_THROW(NotSupportedError, "64-bit integers not supported in shaders");
    }
}

const std::string getFormatQualifier(const VkFormat format)
{
    switch (format)
    {
    case VK_FORMAT_R8_UINT:
        return "r8ui";
    case VK_FORMAT_R32_UINT:
        return "r32ui";
    case VK_FORMAT_R32G32B32A32_UINT:
        return "rgba32ui";
    case VK_FORMAT_R8G8B8A8_UINT:
        return "rgba8ui";
    case VK_FORMAT_R16G16B16A16_UINT:
        return "rgba16ui";
    case VK_FORMAT_R64_UINT:
        return "r64ui";
    case VK_FORMAT_R64G64_UINT:
        return "rg64ui";
    case VK_FORMAT_R64G64B64_UINT:
        return "rgb64ui";
    case VK_FORMAT_R64G64B64A64_UINT:
        return "rgba64ui";
    default:
        DE_ASSERT(false);
    }

    return "";
}

std::string getElementFormatStr(const int numComponents, const bool isUint, const bool isSint, const bool is64)
{
    std::ostringstream str;
    if (numComponents == 1)
    {
        str << (isUint ? "uint" : isSint ? "int" : "float");
        if ((isUint || isSint) && is64)
            str << "64_t";
    }
    else
        str << (isUint ? "u" : isSint ? "i" : "") << "vec" << (is64 ? "64" : "") << numComponents;

    return str.str();
}

void MaxTextureElementsTest::initPrograms(SourceCollections &programCollection) const
{
    const char *const versionDecl   = glu::getGLSLVersionDeclaration(glu::GLSL_VERSION_450);
    const bool isTexelUniformBuffer = (m_params.bufferType == TexelBufferType::TEXEL_BUFF_UNIFORM);
    const bool is64BitFormat        = is64BitIntegerFormat(m_params.format);
    const bool isUnormFmt           = isUnormFormat(m_params.format);

    const std::string signStr       = (isUnormFmt || isIntFormat(m_params.format)) ? "" : "u";
    const std::string bufferStr     = signStr + (isTexelUniformBuffer ? "textureBuffer"
        : (de::toString(is64BitFormat ? "64" : "") + "imageBuffer"));
    const std::string bufferFmtStr  = isTexelUniformBuffer ? "" : (", " + getFormatQualifier(m_params.format));
    const std::string bufferOpStr   = isTexelUniformBuffer ? "texelFetch" : "imageLoad";

    const int numComponents         = getNumUsedChannels(mapVkFormat(m_params.format).order);
    const bool isUint               = isUintFormat(m_params.format);
    const bool isSint               = isIntFormat(m_params.format);
    const std::string elemTypeStr   = getElementFormatStr(numComponents, isUint, isSint, is64BitFormat);
    const std::string resTypeStr    = signStr + (is64BitFormat ? "64vec4" : "vec4");
    const bool fmtHasAlpha          = hasAlphaChannel(mapVkFormat(m_params.format).order);
    const std::string alphaValue    = fmtHasAlpha ? "0" : "1";
    const std::string cmpValue   = resTypeStr + "(0, 0, 0, " + alphaValue + ")";

    std::ostringstream prog;

    prog << versionDecl << "\n";
    prog << "#extension GL_EXT_debug_printf : enable\n";
    if (is64BitFormat)
    {
        prog << "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require\n"
             << "#extension GL_EXT_shader_image_int64 : require\n";
    }

    prog << "layout(set = 0, binding = 0" << bufferFmtStr << ") uniform " << bufferStr << " texelBuffer;\n"
         << "layout(set = 0, binding = 1, std430) buffer OutputBuffer\n"
         << "{ \n"
         << "    uint buff[]; \n"
         << "} sb;\n"
         << "layout(push_constant, std430) uniform PushConstants\n"
         << "{\n"
         << "    uint workOffset;\n"
         << "    uint indices[" << kNumRegions << "];\n"
         << "} pc;\n"
         << "layout(local_size_x = 1) in;\n"
         << "void main (void)\n"
         << "{\n"
         << "    uint index = gl_WorkGroupID.x + pc.workOffset;\n"
         << "    bool ok = true;\n"
         << "    " << resTypeStr << " value = " << bufferOpStr << "(texelBuffer, int(index));\n"
         << "    for (uint idx = 0; idx < " << kNumRegions << "; idx++)\n"
         << "    {\n"
         << "        if ((index >= pc.indices[idx]) && (index < (pc.indices[idx] + " << kTexelsPerRegion << ")))\n"
         << "        {\n"
         << "            uint offset = index - pc.indices[idx];\n"
         << "            uint outIndex = (idx * " << kTexelsPerRegion << ") + offset;\n"
         << "            if (idx == " << kNumRegions - 1 << ")\n"
         << "            {\n"
         << "                ok = (value == " << cmpValue << ");\n"
        //  << "	             debugPrintfEXT(\"value at %d is %d,%d,%d,%d ok is %d\\n\", int(index), int(value.r), int(value.g), int(value.b), int(value.a), int(ok));\n"
        //  << "                sb.buff[outIndex] = ((ok == true) ? 1u : 3u);\n"
        //  << "	             debugPrintfEXT(\"value at %d is %d\\n\", int(outIndex), int(sb.buff[outIndex]));\n"
         << "            }\n"
         << "            else {\n"
         << "                ok = (value != " << cmpValue << ");\n"
        //  << "                sb.buff[outIndex] = ((ok == true) ? 1u : 3u);\n"
         << "            }\n"
         << "            sb.buff[outIndex] = ((ok == true) ? 1u : 0u);\n"
         << "        }\n"
         << "    }\n"
         << "}\n";

    programCollection.glslSources.add("comp") << glu::ComputeSource(prog.str()) << ShaderBuildOptions(programCollection.usedVulkanVersion, is64BitFormat ? vk::SPIRV_VERSION_1_3 : vk::SPIRV_VERSION_1_0, 0u, true);
}

TestInstance *MaxTextureElementsTest::createInstance(Context &context) const
{
    return new MaxTextureElementsTestInstance(context, m_params);
}

void MaxTextureElementsTestInstance::initBuffer(const BufferWithMemory &colorBuffer, const uint32_t size,
                                                const VkFormat format)
{
    const auto &vkd   = m_context.getDeviceInterface();
    const auto device = m_context.getDevice();

    auto &colorBufferAlloc = colorBuffer.getAllocation();
    auto colorBufferPtr    = reinterpret_cast<char *>(colorBufferAlloc.getHostPtr()) + colorBufferAlloc.getOffset();
    const PixelBufferAccess colorBufferPixels {mapVkFormat(format), static_cast<int>(size), 1, 1, colorBufferPtr};
    // tcu::clear(colorBufferPixels, Vec4(0.0f, 0.0f, 0.0f, 0.0f));

    de::Random rnd(1234);
    const float channelValue = rnd.getFloat(0.5f, 1.0f); // no zeros
    const Vec4 color         = Vec4(channelValue, channelValue, channelValue, channelValue);

    for (uint32_t x = 0; x < size; ++x)
    {
        colorBufferPixels.setPixel(color, x, 0, 0);
    }

    flushAlloc(vkd, device, colorBufferAlloc);
}

TestStatus MaxTextureElementsTestInstance::iterate(void)
{
    const auto &vki             = m_context.getInstanceInterface();
    const auto physicalDevice   = m_context.getPhysicalDevice();
    const auto &vkd             = m_context.getDeviceInterface();
    const auto device           = m_context.getDevice();
    auto &allocator             = m_context.getDefaultAllocator();
    const auto queue            = m_context.getUniversalQueue();
    const auto queueFamilyIndex = m_context.getUniversalQueueFamilyIndex();
    const auto deviceProps      = getPhysicalDeviceProperties(vki, physicalDevice);

    // Create max sized texel buffer
    const uint32_t texelSize               = getPixelSize(mapVkFormat(m_params.format));
    const uint32_t maxDeviceTexels         = deviceProps.limits.maxTexelBufferElements;
    const uint32_t safetyNet               = 32u * (kTexelsPerRegion * texelSize); // allocation range must not overlap with end of texel address range
    const VkDeviceSize maxMemory           = m_context.getDeviceVulkan11Properties().maxMemoryAllocationSize;
    const VkDeviceSize maxDeviceAllocation = maxMemory - safetyNet;
    VkDeviceSize maxClampedTexels          = de::min(static_cast<VkDeviceSize>(maxDeviceTexels), maxDeviceAllocation / texelSize);

#ifndef CTS_USES_VULKANSC
    if (m_context.isDeviceFunctionalitySupported("VK_KHR_maintenance4"))
    {
        const VkDeviceSize maxDeviceBuffer = m_context.getMaintenance4Properties().maxBufferSize;
        maxClampedTexels                   = de::min(maxClampedTexels, maxDeviceBuffer / texelSize);
    }
#endif

    const uint32_t maxTexels        = static_cast<uint32_t>(maxClampedTexels) - kTexelsPerRegion; // maxTexelBufferElements is uint32_t
    const VkDeviceSize bufferSize   = maxTexels * texelSize;
    const bool isTexelUniformBuffer = (m_params.bufferType == TexelBufferType::TEXEL_BUFF_UNIFORM);
    const VkBufferUsageFlags bufferUsageFlags =
        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        (isTexelUniformBuffer ? VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT : VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT);

    const VkBufferCreateInfo &bufferCreateInfo = makeBufferCreateInfo(bufferSize, bufferUsageFlags);
    const Unique<VkBuffer> texelBuffer(createBuffer(vkd, device, &bufferCreateInfo));
    const VkMemoryRequirements bufferMemRequirements = getBufferMemoryRequirements(vkd, device, *texelBuffer);

    if ((bufferMemRequirements.size > maxMemory)
#ifndef CTS_USES_VULKANSC
        || ((m_context.isDeviceFunctionalitySupported("VK_KHR_maintenance4")) && (bufferMemRequirements.size > m_context.getMaintenance4Properties().maxBufferSize))
#endif
    )
        TCU_THROW(NotSupportedError, "Required memory size for maximum sized texel buffer exceeds device limits");

    const de::UniquePtr<Allocation> texelBufferMemory(allocator.allocate(bufferMemRequirements, MemoryRequirement::Any));
    VK_CHECK(vkd.bindBufferMemory(device, *texelBuffer, texelBufferMemory->getMemory(), texelBufferMemory->getOffset()));

    // Create a texel buffer view
    Move<VkBufferView> texelBufferView = makeBufferView(vkd, device, *texelBuffer, m_params.format, 0u, bufferSize);

    // Create input buffer
    const uint32_t inTexels     = kTexelsPerRegion * kNumRegions;
    const uint32_t inBufferSize = inTexels * texelSize;
    BufferWithMemory inputBuffer{
        vkd, device, allocator,
        makeBufferCreateInfo(inBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT),
        MemoryRequirement::HostVisible};

    // Initialize texels
    initBuffer(inputBuffer, inTexels, m_params.format);

    // Create output buffer
    const VkDeviceSize outBufferSize = inTexels * sizeof(uint32_t);
    BufferWithMemory outputBuffer{vkd, device, allocator,
                                  makeBufferCreateInfo(outBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
                                  MemoryRequirement::HostVisible};

    // Initialize output buffer with 0 values
    {
        const auto &outBufferAlloc = outputBuffer.getAllocation();
        invalidateAlloc(vkd, device, outBufferAlloc);

        auto outBufferPtr = reinterpret_cast<uint32_t *>(outBufferAlloc.getHostPtr()) + outBufferAlloc.getOffset();

        for (uint32_t outIdx = 0u; outIdx < inTexels; outIdx++)
           outBufferPtr[outIdx] = 0u;


        flushAlloc(vkd, device, outBufferAlloc);
    }

    const Unique<VkCommandPool> commandPool(makeCommandPool(vkd, device, queueFamilyIndex));
    const Unique<VkCommandBuffer> commandBuffer(
        allocateCommandBuffer(vkd, device, *commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY));

    beginCommandBuffer(vkd, *commandBuffer);

    // Create barrier to update input/output before being read and written respectively
    {
        const VkBufferMemoryBarrier inputBufferBarrier = makeBufferMemoryBarrier(
            VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, *inputBuffer, 0ull, inBufferSize);

        vkd.cmdPipelineBarrier(*commandBuffer, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0u, 0u,
                               nullptr, 1u, &inputBufferBarrier, 0u, nullptr);

        const VkBufferMemoryBarrier outputBufferBarrier = makeBufferMemoryBarrier(
            VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT, *outputBuffer, 0ull, outBufferSize);

        vkd.cmdPipelineBarrier(*commandBuffer, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u,
                               nullptr, 1u, &outputBufferBarrier, 0u, nullptr);
    }

    // Selected texel indices to check
    const uint32_t startTexelIdx = 0u; // start of allocated region
    const uint32_t midTexelIdx   = deFloorToInt32(static_cast<float>(maxTexels) / 2.0f); // middle of allocated region

    const uint32_t endTexelIdx   = maxTexels - kTexelsPerRegion; // end of allocated memory range
    const uint32_t initTexelIndices[] = {startTexelIdx, midTexelIdx, endTexelIdx}; // indices = kNumRegions = 3

    const uint32_t endAddrTexelIdx    = maxDeviceTexels - kTexelsPerRegion + 1; // end of texel addressable range
    const uint32_t texelIndices[]     = {startTexelIdx, midTexelIdx, endAddrTexelIdx}; // indices = kNumRegions = 3

    // Create regions to upload
    std::vector<VkBufferCopy> copyRegions;
    copyRegions.resize(kNumRegions);

    {
        const uint32_t regionSize = kTexelsPerRegion * texelSize;

        for (uint32_t rIdx = 0u; rIdx < kNumRegions; rIdx++)
            copyRegions[rIdx] = makeBufferCopy(rIdx * regionSize, initTexelIndices[rIdx] * texelSize, regionSize);
    }

    vkd.cmdCopyBuffer(*commandBuffer, *inputBuffer, *texelBuffer, kNumRegions, de::dataOrNull(copyRegions));

    // Now bind the buffer as a uniform/storage texel buffer in a descriptor set
    VkDescriptorType descriptorType =
        (isTexelUniformBuffer ? VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER);

    // Create descriptor set
    const Unique<VkDescriptorSetLayout> descriptorSetLayout(
        DescriptorSetLayoutBuilder()
            .addSingleBinding(descriptorType, VK_SHADER_STAGE_COMPUTE_BIT)
            .addSingleBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
            .build(vkd, device));

    // Push constants
    struct TestPushConstants
    {
        uint32_t workOffset;
        uint32_t indices[3]; // indices = kNumRegions = 3
    };

    // Push constant range
    const VkPushConstantRange pcRange = {
        VK_SHADER_STAGE_COMPUTE_BIT,                      // VkShaderStageFlags stageFlags;
        0u,                                               // uint32_t offset;
        static_cast<uint32_t>(sizeof(TestPushConstants)), // uint32_t size;
    };

    // Create compute pipeline
    const Unique<VkShaderModule> shaderModule(
        createShaderModule(vkd, device, m_context.getBinaryCollection().get("comp"), 0));
    const Unique<VkPipelineLayout> pipelineLayout(makePipelineLayout(vkd, device, *descriptorSetLayout, &pcRange));
    const Unique<VkPipeline> computePipeline(makeComputePipeline(vkd, device, *pipelineLayout, *shaderModule));

    vkd.cmdBindPipeline(*commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, *computePipeline);

    const Unique<VkDescriptorPool> descriptorPool(
        DescriptorPoolBuilder()
            .addType(descriptorType, 1u)
            .addType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u)
            .build(vkd, device, VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT, 1u));

    const Unique<VkDescriptorSet> descriptorSet(makeDescriptorSet(vkd, device, *descriptorPool, *descriptorSetLayout));

    const VkDescriptorBufferInfo outputBufferInfo = makeDescriptorBufferInfo(*outputBuffer, 0ull, outBufferSize);

    DescriptorSetUpdateBuilder()
        .writeSingle(*descriptorSet, DescriptorSetUpdateBuilder::Location::binding(0u), descriptorType,
                     &texelBufferView.get())
        .writeSingle(*descriptorSet, DescriptorSetUpdateBuilder::Location::binding(1u),
                     VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &outputBufferInfo)
        .update(vkd, device);

    vkd.cmdBindDescriptorSets(*commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, *pipelineLayout, 0u, 1u,
                              &descriptorSet.get(), 0u, nullptr);

    {
        const VkBufferMemoryBarrier texelBufferBarrier = makeBufferMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, *texelBuffer, 0ull, bufferSize);

        vkd.cmdPipelineBarrier(*commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u,
                               0u, nullptr, 1u, &texelBufferBarrier, 0u, nullptr);
    }

    uint32_t dispatchWorkGroups  = deviceProps.limits.maxComputeWorkGroupCount[0];
    uint32_t workOffset          = 0u;

    for (uint32_t remainingWorkgroups = maxDeviceTexels + 1; remainingWorkgroups > 0; remainingWorkgroups -= dispatchWorkGroups)
    {
        if (remainingWorkgroups <= dispatchWorkGroups)
            dispatchWorkGroups = remainingWorkgroups;

        {
            struct TestPushConstants pushConsts = {workOffset, {texelIndices[0], texelIndices[1], texelIndices[2]}};

            vkd.cmdPushConstants(*commandBuffer, *pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0u,
                                static_cast<uint32_t>(sizeof(pushConsts)), &pushConsts);
        }

        vkd.cmdDispatch(*commandBuffer, dispatchWorkGroups, 1u, 1u);

        workOffset += dispatchWorkGroups;
    }

    {
        const VkBufferMemoryBarrier outputBufferBarrier = makeBufferMemoryBarrier(
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT, *outputBuffer, 0ull, outBufferSize);

        vkd.cmdPipelineBarrier(*commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0u, 0u,
                               nullptr, 1u, &outputBufferBarrier, 0u, nullptr);
    }

    endCommandBuffer(vkd, *commandBuffer);

    submitCommandsAndWait(vkd, device, queue, *commandBuffer);
    // vkd.deviceWaitIdle(device);

    // Retrieve data from output buffer to host memory
    {
        const auto &outBufferAlloc = outputBuffer.getAllocation();
        invalidateAlloc(vkd, device, outBufferAlloc);

        auto outBufferPtr = reinterpret_cast<const uint32_t *>(outBufferAlloc.getHostPtr()) + outBufferAlloc.getOffset();

        for (uint32_t outIdx = 0; outIdx < inTexels; outIdx++)
        {
            uint32_t result = outBufferPtr[outIdx];

            if (result != 1u)
                return tcu::TestStatus::fail("Fail");
        }
    }

    return tcu::TestStatus::pass("Pass");
}

} // namespace

void addMiscTests(tcu::TestCaseGroup *miscTests)
{
    TestContext &testCtx = miscTests->getTestContext();

    {
        de::MovePtr<tcu::TestCaseGroup> maxElementsGroup(new tcu::TestCaseGroup(testCtx, "max_elements"));

        std::vector<MaxTextureElementsTestParams> params = {
            {TexelBufferType::TEXEL_BUFF_UNIFORM, VK_FORMAT_R8_UINT},
            {TexelBufferType::TEXEL_BUFF_UNIFORM, VK_FORMAT_R32_UINT},
            {TexelBufferType::TEXEL_BUFF_UNIFORM, VK_FORMAT_R32G32B32A32_UINT},
            {TexelBufferType::TEXEL_BUFF_UNIFORM, VK_FORMAT_R8G8B8A8_UNORM},
            {TexelBufferType::TEXEL_BUFF_UNIFORM, VK_FORMAT_R8G8B8A8_UINT},
            {TexelBufferType::TEXEL_BUFF_UNIFORM, VK_FORMAT_R16G16B16A16_UINT},
            {TexelBufferType::TEXEL_BUFF_UNIFORM, VK_FORMAT_R64_UINT},
            // {TexelBufferType::TEXEL_BUFF_UNIFORM, VK_FORMAT_R64G64_UINT},
            // {TexelBufferType::TEXEL_BUFF_UNIFORM, VK_FORMAT_R64G64B64_UINT},
            // {TexelBufferType::TEXEL_BUFF_UNIFORM, VK_FORMAT_R64G64B64A64_UINT},

            {TexelBufferType::TEXEL_BUFF_STORAGE, VK_FORMAT_R8_UINT},
            {TexelBufferType::TEXEL_BUFF_STORAGE, VK_FORMAT_R32_UINT},
            {TexelBufferType::TEXEL_BUFF_STORAGE, VK_FORMAT_R32G32B32A32_UINT},
            {TexelBufferType::TEXEL_BUFF_STORAGE, VK_FORMAT_R8G8B8A8_UINT},
            {TexelBufferType::TEXEL_BUFF_STORAGE, VK_FORMAT_R16G16B16A16_UINT},
            {TexelBufferType::TEXEL_BUFF_STORAGE, VK_FORMAT_R64_UINT},
            {TexelBufferType::TEXEL_BUFF_STORAGE, VK_FORMAT_R64G64_UINT},
            {TexelBufferType::TEXEL_BUFF_STORAGE, VK_FORMAT_R64G64B64_UINT},
            {TexelBufferType::TEXEL_BUFF_STORAGE, VK_FORMAT_R64G64B64A64_UINT},
        };

        const std::string texelBufferNames[] = {"uniform_texel_buffer", "storage_texel_buffer"};

        for (uint32_t paramIdx = 0; paramIdx < de::sizeU32(params); paramIdx++)
        {
            const auto texelBufferTypeName = texelBufferNames[static_cast<uint32_t>(params[paramIdx].bufferType)];
            const auto formatName = de::toLower(std::string(getFormatName(params[paramIdx].format)).substr(10));

            const std::string testName = texelBufferTypeName + "_" + formatName;

            maxElementsGroup->addChild(new MaxTextureElementsTest(testCtx, testName, params[paramIdx]));
        }

        miscTests->addChild(maxElementsGroup.release());
    }
}

tcu::TestCaseGroup *createTextureMiscTests(tcu::TestContext &testCtx)
{
    return createTestGroup(testCtx, "misc", addMiscTests);
}

} // namespace texture
} // namespace vkt
