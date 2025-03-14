/*------------------------------------------------------------------------
 * Vulkan Conformance Tests
 * ------------------------
 *
 * Copyright (c) 2017 The Khronos Group Inc.
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
 * \brief Tests for mutable images
 *//*--------------------------------------------------------------------*/

#include "vktImageMutableTests.hpp"
#include "vktImageLoadStoreUtil.hpp"
#include "vktTestCaseUtil.hpp"
#include "vktImageTexture.hpp"
#include "vktCustomInstancesDevices.hpp"

#include "vkBuilderUtil.hpp"
#include "vkQueryUtil.hpp"
#include "vkImageUtil.hpp"
#include "vkCmdUtil.hpp"
#include "vkObjUtil.hpp"
#include "vkRef.hpp"
#include "vkDefs.hpp"
#include "vkPlatform.hpp"
#include "vkWsiUtil.hpp"
#include "vkDeviceUtil.hpp"
#include "vkSafetyCriticalUtil.hpp"

#include "deUniquePtr.hpp"
#include "deSharedPtr.hpp"

#include "tcuImageCompare.hpp"
#include "tcuTestLog.hpp"
#include "tcuTextureUtil.hpp"
#include "tcuPlatform.hpp"
#include "tcuCommandLine.hpp"

#include <string>
#include <vector>

using namespace vk;
using namespace tcu;
using namespace vk::wsi;

using de::MovePtr;
using de::SharedPtr;
using de::UniquePtr;
using std::string;
using std::vector;

namespace vkt
{
namespace image
{

typedef SharedPtr<Unique<VkPipeline>> SharedPtrVkPipeline;
typedef SharedPtr<Unique<VkImageView>> SharedPtrVkImageView;

template <typename T>
inline SharedPtr<Unique<T>> makeSharedPtr(Move<T> move)
{
    return SharedPtr<Unique<T>>(new Unique<T>(move));
}

enum Upload
{
    UPLOAD_CLEAR = 0,
    UPLOAD_COPY,
    UPLOAD_STORE,
    UPLOAD_DRAW,
    UPLOAD_LAST
};

enum Download
{
    DOWNLOAD_COPY = 0,
    DOWNLOAD_LOAD,
    DOWNLOAD_TEXTURE,
    DOWNLOAD_LAST
};

std::string getUploadString(const int upload)
{
    const char *strs[] = {"clear", "copy", "store", "draw"};
    return strs[upload];
}

std::string getDownloadString(const int download)
{
    const char *strs[] = {"copy", "load", "texture"};
    return strs[download];
}

enum ResolveAttachmentTestType
{
    RA_TEST_NONE        = 0, // Not a resolve attachment test
    RA_TEST_ALL_MUTABLE = 1, // All attachments are mutable
    RA_TEST_RA_MUTABLE  = 2, // Only resolve attachment is mutable, mutisampled color attachment is non-mutable
    RA_TEST_CA_MUTABLE  = 3, // Only mutisampled color attachment is mutable, resolve attachment is non-mutable
};

struct CaseDef
{
    ImageType imageType;
    IVec3 size;
    uint32_t numLayers;
    VkFormat imageFormat;
    VkFormat viewFormat;
    enum Upload upload;
    enum Download download;
    bool isFormatListTest;
    Type wsiType;
    ResolveAttachmentTestType resolveAttachmentTestType;
    bool isLoadOpClearTest;
};

static const uint32_t COLOR_TABLE_SIZE = 4;

// Reference color values for float color rendering. Values have been chosen
// so that when the bit patterns are reinterpreted as a 16-bit float, we do not
// run into NaN / inf / denorm values.
static const Vec4 COLOR_TABLE_FLOAT[COLOR_TABLE_SIZE] = {
    Vec4(0.00f, 0.40f, 0.80f, 0.10f),
    Vec4(0.50f, 0.10f, 0.90f, 0.20f),
    Vec4(0.20f, 0.60f, 1.00f, 0.30f),
    Vec4(0.30f, 0.70f, 0.00f, 0.40f),
};

// Reference color values for integer color rendering. We avoid negative
// values (even for SINT formats) to avoid the situation where sign extension
// leads to NaN / inf values when they are reinterpreted with a float
// format.
static const IVec4 COLOR_TABLE_INT[COLOR_TABLE_SIZE] = {
    IVec4(0x70707070, 0x3C3C3C3C, 0x65656565, 0x29292929),
    IVec4(0x3C3C3C3C, 0x65656565, 0x29292929, 0x70707070),
    IVec4(0x29292929, 0x70707070, 0x3C3C3C3C, 0x65656565),
    IVec4(0x65656565, 0x29292929, 0x70707070, 0x3C3C3C3C),
};

// Reference clear colors created from the color table values
static const VkClearValue REFERENCE_CLEAR_COLOR_FLOAT[COLOR_TABLE_SIZE] = {
    makeClearValueColorF32(COLOR_TABLE_FLOAT[0].x(), COLOR_TABLE_FLOAT[0].y(), COLOR_TABLE_FLOAT[0].z(),
                           COLOR_TABLE_FLOAT[0].w()),
    makeClearValueColorF32(COLOR_TABLE_FLOAT[1].x(), COLOR_TABLE_FLOAT[1].y(), COLOR_TABLE_FLOAT[1].z(),
                           COLOR_TABLE_FLOAT[1].w()),
    makeClearValueColorF32(COLOR_TABLE_FLOAT[2].x(), COLOR_TABLE_FLOAT[2].y(), COLOR_TABLE_FLOAT[2].z(),
                           COLOR_TABLE_FLOAT[2].w()),
    makeClearValueColorF32(COLOR_TABLE_FLOAT[3].x(), COLOR_TABLE_FLOAT[3].y(), COLOR_TABLE_FLOAT[3].z(),
                           COLOR_TABLE_FLOAT[3].w()),
};

static const Texture s_textures[] = {
    Texture(IMAGE_TYPE_2D, tcu::IVec3(32, 32, 1), 1),
    Texture(IMAGE_TYPE_2D_ARRAY, tcu::IVec3(32, 32, 1), 4),
};

static VkClearValue getClearValueInt(const CaseDef &caseDef, uint32_t colorTableIndex)
{
    VkClearValue clearValue;
    uint32_t channelMask = 0;

    if (caseDef.upload == UPLOAD_DRAW)
    {
        // We use this mask to get small color values in the vertex buffer and
        // avoid possible round off errors from int-to-float conversions.
        channelMask = 0xFFu;
    }
    else
    {
        VkFormat format;
        tcu::TextureFormat tcuFormat;

        // Select a mask such that no integer-based color values end up
        // reinterpreted as NaN/Inf/denorm values.
        if (caseDef.upload == UPLOAD_CLEAR || caseDef.upload == UPLOAD_COPY)
            format = caseDef.imageFormat;
        else
            format = caseDef.viewFormat;

        tcuFormat = mapVkFormat(format);

        switch (getChannelSize(tcuFormat.type))
        {
        case 1: // 8-bit
            channelMask = 0xFFu;
            break;
        case 2: // 16-bit
            channelMask = 0xFFFFu;
            break;
        case 4: // 32-bit
            channelMask = 0xFFFFFFFFu;
            break;
        default:
            DE_ASSERT(0);
        }
    }

    clearValue.color.int32[0] = COLOR_TABLE_INT[colorTableIndex].x() & channelMask;
    clearValue.color.int32[1] = COLOR_TABLE_INT[colorTableIndex].y() & channelMask;
    clearValue.color.int32[2] = COLOR_TABLE_INT[colorTableIndex].z() & channelMask;
    clearValue.color.int32[3] = COLOR_TABLE_INT[colorTableIndex].w() & channelMask;

    return clearValue;
}

VkImageType getImageType(const ImageType textureImageType)
{
    switch (textureImageType)
    {
    case IMAGE_TYPE_2D:
    case IMAGE_TYPE_2D_ARRAY:
        return VK_IMAGE_TYPE_2D;

    default:
        DE_ASSERT(0);
        return VK_IMAGE_TYPE_LAST;
    }
}

VkImageViewType getImageViewType(const ImageType textureImageType)
{
    switch (textureImageType)
    {
    case IMAGE_TYPE_2D:
        return VK_IMAGE_VIEW_TYPE_2D;
    case IMAGE_TYPE_2D_ARRAY:
        return VK_IMAGE_VIEW_TYPE_2D_ARRAY;

    default:
        DE_ASSERT(0);
        return VK_IMAGE_VIEW_TYPE_LAST;
    }
}

static const VkFormat s_formats[] = {
    VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R32G32_SFLOAT,
    VK_FORMAT_R16G16_SFLOAT,       VK_FORMAT_R32_SFLOAT,

    VK_FORMAT_R32G32B32A32_UINT,   VK_FORMAT_R16G16B16A16_UINT,   VK_FORMAT_R8G8B8A8_UINT,
    VK_FORMAT_R32G32_UINT,         VK_FORMAT_R16G16_UINT,         VK_FORMAT_R32_UINT,

    VK_FORMAT_R32G32B32A32_SINT,   VK_FORMAT_R16G16B16A16_SINT,   VK_FORMAT_R8G8B8A8_SINT,
    VK_FORMAT_R32G32_SINT,         VK_FORMAT_R16G16_SINT,         VK_FORMAT_R32_SINT,

    VK_FORMAT_R8G8B8A8_UNORM,      VK_FORMAT_R8G8B8A8_SNORM,      VK_FORMAT_R8G8B8A8_SRGB,
    VK_FORMAT_B8G8R8A8_UNORM,      VK_FORMAT_B8G8R8A8_SNORM,      VK_FORMAT_B8G8R8A8_SRGB,
};

static const VkFormat s_swapchainFormats[] = {
    VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SNORM, VK_FORMAT_R8G8B8A8_SRGB,
    VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_SNORM, VK_FORMAT_B8G8R8A8_SRGB,
};

bool isSRGBConversionRequired(const CaseDef &caseDef)
{
    bool required = false;

    if (isSRGB(mapVkFormat(caseDef.imageFormat)))
    {
        if (caseDef.upload == UPLOAD_CLEAR)
        {
            required = true;
        }
    }

    if (isSRGB(mapVkFormat(caseDef.viewFormat)))
    {
        if (caseDef.upload == UPLOAD_DRAW || caseDef.upload == UPLOAD_STORE)
        {
            required = true;
        }
    }

    return required;
}

inline bool formatsAreCompatible(const VkFormat format0, const VkFormat format1)
{
    return format0 == format1 || mapVkFormat(format0).getPixelSize() == mapVkFormat(format1).getPixelSize();
}

std::string getColorFormatStr(const int numComponents, const bool isUint, const bool isSint)
{
    std::ostringstream str;
    if (numComponents == 1)
        str << (isUint ? "uint" : isSint ? "int" : "float");
    else
        str << (isUint ? "u" : isSint ? "i" : "") << "vec" << numComponents;

    return str.str();
}

// Select the highest sample count usable by the platform
VkSampleCountFlagBits getMaxAvailableSampleCount(const Context &context, VkFormat format, VkImageType imageType,
                                                 VkImageUsageFlags usage, VkImageCreateFlags flags)
{
    const InstanceInterface &vki      = context.getInstanceInterface();
    const VkPhysicalDevice physDevice = context.getPhysicalDevice();

    VkPhysicalDeviceProperties deviceProperties;
    vki.getPhysicalDeviceProperties(physDevice, &deviceProperties);

    VkSampleCountFlags supportedSampleCount = std::min(deviceProperties.limits.framebufferColorSampleCounts,
                                                       deviceProperties.limits.framebufferDepthSampleCounts);
    std::vector<VkSampleCountFlagBits> possibleSampleCounts{VK_SAMPLE_COUNT_64_BIT, VK_SAMPLE_COUNT_32_BIT,
                                                            VK_SAMPLE_COUNT_16_BIT, VK_SAMPLE_COUNT_8_BIT,
                                                            VK_SAMPLE_COUNT_4_BIT,  VK_SAMPLE_COUNT_2_BIT};

    VkImageFormatProperties imageFormatProperties;
    vki.getPhysicalDeviceImageFormatProperties(physDevice, format, imageType, VK_IMAGE_TILING_OPTIMAL, usage, flags,
                                               &imageFormatProperties);

    for (auto &possibleSampleCount : possibleSampleCounts)
    {
        if ((supportedSampleCount & possibleSampleCount) && (imageFormatProperties.sampleCounts & possibleSampleCount))
        {
            return possibleSampleCount;
        }
    }
    return VK_SAMPLE_COUNT_1_BIT;
}

std::string getShaderSamplerType(const tcu::TextureFormat &format, VkImageViewType type)
{
    std::ostringstream samplerType;

    if (tcu::getTextureChannelClass(format.type) == tcu::TEXTURECHANNELCLASS_UNSIGNED_INTEGER)
        samplerType << "u";
    else if (tcu::getTextureChannelClass(format.type) == tcu::TEXTURECHANNELCLASS_SIGNED_INTEGER)
        samplerType << "i";

    switch (type)
    {
    case VK_IMAGE_VIEW_TYPE_2D:
        samplerType << "sampler2D";
        break;

    case VK_IMAGE_VIEW_TYPE_2D_ARRAY:
        samplerType << "sampler2DArray";
        break;

    default:
        DE_FATAL("Ivalid image view type");
        break;
    }

    return samplerType.str();
}

void initPrograms(SourceCollections &programCollection, const CaseDef caseDef)
{
    if (caseDef.upload == UPLOAD_DRAW)
    {
        {
            std::ostringstream src;
            src << glu::getGLSLVersionDeclaration(glu::GLSL_VERSION_450) << "\n"
                << "\n"
                << "layout(location = 0) in  vec4 in_position;\n"
                << "layout(location = 1) in  vec4 in_color;\n"
                << "layout(location = 0) out vec4 out_color;\n"
                << "\n"
                << "out gl_PerVertex {\n"
                << "    vec4 gl_Position;\n"
                << "};\n"
                << "\n"
                << "void main(void)\n"
                << "{\n"
                << "    gl_Position = in_position;\n"
                << "    out_color = in_color;\n"
                << "}\n";

            programCollection.glslSources.add("uploadDrawVert") << glu::VertexSource(src.str());
        }

        {
            const int numComponents       = getNumUsedChannels(mapVkFormat(caseDef.viewFormat).order);
            const bool isUint             = isUintFormat(caseDef.viewFormat);
            const bool isSint             = isIntFormat(caseDef.viewFormat);
            const std::string colorFormat = getColorFormatStr(numComponents, isUint, isSint);

            std::ostringstream src;
            src << glu::getGLSLVersionDeclaration(glu::GLSL_VERSION_450) << "\n"
                << "\n"
                << "layout(location = 0) in  vec4 in_color;\n"
                << "layout(location = 0) out " << colorFormat << " out_color;\n"
                << "\n"
                << "void main(void)\n"
                << "{\n"
                << "    out_color = " << colorFormat << "("
                << (numComponents == 1 ? "in_color.r" :
                    numComponents == 2 ? "in_color.rg" :
                    numComponents == 3 ? "in_color.rgb" :
                                         "in_color")
                << ");\n"
                << "}\n";

            programCollection.glslSources.add("uploadDrawFrag") << glu::FragmentSource(src.str());
        }
    }

    if (caseDef.upload == UPLOAD_STORE)
    {
        const TextureFormat tcuFormat    = mapVkFormat(caseDef.viewFormat);
        const std::string imageFormatStr = getShaderImageFormatQualifier(tcuFormat);
        const std::string imageTypeStr   = getShaderImageType(tcuFormat, caseDef.imageType);
        const std::string colorTypeStr   = isUintFormat(caseDef.viewFormat) ? "uvec4" :
                                           isIntFormat(caseDef.viewFormat)  ? "ivec4" :
                                                                              "vec4";
        const bool isIntegerFormat       = isUintFormat(caseDef.viewFormat) || isIntFormat(caseDef.viewFormat);

        std::ostringstream src;
        src << glu::getGLSLVersionDeclaration(glu::GLSL_VERSION_450) << "\n"
            << "\n"
            << "layout (local_size_x = 1) in;\n"
            << "\n"
            << "layout(binding=0, " << imageFormatStr << ") writeonly uniform " << imageTypeStr << " u_image;\n"
            << "\n"
            << "const " << colorTypeStr << " colorTable[] = " << colorTypeStr << "[](\n";
        for (uint32_t idx = 0; idx < COLOR_TABLE_SIZE; idx++)
        {
            if (isIntegerFormat)
            {
                const VkClearValue clearValue = getClearValueInt(caseDef, idx);

                src << "     " << colorTypeStr << "(" << clearValue.color.int32[0] << ", " << clearValue.color.int32[1]
                    << ", " << clearValue.color.int32[2] << ", " << clearValue.color.int32[3] << ")";
            }
            else
                src << "     " << colorTypeStr << "(" << COLOR_TABLE_FLOAT[idx].x() << ", "
                    << COLOR_TABLE_FLOAT[idx].y() << ", " << COLOR_TABLE_FLOAT[idx].z() << ", "
                    << COLOR_TABLE_FLOAT[idx].w() << ")";
            if (idx < COLOR_TABLE_SIZE - 1)
                src << ",";
            src << "\n";
        }
        src << ");\n"
            << "\n"
            << "void main(void)\n"
            << "{\n";
        if (caseDef.imageType == IMAGE_TYPE_2D)
        {
            src << "    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);\n";
        }
        else
        {
            DE_ASSERT(caseDef.imageType == IMAGE_TYPE_2D_ARRAY);
            src << "    ivec3 pos = ivec3(gl_GlobalInvocationID.xyz);\n";
        }
        src << "    " << colorTypeStr << " color = colorTable[gl_GlobalInvocationID.z];\n"
            << "    imageStore(u_image, pos, color);\n"
            << "}\n";

        programCollection.glslSources.add("uploadStoreComp") << glu::ComputeSource(src.str());
    }

    if (caseDef.download == DOWNLOAD_LOAD)
    {
        const TextureFormat tcuFormat    = mapVkFormat(caseDef.viewFormat);
        const std::string imageFormatStr = getShaderImageFormatQualifier(tcuFormat);
        const std::string imageTypeStr   = getShaderImageType(tcuFormat, caseDef.imageType);
        const std::string colorTypeStr   = isUintFormat(caseDef.viewFormat) ? "uvec4" :
                                           isIntFormat(caseDef.viewFormat)  ? "ivec4" :
                                                                              "vec4";

        std::ostringstream src;
        src << glu::getGLSLVersionDeclaration(glu::GLSL_VERSION_450) << "\n"
            << "\n"
            << "layout (local_size_x = 1) in;\n"
            << "\n"
            << "layout(binding=0, " << imageFormatStr << ") readonly uniform " << imageTypeStr << " in_image;\n"
            << "layout(binding=1, " << imageFormatStr << ") writeonly uniform " << imageTypeStr << " out_image;\n"
            << "\n"
            << "void main(void)\n"
            << "{\n";
        if (caseDef.imageType == IMAGE_TYPE_2D)
        {
            src << "    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);\n";
        }
        else
        {
            DE_ASSERT(caseDef.imageType == IMAGE_TYPE_2D_ARRAY);
            src << "    ivec3 pos = ivec3(gl_GlobalInvocationID.xyz);\n";
        }
        src << "    imageStore(out_image, pos, imageLoad(in_image, pos));\n"
            << "}\n";

        programCollection.glslSources.add("downloadLoadComp") << glu::ComputeSource(src.str());
    }

    if (caseDef.download == DOWNLOAD_TEXTURE)
    {
        const TextureFormat tcuFormat    = mapVkFormat(caseDef.viewFormat);
        const VkImageViewType viewType   = getImageViewType(caseDef.imageType);
        const std::string samplerTypeStr = getShaderSamplerType(tcuFormat, viewType);
        const std::string imageFormatStr = getShaderImageFormatQualifier(tcuFormat);
        const std::string imageTypeStr   = getShaderImageType(tcuFormat, caseDef.imageType);
        const std::string colorTypeStr   = isUintFormat(caseDef.viewFormat) ? "uvec4" :
                                           isIntFormat(caseDef.viewFormat)  ? "ivec4" :
                                                                              "vec4";

        std::ostringstream src;
        src << glu::getGLSLVersionDeclaration(glu::GLSL_VERSION_450) << "\n"
            << "\n"
            << "layout (local_size_x = 1) in;\n"
            << "\n"
            << "layout(binding=0) uniform " << samplerTypeStr << " u_tex;\n"
            << "layout(binding=1, " << imageFormatStr << ") writeonly uniform " << imageTypeStr << " out_image;\n"
            << "\n"
            << "void main(void)\n"
            << "{\n";
        if (caseDef.imageType == IMAGE_TYPE_2D)
        {
            src << "    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);\n";
        }
        else
        {
            DE_ASSERT(caseDef.imageType == IMAGE_TYPE_2D_ARRAY);
            src << "    ivec3 pos = ivec3(gl_GlobalInvocationID.xyz);\n";
        }
        src << "    imageStore(out_image, pos, texelFetch(u_tex, pos, 0));\n"
            << "}\n";

        programCollection.glslSources.add("downloadTextureComp") << glu::ComputeSource(src.str());
    }
}

Move<VkImage> makeImage(const DeviceInterface &vk, const VkDevice device, VkImageCreateFlags flags,
                        VkImageType imageType, const VkFormat format, const VkFormat viewFormat,
                        const bool useImageFormatList, const IVec3 &size, const uint32_t numMipLevels,
                        const uint32_t numLayers, const VkImageUsageFlags usage,
                        VkSampleCountFlagBits sampleCount = VK_SAMPLE_COUNT_1_BIT)
{
    const VkFormat formatList[2] = {format, viewFormat};

    const VkImageFormatListCreateInfo formatListInfo = {
        VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO, // VkStructureType sType;
        nullptr,                                         // const void* pNext;
        2u,                                              // uint32_t                    viewFormatCount
        formatList                                       // const VkFormat*            pViewFormats
    };

    const VkImageCreateInfo imageParams = {
        VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,            // VkStructureType sType;
        useImageFormatList ? &formatListInfo : nullptr, // const void* pNext;
        flags,                                          // VkImageCreateFlags flags;
        imageType,                                      // VkImageType imageType;
        format,                                         // VkFormat format;
        makeExtent3D(size),                             // VkExtent3D extent;
        numMipLevels,                                   // uint32_t mipLevels;
        numLayers,                                      // uint32_t arrayLayers;
        sampleCount,                                    // VkSampleCountFlagBits samples;
        VK_IMAGE_TILING_OPTIMAL,                        // VkImageTiling tiling;
        usage,                                          // VkImageUsageFlags usage;
        VK_SHARING_MODE_EXCLUSIVE,                      // VkSharingMode sharingMode;
        0u,                                             // uint32_t queueFamilyIndexCount;
        nullptr,                                        // const uint32_t* pQueueFamilyIndices;
        VK_IMAGE_LAYOUT_UNDEFINED,                      // VkImageLayout initialLayout;
    };
    return createImage(vk, device, &imageParams);
}

inline VkImageSubresourceRange makeColorSubresourceRange(const int baseArrayLayer, const int layerCount)
{
    return makeImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, static_cast<uint32_t>(baseArrayLayer),
                                     static_cast<uint32_t>(layerCount));
}

Move<VkSampler> makeSampler(const DeviceInterface &vk, const VkDevice device)
{
    const VkSamplerCreateInfo samplerParams = {
        VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,   // VkStructureType sType;
        nullptr,                                 // const void* pNext;
        (VkSamplerCreateFlags)0,                 // VkSamplerCreateFlags flags;
        VK_FILTER_NEAREST,                       // VkFilter magFilter;
        VK_FILTER_NEAREST,                       // VkFilter minFilter;
        VK_SAMPLER_MIPMAP_MODE_NEAREST,          // VkSamplerMipmapMode mipmapMode;
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,   // VkSamplerAddressMode addressModeU;
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,   // VkSamplerAddressMode addressModeV;
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,   // VkSamplerAddressMode addressModeW;
        0.0f,                                    // float mipLodBias;
        VK_FALSE,                                // VkBool32 anisotropyEnable;
        1.0f,                                    // float maxAnisotropy;
        VK_FALSE,                                // VkBool32 compareEnable;
        VK_COMPARE_OP_ALWAYS,                    // VkCompareOp compareOp;
        0.0f,                                    // float minLod;
        0.0f,                                    // float maxLod;
        VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK, // VkBorderColor borderColor;
        VK_FALSE,                                // VkBool32 unnormalizedCoordinates;
    };

    return createSampler(vk, device, &samplerParams);
}

Move<VkPipeline> makeGraphicsPipeline(
    const DeviceInterface &vk, const VkDevice device, const VkPipelineLayout pipelineLayout,
    const VkRenderPass renderPass, const VkShaderModule vertexModule, const VkShaderModule fragmentModule,
    const IVec2 &renderSize, const VkPrimitiveTopology topology, const uint32_t subpass,
    const VkSampleCountFlagBits sampleCount               = VK_SAMPLE_COUNT_1_BIT,
    const ResolveAttachmentTestType resolveAttachmentTest = ResolveAttachmentTestType::RA_TEST_NONE)
{
    const std::vector<VkViewport> viewports(1, makeViewport(renderSize));
    const std::vector<VkRect2D> scissors(1, makeRect2D(renderSize));
    const bool isResolveAttachmentTest = (resolveAttachmentTest != ResolveAttachmentTestType::RA_TEST_NONE);

    const VkVertexInputBindingDescription vertexInputBindingDescription = {
        0u,                           // uint32_t binding;
        (uint32_t)(2 * sizeof(Vec4)), // uint32_t stride;
        VK_VERTEX_INPUT_RATE_VERTEX,  // VkVertexInputRate inputRate;
    };

    const VkVertexInputAttributeDescription vertexInputAttributeDescriptions[] = {
        {
            0u,                            // uint32_t location;
            0u,                            // uint32_t binding;
            VK_FORMAT_R32G32B32A32_SFLOAT, // VkFormat format;
            0u,                            // uint32_t offset;
        },
        {
            1u,                            // uint32_t location;
            0u,                            // uint32_t binding;
            VK_FORMAT_R32G32B32A32_SFLOAT, // VkFormat format;
            (uint32_t)sizeof(Vec4),        // uint32_t offset;
        }};

    const VkPipelineVertexInputStateCreateInfo vertexInputStateCreateInfo = {
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, // VkStructureType                             sType;
        nullptr,                                                   // const void*                                 pNext;
        (VkPipelineVertexInputStateCreateFlags)0,                  // VkPipelineVertexInputStateCreateFlags       flags;
        1u,                              // uint32_t                                    vertexBindingDescriptionCount;
        &vertexInputBindingDescription,  // const VkVertexInputBindingDescription*      pVertexBindingDescriptions;
        2u,                              // uint32_t                                    vertexAttributeDescriptionCount;
        vertexInputAttributeDescriptions // const VkVertexInputAttributeDescription*    pVertexAttributeDescriptions;
    };

    const VkPipelineMultisampleStateCreateInfo multisampleStateCreateInfo = {
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, // VkStructureType                          sType
        nullptr,                                                  // const void*                              pNext
        0u,                                                       // VkPipelineMultisampleStateCreateFlags    flags
        sampleCount, // VkSampleCountFlagBits                    rasterizationSamples
        VK_FALSE,    // VkBool32                                 sampleShadingEnable
        1.0f,        // float                                    minSampleShading
        nullptr,     // const VkSampleMask*                      pSampleMask
        VK_FALSE,    // VkBool32                                 alphaToCoverageEnable
        VK_FALSE     // VkBool32                                 alphaToOneEnable
    };

    return vk::makeGraphicsPipeline(
        vk,                          // const DeviceInterface&                        vk
        device,                      // const VkDevice                                device
        pipelineLayout,              // const VkPipelineLayout                        pipelineLayout
        vertexModule,                // const VkShaderModule                          vertexShaderModule
        VK_NULL_HANDLE,              // const VkShaderModule                          tessellationControlModule
        VK_NULL_HANDLE,              // const VkShaderModule                          tessellationEvalModule
        VK_NULL_HANDLE,              // const VkShaderModule                          geometryShaderModule
        fragmentModule,              // const VkShaderModule                          fragmentShaderModule
        renderPass,                  // const VkRenderPass                            renderPass
        viewports,                   // const std::vector<VkViewport>&                viewports
        scissors,                    // const std::vector<VkRect2D>&                  scissors
        topology,                    // const VkPrimitiveTopology                     topology
        subpass,                     // const uint32_t                                subpass
        0u,                          // const uint32_t                                patchControlPoints
        &vertexInputStateCreateInfo, // const VkPipelineVertexInputStateCreateInfo*   vertexInputStateCreateInfo
        nullptr,                     // const VkPipelineRasterizationStateCreateInfo* rasterizationStateCreateInfo
        isResolveAttachmentTest ? &multisampleStateCreateInfo :
                                  nullptr // const VkPipelineMultisampleStateCreateInfo*   multisampleStateCreateInfo
    );
}

Move<VkRenderPass> makeRenderPass(
    const DeviceInterface &vk, const VkDevice device, const VkFormat viewFormat, const uint32_t numLayers,
    VkSampleCountFlagBits sampleCount                     = VK_SAMPLE_COUNT_1_BIT,
    const ResolveAttachmentTestType resolveAttachmentTest = ResolveAttachmentTestType::RA_TEST_NONE)
{
    const bool isResolveAttachmentTest = (resolveAttachmentTest != ResolveAttachmentTestType::RA_TEST_NONE);

    const VkAttachmentDescription colorAttachmentDescription = {
        (VkAttachmentDescriptionFlags)0, // VkAttachmentDescriptionFlags flags;
        viewFormat,                      // VkFormat format;
        sampleCount,                     // VkSampleCountFlagBits samples;
        VK_ATTACHMENT_LOAD_OP_CLEAR,     // VkAttachmentLoadOp loadOp;
        isResolveAttachmentTest ? VK_ATTACHMENT_STORE_OP_DONT_CARE :
                                  VK_ATTACHMENT_STORE_OP_STORE, // VkAttachmentStoreOp storeOp;
        VK_ATTACHMENT_LOAD_OP_DONT_CARE,                        // VkAttachmentLoadOp stencilLoadOp;
        VK_ATTACHMENT_STORE_OP_DONT_CARE,                       // VkAttachmentStoreOp stencilStoreOp;
        VK_IMAGE_LAYOUT_UNDEFINED,                              // VkImageLayout initialLayout;
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,               // VkImageLayout finalLayout;
    };
    vector<VkAttachmentDescription> attachmentDescriptions(numLayers, colorAttachmentDescription);

    if (isResolveAttachmentTest)
    {
        const VkAttachmentDescription resolveAttachmentDescription = {
            (VkAttachmentDescriptionFlags)0,          // VkAttachmentDescriptionFlags flags;
            viewFormat,                               // VkFormat format;
            VK_SAMPLE_COUNT_1_BIT,                    // VkSampleCountFlagBits samples;
            VK_ATTACHMENT_LOAD_OP_CLEAR,              // VkAttachmentLoadOp loadOp;
            VK_ATTACHMENT_STORE_OP_STORE,             // VkAttachmentStoreOp storeOp;
            VK_ATTACHMENT_LOAD_OP_DONT_CARE,          // VkAttachmentLoadOp stencilLoadOp;
            VK_ATTACHMENT_STORE_OP_DONT_CARE,         // VkAttachmentStoreOp stencilStoreOp;
            VK_IMAGE_LAYOUT_UNDEFINED,                // VkImageLayout initialLayout;
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, // VkImageLayout finalLayout;
        };

        // Resolve attachments are appended at the end of multisampled attachments
        attachmentDescriptions.insert(attachmentDescriptions.end(), numLayers, resolveAttachmentDescription);
    }

    // Create a subpass for each attachment (each attachement is a layer of an arrayed image).
    vector<VkAttachmentReference> colorAttachmentReferences(numLayers);
    vector<VkAttachmentReference> resolveAttachmentReferences(numLayers);
    vector<VkSubpassDescription> subpasses;

    // Ordering here must match the framebuffer attachments
    for (uint32_t i = 0; i < numLayers; ++i)
    {
        const VkAttachmentReference attachmentRef = {
            i,                                       // uint32_t attachment;
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL // VkImageLayout layout;
        };

        colorAttachmentReferences[i] = attachmentRef;

        const VkAttachmentReference resolveAttachmentRef = {
            // All resolve attachments go after all color attachments in the frame buffer
            numLayers + i,                           // uint32_t attachment;
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL // VkImageLayout layout;
        };

        resolveAttachmentReferences[i] = resolveAttachmentRef;

        const VkSubpassDescription subpassDescription = {
            (VkSubpassDescriptionFlags)0,    // VkSubpassDescriptionFlags flags;
            VK_PIPELINE_BIND_POINT_GRAPHICS, // VkPipelineBindPoint pipelineBindPoint;
            0u,                              // uint32_t inputAttachmentCount;
            nullptr,                         // const VkAttachmentReference* pInputAttachments;
            1u,                              // uint32_t colorAttachmentCount;
            &colorAttachmentReferences[i],   // const VkAttachmentReference* pColorAttachments;
            isResolveAttachmentTest ? &resolveAttachmentReferences[i] :
                                      nullptr, // const VkAttachmentReference* pResolveAttachments;
            nullptr,                           // const VkAttachmentReference* pDepthStencilAttachment;
            0u,                                // uint32_t preserveAttachmentCount;
            nullptr                            // const uint32_t* pPreserveAttachments;
        };
        subpasses.push_back(subpassDescription);
    }

    const VkRenderPassCreateInfo renderPassInfo = {
        VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, // VkStructureType sType;
        nullptr,                                   // const void* pNext;
        (VkRenderPassCreateFlags)0,                // VkRenderPassCreateFlags flags;
        de::sizeU32(attachmentDescriptions),       // uint32_t attachmentCount;
        de::dataOrNull(attachmentDescriptions),    // const VkAttachmentDescription* pAttachments;
        de::sizeU32(subpasses),                    // uint32_t subpassCount;
        &subpasses[0],                             // const VkSubpassDescription* pSubpasses;
        0u,                                        // uint32_t dependencyCount;
        nullptr                                    // const VkSubpassDependency* pDependencies;
    };

    return createRenderPass(vk, device, &renderPassInfo);
}

Move<VkCommandBuffer> makeCommandBuffer(const DeviceInterface &vk, const VkDevice device,
                                        const VkCommandPool commandPool)
{
    return allocateCommandBuffer(vk, device, commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
}

vector<Vec4> genVertexData(const CaseDef &caseDef)
{
    vector<Vec4> vectorData;
    const bool isIntegerFormat = isUintFormat(caseDef.viewFormat) || isIntFormat(caseDef.viewFormat);

    const float position = caseDef.isLoadOpClearTest ? 0.5f : 1.0f;

    for (uint32_t z = 0; z < caseDef.numLayers; z++)
    {
        const uint32_t colorIdx = z % COLOR_TABLE_SIZE;
        Vec4 color;

        if (isIntegerFormat)
        {
            const VkClearValue clearValue = getClearValueInt(caseDef, colorIdx);
            const IVec4 colorInt(clearValue.color.int32[0], clearValue.color.int32[1], clearValue.color.int32[2],
                                 clearValue.color.int32[3]);

            color = colorInt.cast<float>();
        }
        else
        {
            color = COLOR_TABLE_FLOAT[colorIdx];
        }

        vectorData.push_back(Vec4(-position, -position, 0.0f, 1.0f));
        vectorData.push_back(color);
        vectorData.push_back(Vec4(-position, position, 0.0f, 1.0f));
        vectorData.push_back(color);
        vectorData.push_back(Vec4(position, -position, 0.0f, 1.0f));
        vectorData.push_back(color);
        vectorData.push_back(Vec4(position, position, 0.0f, 1.0f));
        vectorData.push_back(color);
    }

    return vectorData;
}

void generateExpectedImage(const tcu::PixelBufferAccess &image, const CaseDef &caseDef)
{
    const tcu::TextureChannelClass channelClass = tcu::getTextureChannelClass(image.getFormat().type);
    const bool isIntegerFormat                  = channelClass == tcu::TEXTURECHANNELCLASS_SIGNED_INTEGER ||
                                 channelClass == tcu::TEXTURECHANNELCLASS_UNSIGNED_INTEGER;
    const IVec2 size = caseDef.size.swizzle(0, 1);

    for (int z = 0; z < static_cast<int>(caseDef.numLayers); z++)
    {
        const uint32_t colorIdx = z % COLOR_TABLE_SIZE;
        for (int y = 0; y < size.y(); y++)
            for (int x = 0; x < size.x(); x++)
            {
                if (isIntegerFormat)
                {
                    const VkClearValue clearValue = getClearValueInt(caseDef, colorIdx);
                    const IVec4 colorInt(clearValue.color.int32[0], clearValue.color.int32[1],
                                         clearValue.color.int32[2], clearValue.color.int32[3]);

                    image.setPixel(colorInt, x, y, z);
                }
                else if (isSRGBConversionRequired(caseDef))
                    image.setPixel(tcu::linearToSRGB(COLOR_TABLE_FLOAT[colorIdx]), x, y, z);
                else
                    image.setPixel(COLOR_TABLE_FLOAT[colorIdx], x, y, z);
            }
    }
}

VkImageUsageFlags getImageUsageForTestCase(const CaseDef &caseDef)
{
    VkImageUsageFlags flags = 0u;

    switch (caseDef.upload)
    {
    case UPLOAD_CLEAR:
        flags |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        break;
    case UPLOAD_DRAW:
        flags |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        break;
    case UPLOAD_STORE:
        flags |= VK_IMAGE_USAGE_STORAGE_BIT;
        break;
    case UPLOAD_COPY:
        flags |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        break;
    default:
        DE_FATAL("Invalid upload method");
        break;
    }

    switch (caseDef.download)
    {
    case DOWNLOAD_TEXTURE:
        flags |= VK_IMAGE_USAGE_SAMPLED_BIT;
        break;
    case DOWNLOAD_LOAD:
        flags |= VK_IMAGE_USAGE_STORAGE_BIT;
        break;
    case DOWNLOAD_COPY:
        flags |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        break;
    default:
        DE_FATAL("Invalid download method");
        break;
    }

    // We can only create a view for the image if it is going to be used for any of these usages,
    // so let's make sure that we have at least one of them.
    VkImageUsageFlags viewRequiredFlags =
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (!(flags & viewRequiredFlags))
        flags |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    return flags;
}

// Executes a combination of upload/download methods
class UploadDownloadExecutor
{
public:
    UploadDownloadExecutor(Context &context, bool haveMaintenance2, const DeviceInterface &deviceInterface,
                           VkDevice device, VkQueue queue, uint32_t queueFamilyIndex, const CaseDef &caseSpec)
        : m_caseDef(caseSpec)
        , m_haveMaintenance2(haveMaintenance2)
        , m_vk(deviceInterface)
        , m_device(device)
        , m_queue(queue)
        , m_queueFamilyIndex(queueFamilyIndex)
        , m_allocator(m_vk, device,
                      getPhysicalDeviceMemoryProperties(context.getInstanceInterface(), context.getPhysicalDevice()))
    {
    }

    void runSwapchain(Context &context, VkBuffer buffer, VkImage image);

    void run(Context &context, VkBuffer buffer);

private:
    void uploadClear(Context &context);
    void uploadStore(Context &context);
    void uploadCopy(Context &context);
    void uploadDraw(Context &context);
    void downloadCopy(Context &context, VkBuffer buffer);
    void downloadTexture(Context &context, VkBuffer buffer);
    void downloadLoad(Context &context, VkBuffer buffer);

    void copyImageToBuffer(VkImage image, VkBuffer buffer, const IVec3 size, const VkAccessFlags srcAccessMask,
                           const VkImageLayout oldLayout, const uint32_t numLayers);

    const CaseDef &m_caseDef;

    bool m_haveMaintenance2;

    const DeviceInterface &m_vk;
    const VkDevice m_device;
    const VkQueue m_queue;
    const uint32_t m_queueFamilyIndex;
    SimpleAllocator m_allocator;

    Move<VkCommandPool> m_cmdPool;
    Move<VkCommandBuffer> m_cmdBuffer;

    bool m_imageIsIntegerFormat;
    bool m_viewIsIntegerFormat;

    // Target image for upload paths
    VkImage m_image;
    Move<VkImage> m_imageHolder;
    MovePtr<Allocation> m_imageAlloc;

    // Multisampled image
    VkImage m_multisampledImage;
    Move<VkImage> m_multisampledImageHolder;
    MovePtr<Allocation> m_multisampledImageAlloc;

    // Upload copy
    struct
    {
        Move<VkBuffer> colorBuffer;
        VkDeviceSize colorBufferSize;
        MovePtr<Allocation> colorBufferAlloc;
    } m_uCopy;

    // Upload draw
    struct
    {
        Move<VkBuffer> vertexBuffer;
        MovePtr<Allocation> vertexBufferAlloc;
        Move<VkPipelineLayout> pipelineLayout;
        Move<VkRenderPass> renderPass;
        Move<VkShaderModule> vertexModule;
        Move<VkShaderModule> fragmentModule;
        vector<SharedPtrVkImageView> attachments;
        vector<VkImageView> attachmentHandles;
        vector<SharedPtrVkPipeline> pipelines;
        Move<VkFramebuffer> framebuffer;
    } m_uDraw;

    // Upload store
    struct
    {
        Move<VkDescriptorPool> descriptorPool;
        Move<VkPipelineLayout> pipelineLayout;
        Move<VkDescriptorSetLayout> descriptorSetLayout;
        Move<VkDescriptorSet> descriptorSet;
        VkDescriptorImageInfo imageDescriptorInfo;
        Move<VkShaderModule> computeModule;
        Move<VkPipeline> computePipeline;
        Move<VkImageView> imageView;
    } m_uStore;

    // Download load
    struct
    {
        Move<VkDescriptorPool> descriptorPool;
        Move<VkPipelineLayout> pipelineLayout;
        Move<VkDescriptorSetLayout> descriptorSetLayout;
        Move<VkDescriptorSet> descriptorSet;
        Move<VkShaderModule> computeModule;
        Move<VkPipeline> computePipeline;
        Move<VkImageView> inImageView;
        VkDescriptorImageInfo inImageDescriptorInfo;
        Move<VkImage> outImage;
        Move<VkImageView> outImageView;
        MovePtr<Allocation> outImageAlloc;
        VkDescriptorImageInfo outImageDescriptorInfo;
    } m_dLoad;

    // Download texture
    struct
    {
        Move<VkDescriptorPool> descriptorPool;
        Move<VkPipelineLayout> pipelineLayout;
        Move<VkDescriptorSetLayout> descriptorSetLayout;
        Move<VkDescriptorSet> descriptorSet;
        Move<VkShaderModule> computeModule;
        Move<VkPipeline> computePipeline;
        Move<VkImageView> inImageView;
        VkDescriptorImageInfo inImageDescriptorInfo;
        Move<VkSampler> sampler;
        Move<VkImage> outImage;
        Move<VkImageView> outImageView;
        MovePtr<Allocation> outImageAlloc;
        VkDescriptorImageInfo outImageDescriptorInfo;
    } m_dTex;

    VkImageLayout m_imageLayoutAfterUpload;
    VkAccessFlagBits m_imageUploadAccessMask;
};

void UploadDownloadExecutor::runSwapchain(Context &context, VkBuffer buffer, VkImage image)
{
    m_imageIsIntegerFormat = isUintFormat(m_caseDef.imageFormat) || isIntFormat(m_caseDef.imageFormat);
    m_viewIsIntegerFormat  = isUintFormat(m_caseDef.viewFormat) || isIntFormat(m_caseDef.viewFormat);

    m_cmdPool = createCommandPool(m_vk, m_device, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, m_queueFamilyIndex);
    m_cmdBuffer = makeCommandBuffer(m_vk, m_device, *m_cmdPool);
    beginCommandBuffer(m_vk, *m_cmdBuffer);

    m_image = image;

    switch (m_caseDef.upload)
    {
    case UPLOAD_DRAW:
        uploadDraw(context);
        break;
    case UPLOAD_STORE:
        uploadStore(context);
        break;
    case UPLOAD_CLEAR:
        uploadClear(context);
        break;
    case UPLOAD_COPY:
        uploadCopy(context);
        break;
    default:
        DE_FATAL("Unsupported upload method");
    }

    switch (m_caseDef.download)
    {
    case DOWNLOAD_COPY:
        downloadCopy(context, buffer);
        break;
    case DOWNLOAD_LOAD:
        downloadLoad(context, buffer);
        break;
    case DOWNLOAD_TEXTURE:
        downloadTexture(context, buffer);
        break;
    default:
        DE_FATAL("Unsupported download method");
    }

    endCommandBuffer(m_vk, *m_cmdBuffer);
    submitCommandsAndWait(m_vk, m_device, m_queue, *m_cmdBuffer);
}

void UploadDownloadExecutor::run(Context &context, VkBuffer buffer)
{
    m_imageIsIntegerFormat = isUintFormat(m_caseDef.imageFormat) || isIntFormat(m_caseDef.imageFormat);
    m_viewIsIntegerFormat  = isUintFormat(m_caseDef.viewFormat) || isIntFormat(m_caseDef.viewFormat);

    m_cmdPool = createCommandPool(m_vk, m_device, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, m_queueFamilyIndex);
    m_cmdBuffer = makeCommandBuffer(m_vk, m_device, *m_cmdPool);
    beginCommandBuffer(m_vk, *m_cmdBuffer);

    const VkImageUsageFlags imageUsage = getImageUsageForTestCase(m_caseDef);
    const VkImageCreateFlags imageFlags =
        VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | (m_haveMaintenance2 ? VK_IMAGE_CREATE_EXTENDED_USAGE_BIT : 0);
    const VkImageCreateFlags imageFlagsNonMutable = (m_haveMaintenance2 ? VK_IMAGE_CREATE_EXTENDED_USAGE_BIT : 0);
    VkImageFormatProperties properties;

    if (m_caseDef.resolveAttachmentTestType)
    {
        const vk::VkImageType imageType = getImageType(m_caseDef.imageType);
        const VkImageCreateFlags msImgflags =
            (m_caseDef.resolveAttachmentTestType == ResolveAttachmentTestType::RA_TEST_RA_MUTABLE) ?
                imageFlagsNonMutable :
                imageFlags;
        const VkFormat format = (msImgflags == imageFlagsNonMutable) ? m_caseDef.viewFormat : m_caseDef.imageFormat;
        const vk::VkSampleCountFlagBits samples =
            getMaxAvailableSampleCount(context, format, imageType, imageUsage, msImgflags);

        if ((context.getInstanceInterface().getPhysicalDeviceImageFormatProperties(
                 context.getPhysicalDevice(), format, getImageType(m_caseDef.imageType), VK_IMAGE_TILING_OPTIMAL,
                 imageUsage, msImgflags, &properties) == VK_ERROR_FORMAT_NOT_SUPPORTED))
        {
            TCU_THROW(NotSupportedError, "Format not supported for multisampled image");
        }

        m_multisampledImageHolder =
            makeImage(m_vk, m_device, msImgflags, imageType, format, m_caseDef.viewFormat, m_caseDef.isFormatListTest,
                      m_caseDef.size, 1u, m_caseDef.numLayers, imageUsage, samples);
        m_multisampledImage      = *m_multisampledImageHolder;
        m_multisampledImageAlloc = bindImage(m_vk, m_device, m_allocator, m_multisampledImage, MemoryRequirement::Any);
    }

    const VkImageCreateFlags imgFlags =
        (m_caseDef.resolveAttachmentTestType == ResolveAttachmentTestType::RA_TEST_CA_MUTABLE) ? imageFlagsNonMutable :
                                                                                                 imageFlags;
    const VkFormat imgFormat = (imgFlags == imageFlagsNonMutable) ? m_caseDef.viewFormat : m_caseDef.imageFormat;

    if ((context.getInstanceInterface().getPhysicalDeviceImageFormatProperties(
             context.getPhysicalDevice(), imgFormat, getImageType(m_caseDef.imageType), VK_IMAGE_TILING_OPTIMAL,
             imageUsage, imgFlags, &properties) == VK_ERROR_FORMAT_NOT_SUPPORTED))
    {
        TCU_THROW(NotSupportedError, "Format not supported");
    }

    m_imageHolder =
        makeImage(m_vk, m_device, imgFlags, getImageType(m_caseDef.imageType), imgFormat, m_caseDef.viewFormat,
                  m_caseDef.isFormatListTest, m_caseDef.size, 1u, m_caseDef.numLayers, imageUsage);
    m_image      = *m_imageHolder;
    m_imageAlloc = bindImage(m_vk, m_device, m_allocator, m_image, MemoryRequirement::Any);

    switch (m_caseDef.upload)
    {
    case UPLOAD_DRAW:
        uploadDraw(context);
        break;
    case UPLOAD_STORE:
        uploadStore(context);
        break;
    case UPLOAD_CLEAR:
        uploadClear(context);
        break;
    case UPLOAD_COPY:
        uploadCopy(context);
        break;
    default:
        DE_FATAL("Unsupported upload method");
    }

    switch (m_caseDef.download)
    {
    case DOWNLOAD_COPY:
        downloadCopy(context, buffer);
        break;
    case DOWNLOAD_LOAD:
        downloadLoad(context, buffer);
        break;
    case DOWNLOAD_TEXTURE:
        downloadTexture(context, buffer);
        break;
    default:
        DE_FATAL("Unsupported download method");
    }

    endCommandBuffer(m_vk, *m_cmdBuffer);
    submitCommandsAndWait(m_vk, m_device, m_queue, *m_cmdBuffer);
}

void UploadDownloadExecutor::uploadClear(Context &context)
{
    (void)context;

    VkImageLayout requiredImageLayout = VK_IMAGE_LAYOUT_GENERAL;

    const VkImageSubresourceRange subresourceRange = makeColorSubresourceRange(0, m_caseDef.numLayers);
    const VkImageMemoryBarrier imageInitBarrier    = {
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, // VkStructureType sType;
        nullptr,                                // const void* pNext;
        0u,                                     // VkAccessFlags srcAccessMask;
        VK_ACCESS_TRANSFER_WRITE_BIT,           // VkAccessFlags dstAcessMask;
        VK_IMAGE_LAYOUT_UNDEFINED,              // VkImageLayout oldLayout;
        requiredImageLayout,                    // VkImageLayout newLayout;
        VK_QUEUE_FAMILY_IGNORED,                // uint32_t srcQueueFamilyIndex;
        VK_QUEUE_FAMILY_IGNORED,                // uint32_t destQueueFamilyIndex;
        m_image,                                // VkImage image;
        subresourceRange                        // VkImageSubresourceRange subresourceRange;
    };

    m_vk.cmdPipelineBarrier(*m_cmdBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0u, 0u,
                            nullptr, 0u, nullptr, 1u, &imageInitBarrier);

    for (uint32_t layer = 0; layer < m_caseDef.numLayers; layer++)
    {
        const VkImageSubresourceRange layerSubresourceRange = makeColorSubresourceRange(layer, 1u);
        const uint32_t colorIdx                             = layer % COLOR_TABLE_SIZE;
        const VkClearColorValue clearColor = m_imageIsIntegerFormat ? getClearValueInt(m_caseDef, colorIdx).color :
                                                                      REFERENCE_CLEAR_COLOR_FLOAT[colorIdx].color;
        m_vk.cmdClearColorImage(*m_cmdBuffer, m_image, requiredImageLayout, &clearColor, 1u, &layerSubresourceRange);
    }

    m_imageLayoutAfterUpload = requiredImageLayout;
    m_imageUploadAccessMask  = VK_ACCESS_TRANSFER_WRITE_BIT;
}

void UploadDownloadExecutor::uploadStore(Context &context)
{
    const vk::VkImageViewUsageCreateInfo viewUsageCreateInfo = {
        VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO, // VkStructureType        sType
        nullptr,                                        // const void*            pNext
        VK_IMAGE_USAGE_STORAGE_BIT,                     // VkImageUsageFlags usage;
    };
    m_uStore.imageView = makeImageView(m_vk, m_device, m_image, getImageViewType(m_caseDef.imageType),
                                       m_caseDef.viewFormat, makeColorSubresourceRange(0, m_caseDef.numLayers),
                                       m_haveMaintenance2 ? &viewUsageCreateInfo : nullptr);

    // Setup compute pipeline
    m_uStore.descriptorPool = DescriptorPoolBuilder()
                                  .addType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                                  .build(m_vk, m_device, VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT, 1u);

    m_uStore.descriptorSetLayout = DescriptorSetLayoutBuilder()
                                       .addSingleBinding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT)
                                       .build(m_vk, m_device);

    m_uStore.pipelineLayout = makePipelineLayout(m_vk, m_device, *m_uStore.descriptorSetLayout);
    m_uStore.descriptorSet = makeDescriptorSet(m_vk, m_device, *m_uStore.descriptorPool, *m_uStore.descriptorSetLayout);
    m_uStore.imageDescriptorInfo =
        makeDescriptorImageInfo(VK_NULL_HANDLE, *m_uStore.imageView, VK_IMAGE_LAYOUT_GENERAL);
    m_uStore.computeModule =
        createShaderModule(m_vk, m_device, context.getBinaryCollection().get("uploadStoreComp"), 0);
    m_uStore.computePipeline = makeComputePipeline(m_vk, m_device, *m_uStore.pipelineLayout, *m_uStore.computeModule);

    DescriptorSetUpdateBuilder()
        .writeSingle(*m_uStore.descriptorSet, DescriptorSetUpdateBuilder::Location::binding(0u),
                     VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &m_uStore.imageDescriptorInfo)
        .update(m_vk, m_device);

    // Transition storage image for shader access (imageStore)
    VkImageLayout requiredImageLayout       = VK_IMAGE_LAYOUT_GENERAL;
    const VkImageMemoryBarrier imageBarrier = {
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,            // VkStructureType sType;
        nullptr,                                           // const void* pNext;
        (VkAccessFlags)0,                                  // VkAccessFlags srcAccessMask;
        (VkAccessFlags)VK_ACCESS_SHADER_WRITE_BIT,         // VkAccessFlags dstAccessMask;
        VK_IMAGE_LAYOUT_UNDEFINED,                         // VkImageLayout oldLayout;
        requiredImageLayout,                               // VkImageLayout newLayout;
        VK_QUEUE_FAMILY_IGNORED,                           // uint32_t srcQueueFamilyIndex;
        VK_QUEUE_FAMILY_IGNORED,                           // uint32_t destQueueFamilyIndex;
        m_image,                                           // VkImage image;
        makeColorSubresourceRange(0, m_caseDef.numLayers), // VkImageSubresourceRange subresourceRange;
    };

    m_vk.cmdPipelineBarrier(*m_cmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u,
                            nullptr, 0u, nullptr, 1u, &imageBarrier);

    // Dispatch
    m_vk.cmdBindPipeline(*m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, *m_uStore.computePipeline);
    m_vk.cmdBindDescriptorSets(*m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, *m_uStore.pipelineLayout, 0u, 1u,
                               &m_uStore.descriptorSet.get(), 0u, nullptr);
    m_vk.cmdDispatch(*m_cmdBuffer, m_caseDef.size.x(), m_caseDef.size.y(), m_caseDef.numLayers);

    m_imageLayoutAfterUpload = requiredImageLayout;
    m_imageUploadAccessMask  = VK_ACCESS_SHADER_WRITE_BIT;
}

void UploadDownloadExecutor::uploadCopy(Context &context)
{
    (void)context;

    // Create a host-mappable buffer with the color data to upload
    const VkDeviceSize pixelSize = tcu::getPixelSize(mapVkFormat(m_caseDef.imageFormat));
    const VkDeviceSize layerSize = m_caseDef.size.x() * m_caseDef.size.y() * m_caseDef.size.z() * pixelSize;

    m_uCopy.colorBufferSize = layerSize * m_caseDef.numLayers;
    m_uCopy.colorBuffer     = makeBuffer(m_vk, m_device, m_uCopy.colorBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    m_uCopy.colorBufferAlloc =
        bindBuffer(m_vk, m_device, m_allocator, *m_uCopy.colorBuffer, MemoryRequirement::HostVisible);

    // Fill color buffer
    const tcu::TextureFormat tcuFormat = mapVkFormat(m_caseDef.imageFormat);
    VkDeviceSize layerOffset           = 0ull;
    for (uint32_t layer = 0; layer < m_caseDef.numLayers; layer++)
    {
        tcu::PixelBufferAccess imageAccess =
            tcu::PixelBufferAccess(tcuFormat, m_caseDef.size.x(), m_caseDef.size.y(), 1u,
                                   (uint8_t *)m_uCopy.colorBufferAlloc->getHostPtr() + layerOffset);
        const uint32_t colorIdx = layer % COLOR_TABLE_SIZE;
        if (m_imageIsIntegerFormat)
        {
            const VkClearValue clearValue = getClearValueInt(m_caseDef, colorIdx);
            const IVec4 colorInt(clearValue.color.int32[0], clearValue.color.int32[1], clearValue.color.int32[2],
                                 clearValue.color.int32[3]);

            tcu::clear(imageAccess, colorInt);
        }
        else
            tcu::clear(imageAccess, COLOR_TABLE_FLOAT[colorIdx]);
        layerOffset += layerSize;
    }

    flushAlloc(m_vk, m_device, *(m_uCopy.colorBufferAlloc));

    // Prepare buffer and image for copy
    const VkBufferMemoryBarrier bufferInitBarrier = {
        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, // VkStructureType    sType;
        nullptr,                                 // const void*        pNext;
        VK_ACCESS_HOST_WRITE_BIT,                // VkAccessFlags      srcAccessMask;
        VK_ACCESS_TRANSFER_READ_BIT,             // VkAccessFlags      dstAccessMask;
        VK_QUEUE_FAMILY_IGNORED,                 // uint32_t           srcQueueFamilyIndex;
        VK_QUEUE_FAMILY_IGNORED,                 // uint32_t           dstQueueFamilyIndex;
        *m_uCopy.colorBuffer,                    // VkBuffer           buffer;
        0ull,                                    // VkDeviceSize       offset;
        VK_WHOLE_SIZE,                           // VkDeviceSize       size;
    };

    const VkImageMemoryBarrier imageInitBarrier = {
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,           // VkStructureType sType;
        nullptr,                                          // const void* pNext;
        0u,                                               // VkAccessFlags srcAccessMask;
        VK_ACCESS_TRANSFER_WRITE_BIT,                     // VkAccessFlags dstAccessMask;
        VK_IMAGE_LAYOUT_UNDEFINED,                        // VkImageLayout oldLayout;
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,             // VkImageLayout newLayout;
        VK_QUEUE_FAMILY_IGNORED,                          // uint32_t srcQueueFamilyIndex;
        VK_QUEUE_FAMILY_IGNORED,                          // uint32_t destQueueFamilyIndex;
        m_image,                                          // VkImage image;
        makeColorSubresourceRange(0, m_caseDef.numLayers) // VkImageSubresourceRange subresourceRange;
    };

    m_vk.cmdPipelineBarrier(*m_cmdBuffer, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0u, 0u, nullptr,
                            1u, &bufferInitBarrier, 1u, &imageInitBarrier);

    // Copy buffer to image
    const VkImageSubresourceLayers subresource = {
        VK_IMAGE_ASPECT_COLOR_BIT, // VkImageAspectFlags    aspectMask;
        0u,                        // uint32_t              mipLevel;
        0u,                        // uint32_t              baseArrayLayer;
        m_caseDef.numLayers,       // uint32_t              layerCount;
    };

    const VkBufferImageCopy region = {
        0ull,                         // VkDeviceSize                bufferOffset;
        0u,                           // uint32_t                    bufferRowLength;
        0u,                           // uint32_t                    bufferImageHeight;
        subresource,                  // VkImageSubresourceLayers    imageSubresource;
        makeOffset3D(0, 0, 0),        // VkOffset3D                  imageOffset;
        makeExtent3D(m_caseDef.size), // VkExtent3D                  imageExtent;
    };

    m_vk.cmdCopyBufferToImage(*m_cmdBuffer, *m_uCopy.colorBuffer, m_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1u,
                              &region);

    const VkImageMemoryBarrier imagePostInitBarrier = {
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,           // VkStructureType sType;
        nullptr,                                          // const void* pNext;
        VK_ACCESS_TRANSFER_WRITE_BIT,                     // VkAccessFlags srcAccessMask;
        VK_ACCESS_TRANSFER_READ_BIT,                      // VkAccessFlags dstAccessMask;
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,             // VkImageLayout oldLayout;
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,             // VkImageLayout newLayout;
        VK_QUEUE_FAMILY_IGNORED,                          // uint32_t srcQueueFamilyIndex;
        VK_QUEUE_FAMILY_IGNORED,                          // uint32_t destQueueFamilyIndex;
        m_image,                                          // VkImage image;
        makeColorSubresourceRange(0, m_caseDef.numLayers) // VkImageSubresourceRange subresourceRange;
    };

    m_vk.cmdPipelineBarrier(*m_cmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0u, 0u,
                            nullptr, 0u, nullptr, 1u, &imagePostInitBarrier);

    m_imageLayoutAfterUpload = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    m_imageUploadAccessMask  = VK_ACCESS_TRANSFER_WRITE_BIT;
}

void UploadDownloadExecutor::uploadDraw(Context &context)
{
    VkSampleCountFlagBits maxSampleCount = getMaxAvailableSampleCount(
        context, m_caseDef.imageFormat, getImageType(m_caseDef.imageType), getImageUsageForTestCase(m_caseDef), 0u);
    VkSampleCountFlagBits sampleCount = m_caseDef.resolveAttachmentTestType ? maxSampleCount : VK_SAMPLE_COUNT_1_BIT;
    // Create vertex buffer
    {
        const vector<Vec4> vertices         = genVertexData(m_caseDef);
        const VkDeviceSize vertexBufferSize = vertices.size() * sizeof(Vec4);

        m_uDraw.vertexBuffer = makeBuffer(m_vk, m_device, vertexBufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        m_uDraw.vertexBufferAlloc =
            bindBuffer(m_vk, m_device, m_allocator, *m_uDraw.vertexBuffer, MemoryRequirement::HostVisible);
        deMemcpy(m_uDraw.vertexBufferAlloc->getHostPtr(), &vertices[0], static_cast<std::size_t>(vertexBufferSize));
        flushAlloc(m_vk, m_device, *(m_uDraw.vertexBufferAlloc));
    }

    // Create attachments and pipelines for each image layer
    m_uDraw.pipelineLayout = makePipelineLayout(m_vk, m_device);
    m_uDraw.renderPass     = makeRenderPass(m_vk, m_device, m_caseDef.viewFormat, m_caseDef.numLayers, sampleCount,
                                            m_caseDef.resolveAttachmentTestType);
    m_uDraw.vertexModule = createShaderModule(m_vk, m_device, context.getBinaryCollection().get("uploadDrawVert"), 0u);
    m_uDraw.fragmentModule =
        createShaderModule(m_vk, m_device, context.getBinaryCollection().get("uploadDrawFrag"), 0u);

    const vk::VkImageViewUsageCreateInfo viewUsageCreateInfo = {
        VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO, // VkStructureType        sType
        nullptr,                                        // const void*            pNext
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,            // VkImageUsageFlags usage;
    };

    // Create multisampled attachment view
    if (m_caseDef.resolveAttachmentTestType)
    {
        for (uint32_t subpassNdx = 0; subpassNdx < m_caseDef.numLayers; ++subpassNdx)
        {
            Move<VkImageView> multiSampledImageView = makeImageView(
                m_vk, m_device, m_multisampledImage, getImageViewType(m_caseDef.imageType), m_caseDef.viewFormat,
                makeColorSubresourceRange(subpassNdx, 1), m_haveMaintenance2 ? &viewUsageCreateInfo : nullptr);

            // Add multisampled image first in attachments as we did in the renderpass
            m_uDraw.attachmentHandles.push_back(*multiSampledImageView);
            m_uDraw.attachments.push_back(makeSharedPtr(multiSampledImageView));
        }
    }

    for (uint32_t subpassNdx = 0; subpassNdx < m_caseDef.numLayers; ++subpassNdx)
    {
        Move<VkImageView> imageView = makeImageView(m_vk, m_device, m_image, getImageViewType(m_caseDef.imageType),
                                                    m_caseDef.viewFormat, makeColorSubresourceRange(subpassNdx, 1),
                                                    m_haveMaintenance2 ? &viewUsageCreateInfo : nullptr);
        m_uDraw.attachmentHandles.push_back(*imageView);
        m_uDraw.attachments.push_back(makeSharedPtr(imageView));

        m_uDraw.pipelines.push_back(makeSharedPtr(makeGraphicsPipeline(
            m_vk, m_device, *m_uDraw.pipelineLayout, *m_uDraw.renderPass, *m_uDraw.vertexModule,
            *m_uDraw.fragmentModule, m_caseDef.size.swizzle(0, 1), VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, subpassNdx,
            sampleCount, m_caseDef.resolveAttachmentTestType)));
    }

    // Create framebuffer
    {
        const IVec2 size = m_caseDef.size.swizzle(0, 1);

        m_uDraw.framebuffer = makeFramebuffer(
            m_vk, m_device, *m_uDraw.renderPass, static_cast<uint32_t>(m_uDraw.attachmentHandles.size()),
            &m_uDraw.attachmentHandles[0], static_cast<uint32_t>(size.x()), static_cast<uint32_t>(size.y()));
    }

    // Create command buffer
    {
        {
            vector<VkClearValue> clearValues(m_caseDef.numLayers * (m_caseDef.resolveAttachmentTestType ? 2 : 1),
                                             m_viewIsIntegerFormat ? getClearValueInt(m_caseDef, 0) :
                                                                     REFERENCE_CLEAR_COLOR_FLOAT[0]);

            beginRenderPass(m_vk, *m_cmdBuffer, *m_uDraw.renderPass, *m_uDraw.framebuffer,
                            makeRect2D(0, 0, m_caseDef.size.x(), m_caseDef.size.y()), (uint32_t)clearValues.size(),
                            &clearValues[0]);
        }

        // Render
        const VkDeviceSize vertexDataPerDraw = 4 * 2 * sizeof(Vec4);
        VkDeviceSize vertexBufferOffset      = 0ull;
        for (uint32_t subpassNdx = 0; subpassNdx < m_caseDef.numLayers; ++subpassNdx)
        {
            if (subpassNdx != 0)
                m_vk.cmdNextSubpass(*m_cmdBuffer, VK_SUBPASS_CONTENTS_INLINE);

            m_vk.cmdBindPipeline(*m_cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, **m_uDraw.pipelines[subpassNdx]);

            m_vk.cmdBindVertexBuffers(*m_cmdBuffer, 0u, 1u, &m_uDraw.vertexBuffer.get(), &vertexBufferOffset);
            m_vk.cmdDraw(*m_cmdBuffer, 4u, 1u, 0u, 0u);
            vertexBufferOffset += vertexDataPerDraw;
        }

        endRenderPass(m_vk, *m_cmdBuffer);
    }

    m_imageLayoutAfterUpload = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    m_imageUploadAccessMask  = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
}

void UploadDownloadExecutor::downloadCopy(Context &context, VkBuffer buffer)
{
    (void)context;

    copyImageToBuffer(m_image, buffer, m_caseDef.size, m_imageUploadAccessMask, m_imageLayoutAfterUpload,
                      m_caseDef.numLayers);
}

void UploadDownloadExecutor::downloadTexture(Context &context, VkBuffer buffer)
{
    // Create output image with download result
    const VkImageUsageFlags usageFlags = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    m_dTex.outImage = makeImage(m_vk, m_device, 0u, VK_IMAGE_TYPE_2D, m_caseDef.viewFormat, m_caseDef.viewFormat, false,
                                m_caseDef.size, 1u, m_caseDef.numLayers, usageFlags);
    m_dTex.outImageAlloc = bindImage(m_vk, m_device, m_allocator, *m_dTex.outImage, MemoryRequirement::Any);
    m_dTex.outImageView  = makeImageView(m_vk, m_device, *m_dTex.outImage, getImageViewType(m_caseDef.imageType),
                                         m_caseDef.viewFormat, makeColorSubresourceRange(0, m_caseDef.numLayers));

    const vk::VkImageViewUsageCreateInfo viewUsageCreateInfo = {
        VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO, // VkStructureType        sType
        nullptr,                                        // const void*            pNext
        VK_IMAGE_USAGE_SAMPLED_BIT,                     // VkImageUsageFlags usage;
    };
    m_dTex.inImageView = makeImageView(m_vk, m_device, m_image, getImageViewType(m_caseDef.imageType),
                                       m_caseDef.viewFormat, makeColorSubresourceRange(0, m_caseDef.numLayers),
                                       m_haveMaintenance2 ? &viewUsageCreateInfo : nullptr);
    m_dTex.sampler     = makeSampler(m_vk, m_device);

    // Setup compute pipeline
    m_dTex.descriptorPool = DescriptorPoolBuilder()
                                .addType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                                .addType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                                .build(m_vk, m_device, VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT, 1u);

    m_dTex.descriptorSetLayout = DescriptorSetLayoutBuilder()
                                     .addSingleSamplerBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                              VK_SHADER_STAGE_COMPUTE_BIT, &m_dTex.sampler.get())
                                     .addSingleBinding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT)
                                     .build(m_vk, m_device);

    m_dTex.pipelineLayout = makePipelineLayout(m_vk, m_device, *m_dTex.descriptorSetLayout);
    m_dTex.descriptorSet  = makeDescriptorSet(m_vk, m_device, *m_dTex.descriptorPool, *m_dTex.descriptorSetLayout);
    m_dTex.inImageDescriptorInfo =
        makeDescriptorImageInfo(VK_NULL_HANDLE, *m_dTex.inImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_dTex.outImageDescriptorInfo =
        makeDescriptorImageInfo(VK_NULL_HANDLE, *m_dTex.outImageView, VK_IMAGE_LAYOUT_GENERAL);
    m_dTex.computeModule =
        createShaderModule(m_vk, m_device, context.getBinaryCollection().get("downloadTextureComp"), 0);
    m_dTex.computePipeline = makeComputePipeline(m_vk, m_device, *m_dTex.pipelineLayout, *m_dTex.computeModule);

    DescriptorSetUpdateBuilder()
        .writeSingle(*m_dTex.descriptorSet, DescriptorSetUpdateBuilder::Location::binding(0u),
                     VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &m_dTex.inImageDescriptorInfo)
        .writeSingle(*m_dTex.descriptorSet, DescriptorSetUpdateBuilder::Location::binding(1u),
                     VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &m_dTex.outImageDescriptorInfo)
        .update(m_vk, m_device);

    // Transition images for shader access (texture / imageStore)
    const VkImageMemoryBarrier imageBarriers[] = {
        {
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,            // VkStructureType sType;
            nullptr,                                           // const void* pNext;
            (VkAccessFlags)m_imageUploadAccessMask,            // VkAccessFlags srcAccessMask;
            (VkAccessFlags)VK_ACCESS_SHADER_READ_BIT,          // VkAccessFlags dstAccessMask;
            m_imageLayoutAfterUpload,                          // VkImageLayout oldLayout;
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,          // VkImageLayout newLayout;
            VK_QUEUE_FAMILY_IGNORED,                           // uint32_t srcQueueFamilyIndex;
            VK_QUEUE_FAMILY_IGNORED,                           // uint32_t destQueueFamilyIndex;
            m_image,                                           // VkImage image;
            makeColorSubresourceRange(0, m_caseDef.numLayers), // VkImageSubresourceRange subresourceRange;
        },
        {
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,            // VkStructureType sType;
            nullptr,                                           // const void* pNext;
            (VkAccessFlags)0,                                  // VkAccessFlags srcAccessMask;
            (VkAccessFlags)VK_ACCESS_SHADER_WRITE_BIT,         // VkAccessFlags dstAccessMask;
            VK_IMAGE_LAYOUT_UNDEFINED,                         // VkImageLayout oldLayout;
            VK_IMAGE_LAYOUT_GENERAL,                           // VkImageLayout newLayout;
            VK_QUEUE_FAMILY_IGNORED,                           // uint32_t srcQueueFamilyIndex;
            VK_QUEUE_FAMILY_IGNORED,                           // uint32_t destQueueFamilyIndex;
            *m_dTex.outImage,                                  // VkImage image;
            makeColorSubresourceRange(0, m_caseDef.numLayers), // VkImageSubresourceRange subresourceRange;
        }};

    m_vk.cmdPipelineBarrier(*m_cmdBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u,
                            0u, nullptr, 0u, nullptr, 2u, imageBarriers);

    // Dispatch
    m_vk.cmdBindPipeline(*m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, *m_dTex.computePipeline);
    m_vk.cmdBindDescriptorSets(*m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, *m_dTex.pipelineLayout, 0u, 1u,
                               &m_dTex.descriptorSet.get(), 0u, nullptr);
    m_vk.cmdDispatch(*m_cmdBuffer, m_caseDef.size.x(), m_caseDef.size.y(), m_caseDef.numLayers);

    // Copy output image to color buffer
    copyImageToBuffer(*m_dTex.outImage, buffer, m_caseDef.size, VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL,
                      m_caseDef.numLayers);
}

void UploadDownloadExecutor::downloadLoad(Context &context, VkBuffer buffer)
{
    // Create output image with download result
    const VkImageUsageFlags usageFlags = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    m_dLoad.outImage      = makeImage(m_vk, m_device, 0u, VK_IMAGE_TYPE_2D, m_caseDef.viewFormat, m_caseDef.viewFormat,
                                      false, m_caseDef.size, 1u, m_caseDef.numLayers, usageFlags);
    m_dLoad.outImageAlloc = bindImage(m_vk, m_device, m_allocator, *m_dLoad.outImage, MemoryRequirement::Any);
    m_dLoad.outImageView  = makeImageView(m_vk, m_device, *m_dLoad.outImage, getImageViewType(m_caseDef.imageType),
                                          m_caseDef.viewFormat, makeColorSubresourceRange(0, m_caseDef.numLayers));

    const vk::VkImageViewUsageCreateInfo viewUsageCreateInfo = {
        VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO, // VkStructureType        sType
        nullptr,                                        // const void*            pNext
        VK_IMAGE_USAGE_STORAGE_BIT,                     // VkImageUsageFlags usage;
    };
    m_dLoad.inImageView = makeImageView(m_vk, m_device, m_image, getImageViewType(m_caseDef.imageType),
                                        m_caseDef.viewFormat, makeColorSubresourceRange(0, m_caseDef.numLayers),
                                        m_haveMaintenance2 ? &viewUsageCreateInfo : nullptr);

    // Setup compute pipeline
    m_dLoad.descriptorPool = DescriptorPoolBuilder()
                                 .addType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2u)
                                 .build(m_vk, m_device, VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT, 1u);

    m_dLoad.descriptorSetLayout = DescriptorSetLayoutBuilder()
                                      .addSingleBinding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT)
                                      .addSingleBinding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT)
                                      .build(m_vk, m_device);

    m_dLoad.pipelineLayout = makePipelineLayout(m_vk, m_device, *m_dLoad.descriptorSetLayout);
    m_dLoad.descriptorSet  = makeDescriptorSet(m_vk, m_device, *m_dLoad.descriptorPool, *m_dLoad.descriptorSetLayout);
    m_dLoad.inImageDescriptorInfo =
        makeDescriptorImageInfo(VK_NULL_HANDLE, *m_dLoad.inImageView, VK_IMAGE_LAYOUT_GENERAL);
    m_dLoad.outImageDescriptorInfo =
        makeDescriptorImageInfo(VK_NULL_HANDLE, *m_dLoad.outImageView, VK_IMAGE_LAYOUT_GENERAL);
    m_dLoad.computeModule =
        createShaderModule(m_vk, m_device, context.getBinaryCollection().get("downloadLoadComp"), 0);
    m_dLoad.computePipeline = makeComputePipeline(m_vk, m_device, *m_dLoad.pipelineLayout, *m_dLoad.computeModule);

    DescriptorSetUpdateBuilder()
        .writeSingle(*m_dLoad.descriptorSet, DescriptorSetUpdateBuilder::Location::binding(0u),
                     VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &m_dLoad.inImageDescriptorInfo)
        .writeSingle(*m_dLoad.descriptorSet, DescriptorSetUpdateBuilder::Location::binding(1u),
                     VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &m_dLoad.outImageDescriptorInfo)
        .update(m_vk, m_device);

    // Transition storage images for shader access (imageLoad/Store)
    VkImageLayout requiredImageLayout          = VK_IMAGE_LAYOUT_GENERAL;
    const VkImageMemoryBarrier imageBarriers[] = {
        {
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,            // VkStructureType sType;
            nullptr,                                           // const void* pNext;
            (VkAccessFlags)m_imageUploadAccessMask,            // VkAccessFlags srcAccessMask;
            (VkAccessFlags)VK_ACCESS_SHADER_READ_BIT,          // VkAccessFlags dstAccessMask;
            m_imageLayoutAfterUpload,                          // VkImageLayout oldLayout;
            requiredImageLayout,                               // VkImageLayout newLayout;
            VK_QUEUE_FAMILY_IGNORED,                           // uint32_t srcQueueFamilyIndex;
            VK_QUEUE_FAMILY_IGNORED,                           // uint32_t destQueueFamilyIndex;
            m_image,                                           // VkImage image;
            makeColorSubresourceRange(0, m_caseDef.numLayers), // VkImageSubresourceRange subresourceRange;
        },
        {
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,            // VkStructureType sType;
            nullptr,                                           // const void* pNext;
            (VkAccessFlags)0,                                  // VkAccessFlags srcAccessMask;
            (VkAccessFlags)VK_ACCESS_SHADER_WRITE_BIT,         // VkAccessFlags dstAccessMask;
            VK_IMAGE_LAYOUT_UNDEFINED,                         // VkImageLayout oldLayout;
            requiredImageLayout,                               // VkImageLayout newLayout;
            VK_QUEUE_FAMILY_IGNORED,                           // uint32_t srcQueueFamilyIndex;
            VK_QUEUE_FAMILY_IGNORED,                           // uint32_t destQueueFamilyIndex;
            *m_dLoad.outImage,                                 // VkImage image;
            makeColorSubresourceRange(0, m_caseDef.numLayers), // VkImageSubresourceRange subresourceRange;
        }};

    m_vk.cmdPipelineBarrier(*m_cmdBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u,
                            0u, nullptr, 0u, nullptr, 2u, imageBarriers);

    // Dispatch
    m_vk.cmdBindPipeline(*m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, *m_dLoad.computePipeline);
    m_vk.cmdBindDescriptorSets(*m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, *m_dLoad.pipelineLayout, 0u, 1u,
                               &m_dLoad.descriptorSet.get(), 0u, nullptr);
    m_vk.cmdDispatch(*m_cmdBuffer, m_caseDef.size.x(), m_caseDef.size.y(), m_caseDef.numLayers);

    // Copy output image to color buffer
    copyImageToBuffer(*m_dLoad.outImage, buffer, m_caseDef.size, VK_ACCESS_SHADER_WRITE_BIT, requiredImageLayout,
                      m_caseDef.numLayers);
}

void UploadDownloadExecutor::copyImageToBuffer(VkImage sourceImage, VkBuffer buffer, const IVec3 size,
                                               const VkAccessFlags srcAccessMask, const VkImageLayout oldLayout,
                                               const uint32_t numLayers)
{
    // Copy result to host visible buffer for inspection
    const VkImageMemoryBarrier imageBarrier = {
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, // VkStructureType sType;
        nullptr,                                // const void* pNext;
        srcAccessMask,                          // VkAccessFlags srcAccessMask;
        VK_ACCESS_TRANSFER_READ_BIT,            // VkAccessFlags dstAccessMask;
        oldLayout,                              // VkImageLayout oldLayout;
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,   // VkImageLayout newLayout;
        VK_QUEUE_FAMILY_IGNORED,                // uint32_t srcQueueFamilyIndex;
        VK_QUEUE_FAMILY_IGNORED,                // uint32_t destQueueFamilyIndex;
        sourceImage,                            // VkImage image;
        makeColorSubresourceRange(0, numLayers) // VkImageSubresourceRange subresourceRange;
    };

    m_vk.cmdPipelineBarrier(*m_cmdBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0u, 0u,
                            nullptr, 0u, nullptr, 1u, &imageBarrier);

    const VkImageSubresourceLayers subresource = {
        VK_IMAGE_ASPECT_COLOR_BIT, // VkImageAspectFlags    aspectMask;
        0u,                        // uint32_t              mipLevel;
        0u,                        // uint32_t              baseArrayLayer;
        numLayers,                 // uint32_t              layerCount;
    };

    const VkBufferImageCopy region = {
        0ull,                  // VkDeviceSize                bufferOffset;
        0u,                    // uint32_t                    bufferRowLength;
        0u,                    // uint32_t                    bufferImageHeight;
        subresource,           // VkImageSubresourceLayers    imageSubresource;
        makeOffset3D(0, 0, 0), // VkOffset3D                  imageOffset;
        makeExtent3D(size),    // VkExtent3D                  imageExtent;
    };

    m_vk.cmdCopyImageToBuffer(*m_cmdBuffer, sourceImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1u, &region);

    const VkBufferMemoryBarrier bufferBarrier = {
        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, // VkStructureType    sType;
        nullptr,                                 // const void*        pNext;
        VK_ACCESS_TRANSFER_WRITE_BIT,            // VkAccessFlags      srcAccessMask;
        VK_ACCESS_HOST_READ_BIT,                 // VkAccessFlags      dstAccessMask;
        VK_QUEUE_FAMILY_IGNORED,                 // uint32_t           srcQueueFamilyIndex;
        VK_QUEUE_FAMILY_IGNORED,                 // uint32_t           dstQueueFamilyIndex;
        buffer,                                  // VkBuffer           buffer;
        0ull,                                    // VkDeviceSize       offset;
        VK_WHOLE_SIZE,                           // VkDeviceSize       size;
    };

    m_vk.cmdPipelineBarrier(*m_cmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0u, 0u, nullptr,
                            1u, &bufferBarrier, 0u, nullptr);
}

tcu::TestStatus testMutable(Context &context, const CaseDef caseDef)
{
    const DeviceInterface &vk = context.getDeviceInterface();
    const VkDevice device     = context.getDevice();
    Allocator &allocator      = context.getDefaultAllocator();

    // Create a color buffer for host-inspection of results
    // For the Copy download method, this is the target of the download, for other
    // download methods, pixel data will be copied to this buffer from the download
    // target
    const VkDeviceSize colorBufferSize = caseDef.size.x() * caseDef.size.y() * caseDef.size.z() * caseDef.numLayers *
                                         tcu::getPixelSize(mapVkFormat(caseDef.imageFormat));
    const Unique<VkBuffer> colorBuffer(makeBuffer(vk, device, colorBufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT));
    const UniquePtr<Allocation> colorBufferAlloc(
        bindBuffer(vk, device, allocator, *colorBuffer, MemoryRequirement::HostVisible));
    deMemset(colorBufferAlloc->getHostPtr(), 0, static_cast<std::size_t>(colorBufferSize));
    flushAlloc(vk, device, *colorBufferAlloc);

    // Execute the test
    UploadDownloadExecutor executor(context, context.isDeviceFunctionalitySupported("VK_KHR_maintenance2"),
                                    context.getDeviceInterface(), device, context.getUniversalQueue(),
                                    context.getUniversalQueueFamilyIndex(), caseDef);
    executor.run(context, *colorBuffer);

    // Verify results
    {
        invalidateAlloc(vk, device, *colorBufferAlloc);

        // For verification purposes, we use the format of the upload to generate the expected image
        const VkFormat format =
            caseDef.upload == UPLOAD_CLEAR || caseDef.upload == UPLOAD_COPY ? caseDef.imageFormat : caseDef.viewFormat;
        const tcu::TextureFormat tcuFormat = mapVkFormat(format);
        const bool isIntegerFormat         = isUintFormat(format) || isIntFormat(format);
        const tcu::ConstPixelBufferAccess resultImage(tcuFormat, caseDef.size.x(), caseDef.size.y(), caseDef.numLayers,
                                                      colorBufferAlloc->getHostPtr());
        tcu::TextureLevel textureLevel(tcuFormat, caseDef.size.x(), caseDef.size.y(), caseDef.numLayers);
        const tcu::PixelBufferAccess expectedImage = textureLevel.getAccess();
        generateExpectedImage(expectedImage, caseDef);

        bool ok;
        if (isIntegerFormat)
            ok = tcu::intThresholdCompare(context.getTestContext().getLog(), "Image comparison", "", expectedImage,
                                          resultImage, tcu::UVec4(1), tcu::COMPARE_LOG_RESULT);
        else
            ok = tcu::floatThresholdCompare(context.getTestContext().getLog(), "Image comparison", "", expectedImage,
                                            resultImage, tcu::Vec4(0.01f), tcu::COMPARE_LOG_RESULT);
        return ok ? tcu::TestStatus::pass("Pass") : tcu::TestStatus::fail("Fail");
    }
}

void checkSupport(Context &context, const CaseDef caseDef)
{
    const InstanceInterface &vki      = context.getInstanceInterface();
    const VkPhysicalDevice physDevice = context.getPhysicalDevice();

    // If this is a VK_KHR_image_format_list test, check that the extension is supported
    if (caseDef.isFormatListTest)
        context.requireDeviceFunctionality("VK_KHR_image_format_list");

    // Check required features on the format for the required upload/download methods
    VkFormatProperties imageFormatProps, viewFormatProps;
    vki.getPhysicalDeviceFormatProperties(physDevice, caseDef.imageFormat, &imageFormatProps);
    vki.getPhysicalDeviceFormatProperties(physDevice, caseDef.viewFormat, &viewFormatProps);

    VkFormatFeatureFlags viewFormatFeatureFlags = 0u;
    switch (caseDef.upload)
    {
    case UPLOAD_DRAW:
        viewFormatFeatureFlags |= VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT;
        break;
    case UPLOAD_STORE:
        viewFormatFeatureFlags |= VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
        break;
    case UPLOAD_CLEAR:
        viewFormatFeatureFlags |= VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
        break;
    case UPLOAD_COPY:
        viewFormatFeatureFlags |= VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
        break;
    default:
        DE_FATAL("Invalid upload method");
        break;
    }
    switch (caseDef.download)
    {
    case DOWNLOAD_TEXTURE:
        viewFormatFeatureFlags |= VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
        // For the texture case we write the samples read to a separate output image with the same view format
        // so we need to check that we can also use the view format for storage
        viewFormatFeatureFlags |= VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
        break;
    case DOWNLOAD_LOAD:
        viewFormatFeatureFlags |= VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
        break;
    case DOWNLOAD_COPY:
        viewFormatFeatureFlags |= VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
        break;
    default:
        DE_FATAL("Invalid download method");
        break;
    }

    if ((viewFormatProps.optimalTilingFeatures & viewFormatFeatureFlags) != viewFormatFeatureFlags)
        TCU_THROW(NotSupportedError, "View format doesn't support upload/download method");

    const bool haveMaintenance2 = context.isDeviceFunctionalitySupported("VK_KHR_maintenance2");

    // We don't use the base image for anything other than transfer
    // operations so there are no features to check.  However, The Vulkan
    // 1.0 spec does not allow us to create an image view with usage that
    // is not supported by the main format.  With VK_KHR_maintenance2, we
    // can do this via VK_IMAGE_CREATE_EXTENDED_USAGE_BIT_KHR.
    if ((imageFormatProps.optimalTilingFeatures & viewFormatFeatureFlags) != viewFormatFeatureFlags &&
        !haveMaintenance2)
    {
        TCU_THROW(NotSupportedError, "Image format doesn't support upload/download method");
    }

    // If no format feature flags are supported, the format itself is not supported,
    // and images of that format cannot be created.
    if (imageFormatProps.optimalTilingFeatures == 0)
    {
        TCU_THROW(NotSupportedError, "Base image format is not supported");
    }

    const vk::VkImageUsageFlags usage = getImageUsageForTestCase(caseDef);
    const VkImageCreateFlags imageFlags =
        VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | (haveMaintenance2 ? VK_IMAGE_CREATE_EXTENDED_USAGE_BIT : 0);

    if (getMaxAvailableSampleCount(context, caseDef.imageFormat, getImageType(caseDef.imageType), usage, imageFlags) ==
        VK_SAMPLE_COUNT_1_BIT)
        TCU_THROW(NotSupportedError, "Maximum available sample count is VK_SAMPLE_COUNT_1_BIT");
}

tcu::TestCaseGroup *createImageMutableTests(TestContext &testCtx)
{
    de::MovePtr<TestCaseGroup> testGroup(new TestCaseGroup(testCtx, "mutable"));
    for (int textureNdx = 0; textureNdx < DE_LENGTH_OF_ARRAY(s_textures); ++textureNdx)
    {
        const Texture &texture = s_textures[textureNdx];
        de::MovePtr<tcu::TestCaseGroup> groupByImageViewType(
            new tcu::TestCaseGroup(testCtx, getImageTypeName(texture.type()).c_str()));

        for (int imageFormatNdx = 0; imageFormatNdx < DE_LENGTH_OF_ARRAY(s_formats); ++imageFormatNdx)
            for (int viewFormatNdx = 0; viewFormatNdx < DE_LENGTH_OF_ARRAY(s_formats); ++viewFormatNdx)
            {
                if (imageFormatNdx != viewFormatNdx &&
                    formatsAreCompatible(s_formats[imageFormatNdx], s_formats[viewFormatNdx]))
                {
                    for (int upload = 0; upload < UPLOAD_LAST; upload++)
                    {
                        if (upload == UPLOAD_STORE && !isFormatImageLoadStoreCapable(s_formats[viewFormatNdx]))
                            continue;

                        for (int download = 0; download < DOWNLOAD_LAST; download++)
                        {
                            if ((download == DOWNLOAD_LOAD || download == DOWNLOAD_TEXTURE) &&
                                !isFormatImageLoadStoreCapable(s_formats[viewFormatNdx]))
                                continue;

                            CaseDef caseDef = {
                                texture.type(),
                                texture.layerSize(),
                                static_cast<uint32_t>(texture.numLayers()),
                                s_formats[imageFormatNdx],
                                s_formats[viewFormatNdx],
                                static_cast<enum Upload>(upload),
                                static_cast<enum Download>(download),
                                false,                                   // isFormatListTest;
                                vk::wsi::TYPE_LAST,                      // wsiType
                                ResolveAttachmentTestType::RA_TEST_NONE, // resolveAttachmentTestType
                                false                                    // isLoadOpClearTest
                            };

                            std::string caseName = getFormatShortString(s_formats[imageFormatNdx]) + "_" +
                                                   getFormatShortString(s_formats[viewFormatNdx]) + "_" +
                                                   getUploadString(upload) + "_" + getDownloadString(download);
                            addFunctionCaseWithPrograms(groupByImageViewType.get(), caseName, checkSupport,
                                                        initPrograms, testMutable, caseDef);

                            caseDef.isFormatListTest = true;
                            caseName += "_format_list";
                            addFunctionCaseWithPrograms(groupByImageViewType.get(), caseName, checkSupport,
                                                        initPrograms, testMutable, caseDef);
                        }
                    }

                    // Multisampling and resolve attachment tests
                    {
                        // Both multisampled image and resolve attachment are mutable
                        CaseDef caseDef = {
                            texture.type(),
                            texture.layerSize(),
                            static_cast<uint32_t>(texture.numLayers()),
                            s_formats[imageFormatNdx],
                            s_formats[viewFormatNdx],
                            UPLOAD_DRAW,
                            DOWNLOAD_COPY,
                            false,                                          // isFormatListTest;
                            vk::wsi::TYPE_LAST,                             // wsiType
                            ResolveAttachmentTestType::RA_TEST_ALL_MUTABLE, // resolveAttachmentTestType
                            false                                           // isLoadOpClearTest
                        };

                        std::string baseCaseName = getFormatShortString(s_formats[imageFormatNdx]) + "_" +
                                                   getFormatShortString(s_formats[viewFormatNdx]) + "_" +
                                                   getUploadString(UPLOAD_DRAW) + "_" +
                                                   getDownloadString(DOWNLOAD_COPY) + "_resolve";
                        addFunctionCaseWithPrograms(groupByImageViewType.get(), baseCaseName, checkSupport,
                                                    initPrograms, testMutable, caseDef);

                        // Resolve attachment is mutable and color attachment is non-mutable
                        {
                            caseDef.resolveAttachmentTestType = ResolveAttachmentTestType::RA_TEST_RA_MUTABLE;

                            std::string case1Name = baseCaseName + "_mutable_resolve_att";
                            addFunctionCaseWithPrograms(groupByImageViewType.get(), case1Name, checkSupport,
                                                        initPrograms, testMutable, caseDef);
                        }

                        // Color attachment is mutable and resolve attachment is non-mutable
                        {
                            caseDef.resolveAttachmentTestType = ResolveAttachmentTestType::RA_TEST_CA_MUTABLE;

                            std::string case2Name = baseCaseName + "_mutable_color_att";
                            addFunctionCaseWithPrograms(groupByImageViewType.get(), case2Name, checkSupport,
                                                        initPrograms, testMutable, caseDef);
                        }
                    }

                    // VK_ATTACHMENT_LOAD_OP_CLEAR tests
                    {
                        if (texture.numLayers() > 1)
                            continue;

                        CaseDef caseDef = {
                            texture.type(),
                            texture.layerSize(),
                            static_cast<uint32_t>(texture.numLayers()),
                            s_formats[imageFormatNdx],
                            s_formats[viewFormatNdx],
                            UPLOAD_DRAW,
                            DOWNLOAD_COPY,
                            false,                                   // isFormatListTest;
                            vk::wsi::TYPE_LAST,                      // wsiType
                            ResolveAttachmentTestType::RA_TEST_NONE, // resolveAttachmentTestType
                            true                                     // isLoadOpClearTest
                        };

                        std::string caseName = getFormatShortString(s_formats[imageFormatNdx]) + "_" +
                                               getFormatShortString(s_formats[viewFormatNdx]) + "_" +
                                               getUploadString(UPLOAD_DRAW) + "_" + getDownloadString(DOWNLOAD_COPY) +
                                               "_load_op_clear";
                        addFunctionCaseWithPrograms(groupByImageViewType.get(), caseName, checkSupport, initPrograms,
                                                    testMutable, caseDef);
                    }
                }
            }

        testGroup->addChild(groupByImageViewType.release());
    }

    return testGroup.release();
}

typedef vector<VkExtensionProperties> Extensions;

void checkAllSupported(const Extensions &supportedExtensions, const vector<string> &requiredExtensions)
{
    for (vector<string>::const_iterator requiredExtName = requiredExtensions.begin();
         requiredExtName != requiredExtensions.end(); ++requiredExtName)
    {
        if (!isExtensionStructSupported(supportedExtensions, RequiredExtension(*requiredExtName)))
            TCU_THROW(NotSupportedError, (*requiredExtName + " is not supported").c_str());
    }
}

CustomInstance createInstanceWithWsi(Context &context, const Extensions &supportedExtensions, Type wsiType,
                                     const VkAllocationCallbacks *pAllocator = nullptr)
{
    vector<string> extensions;

    extensions.push_back("VK_KHR_surface");
    extensions.push_back(getExtensionName(wsiType));
    if (isDisplaySurface(wsiType))
        extensions.push_back("VK_KHR_display");

    // VK_EXT_swapchain_colorspace adds new surface formats. Driver can enumerate
    // the formats regardless of whether VK_EXT_swapchain_colorspace was enabled,
    // but using them without enabling the extension is not allowed. Thus we have
    // two options:
    //
    // 1) Filter out non-core formats to stay within valid usage.
    //
    // 2) Enable VK_EXT_swapchain colorspace if advertised by the driver.
    //
    // We opt for (2) as it provides basic coverage for the extension as a bonus.
    if (isExtensionStructSupported(supportedExtensions, RequiredExtension("VK_EXT_swapchain_colorspace")))
        extensions.push_back("VK_EXT_swapchain_colorspace");

    checkAllSupported(supportedExtensions, extensions);

    return createCustomInstanceWithExtensions(context, extensions, pAllocator);
}

Move<VkDevice> createDeviceWithWsi(const PlatformInterface &vkp, VkInstance instance, const InstanceInterface &vki,
                                   VkPhysicalDevice physicalDevice, const Extensions &supportedExtensions,
                                   const uint32_t queueFamilyIndex, const VkAllocationCallbacks *pAllocator,
#ifdef CTS_USES_VULKANSC
                                   de::SharedPtr<vk::ResourceInterface> resourceInterface,
#endif // CTS_USES_VULKANSC
                                   const tcu::CommandLine &cmdLine)
{
    const float queuePriorities[]              = {1.0f};
    const VkDeviceQueueCreateInfo queueInfos[] = {{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr,
                                                   (VkDeviceQueueCreateFlags)0, queueFamilyIndex,
                                                   DE_LENGTH_OF_ARRAY(queuePriorities), &queuePriorities[0]}};
    VkPhysicalDeviceFeatures features;
    deMemset(&features, 0x0, sizeof(features));

    const char *const extensions[] = {"VK_KHR_swapchain", "VK_KHR_swapchain_mutable_format"};

    void *pNext = nullptr;
#ifdef CTS_USES_VULKANSC
    VkDeviceObjectReservationCreateInfo memReservationInfo =
        cmdLine.isSubProcess() ? resourceInterface->getStatMax() : resetDeviceObjectReservationCreateInfo();
    memReservationInfo.pNext = pNext;
    pNext                    = &memReservationInfo;

    VkPhysicalDeviceVulkanSC10Features sc10Features = createDefaultSC10Features();
    sc10Features.pNext                              = pNext;
    pNext                                           = &sc10Features;

    VkPipelineCacheCreateInfo pcCI;
    std::vector<VkPipelinePoolSize> poolSizes;
    if (cmdLine.isSubProcess())
    {
        if (resourceInterface->getCacheDataSize() > 0)
        {
            pcCI = {
                VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO, // VkStructureType sType;
                nullptr,                                      // const void* pNext;
                VK_PIPELINE_CACHE_CREATE_READ_ONLY_BIT |
                    VK_PIPELINE_CACHE_CREATE_USE_APPLICATION_STORAGE_BIT, // VkPipelineCacheCreateFlags flags;
                resourceInterface->getCacheDataSize(),                    // uintptr_t initialDataSize;
                resourceInterface->getCacheData()                         // const void* pInitialData;
            };
            memReservationInfo.pipelineCacheCreateInfoCount = 1;
            memReservationInfo.pPipelineCacheCreateInfos    = &pcCI;
        }

        poolSizes = resourceInterface->getPipelinePoolSizes();
        if (!poolSizes.empty())
        {
            memReservationInfo.pipelinePoolSizeCount = uint32_t(poolSizes.size());
            memReservationInfo.pPipelinePoolSizes    = poolSizes.data();
        }
    }
#endif // CTS_USES_VULKANSC

    const VkDeviceCreateInfo deviceParams = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                                             pNext,
                                             (VkDeviceCreateFlags)0,
                                             DE_LENGTH_OF_ARRAY(queueInfos),
                                             &queueInfos[0],
                                             0u,                             // enabledLayerCount
                                             nullptr,                        // ppEnabledLayerNames
                                             DE_LENGTH_OF_ARRAY(extensions), // enabledExtensionCount
                                             DE_ARRAY_BEGIN(extensions),     // ppEnabledExtensionNames
                                             &features};

    for (int ndx = 0; ndx < DE_LENGTH_OF_ARRAY(extensions); ++ndx)
    {
        if (!isExtensionStructSupported(supportedExtensions, RequiredExtension(extensions[ndx])))
            TCU_THROW(NotSupportedError, (string(extensions[ndx]) + " is not supported").c_str());
    }

    return createCustomDevice(cmdLine.isValidationEnabled(), vkp, instance, vki, physicalDevice, &deviceParams,
                              pAllocator);
}

struct InstanceHelper
{
    const vector<VkExtensionProperties> supportedExtensions;
    const CustomInstance instance;
    const InstanceDriver &vki;

    InstanceHelper(Context &context, Type wsiType, const VkAllocationCallbacks *pAllocator = nullptr)
        : supportedExtensions(enumerateInstanceExtensionProperties(context.getPlatformInterface(), nullptr))
        , instance(createInstanceWithWsi(context, supportedExtensions, wsiType, pAllocator))
        , vki(instance.getDriver())
    {
    }
};

struct DeviceHelper
{
    const VkPhysicalDevice physicalDevice;
    const uint32_t queueFamilyIndex;
    const Unique<VkDevice> device;
    const DeviceDriver vkd;
    const VkQueue queue;

    DeviceHelper(Context &context, const InstanceInterface &vki, VkInstance instance, VkSurfaceKHR surface,
                 const VkAllocationCallbacks *pAllocator = nullptr)
        : physicalDevice(chooseDevice(vki, instance, context.getTestContext().getCommandLine()))
        , queueFamilyIndex(chooseQueueFamilyIndex(vki, physicalDevice, surface))
        , device(createDeviceWithWsi(context.getPlatformInterface(), context.getInstance(), vki, physicalDevice,
                                     enumerateDeviceExtensionProperties(vki, physicalDevice, nullptr), queueFamilyIndex,
                                     pAllocator,
#ifdef CTS_USES_VULKANSC
                                     context.getResourceInterface(),
#endif // CTS_USES_VULKANSC
                                     context.getTestContext().getCommandLine()))
        , vkd(context.getPlatformInterface(), context.getInstance(), *device, context.getUsedApiVersion(),
              context.getTestContext().getCommandLine())
        , queue(getDeviceQueue(vkd, *device, queueFamilyIndex, 0))
    {
    }
};

MovePtr<Display> createDisplay(const vk::Platform &platform, const Extensions &supportedExtensions, Type wsiType)
{
    try
    {
        return MovePtr<Display>(platform.createWsiDisplay(wsiType));
    }
    catch (const tcu::NotSupportedError &e)
    {
        if (isExtensionStructSupported(supportedExtensions, RequiredExtension(getExtensionName(wsiType))) &&
            platform.hasDisplay(wsiType))
        {
            // If VK_KHR_{platform}_surface was supported, vk::Platform implementation
            // must support creating native display & window for that WSI type.
            throw tcu::TestError(e.getMessage());
        }
        else
            throw;
    }
}

MovePtr<Window> createWindow(const Display &display, const Maybe<UVec2> &initialSize)
{
    try
    {
        return MovePtr<Window>(display.createWindow(initialSize));
    }
    catch (const tcu::NotSupportedError &e)
    {
        // See createDisplay - assuming that wsi::Display was supported platform port
        // should also support creating a window.
        throw tcu::TestError(e.getMessage());
    }
}

struct NativeObjects
{
    const UniquePtr<Display> display;
    const UniquePtr<Window> window;

    NativeObjects(Context &context, const Extensions &supportedExtensions, Type wsiType,
                  const Maybe<UVec2> &initialWindowSize = tcu::Nothing)
        : display(
              createDisplay(context.getTestContext().getPlatform().getVulkanPlatform(), supportedExtensions, wsiType))
        , window(createWindow(*display, initialWindowSize))
    {
    }
};

Move<VkSwapchainKHR> makeSwapchain(const DeviceInterface &vk, const VkDevice device, const vk::wsi::Type wsiType,
                                   const VkSurfaceKHR surface, const VkSurfaceCapabilitiesKHR capabilities,
                                   const VkSurfaceFormatKHR surfaceFormat, const VkFormat viewFormat,
                                   const uint32_t numLayers, const VkImageUsageFlags usage,
                                   const tcu::UVec2 &desiredSize, uint32_t desiredImageCount)
{
    const VkFormat formatList[2] = {surfaceFormat.format, viewFormat};

    const VkImageFormatListCreateInfo formatListInfo = {
        VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO, // VkStructureType sType;
        nullptr,                                         // const void* pNext;
        2u,                                              // uint32_t                    viewFormatCount
        formatList                                       // const VkFormat*            pViewFormats
    };

    const VkSurfaceTransformFlagBitsKHR transform =
        (capabilities.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR) ?
            VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR :
            capabilities.currentTransform;
    const PlatformProperties &platformProperties = getPlatformProperties(wsiType);

    const VkSwapchainCreateInfoKHR swapchainInfo = {
        VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR, // VkStructureType sType;
        &formatListInfo,                             // const void* pNext;
        VK_SWAPCHAIN_CREATE_MUTABLE_FORMAT_BIT_KHR,  // VkSwapchainCreateFlagsKHR flags;
        surface,                                     // VkSurfaceKHR surface;
        de::clamp(desiredImageCount, capabilities.minImageCount,
                  capabilities.maxImageCount > 0 ?
                      capabilities.maxImageCount :
                      capabilities.minImageCount + desiredImageCount), // uint32_t minImageCount;
        surfaceFormat.format,                                          // VkFormat imageFormat;
        surfaceFormat.colorSpace,                                      // VkColorSpaceKHR imageColorSpace;
        (platformProperties.swapchainExtent == PlatformProperties::SWAPCHAIN_EXTENT_MUST_MATCH_WINDOW_SIZE ?
             capabilities.currentExtent :
             vk::makeExtent2D(desiredSize.x(), desiredSize.y())), // VkExtent2D imageExtent;
        numLayers,                                                // uint32_t imageArrayLayers;
        usage,                                                    // VkImageUsageFlags imageUsage;
        VK_SHARING_MODE_EXCLUSIVE,                                // VkSharingMode imageSharingMode;
        0u,                                                       // uint32_t queueFamilyIndexCount;
        nullptr,                                                  // const uint32_t* pQueueFamilyIndices;
        transform,                                                // VkSurfaceTransformFlagBitsKHR preTransform;
        VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,                        // VkCompositeAlphaFlagBitsKHR compositeAlpha;
        VK_PRESENT_MODE_FIFO_KHR,                                 // VkPresentModeKHR presentMode;
        VK_FALSE,                                                 // VkBool32 clipped;
        VK_NULL_HANDLE                                            // VkSwapchainKHR oldSwapchain;
    };

    return createSwapchainKHR(vk, device, &swapchainInfo);
}

tcu::TestStatus testSwapchainMutable(Context &context, CaseDef caseDef)
{
    const Type wsiType(caseDef.wsiType);
    const tcu::UVec2 desiredSize(256, 256);
    const InstanceHelper instHelper(context, wsiType);
    const NativeObjects native(context, instHelper.supportedExtensions, wsiType, tcu::just(desiredSize));
    const Unique<VkSurfaceKHR> surface(createSurface(instHelper.vki, instHelper.instance, wsiType, *native.display,
                                                     *native.window, context.getTestContext().getCommandLine()));
    const DeviceHelper devHelper(context, instHelper.vki, instHelper.instance, *surface);
    const DeviceInterface &vk         = devHelper.vkd;
    const InstanceDriver &vki         = instHelper.vki;
    const VkDevice device             = *devHelper.device;
    const VkPhysicalDevice physDevice = devHelper.physicalDevice;
    SimpleAllocator allocator(vk, device, getPhysicalDeviceMemoryProperties(vki, context.getPhysicalDevice()));

    const VkImageUsageFlags imageUsage = getImageUsageForTestCase(caseDef);

    {
        VkImageFormatProperties properties;
        VkResult result;

        result =
            vki.getPhysicalDeviceImageFormatProperties(physDevice, caseDef.imageFormat, getImageType(caseDef.imageType),
                                                       VK_IMAGE_TILING_OPTIMAL, imageUsage, 0, &properties);

        if (result == VK_ERROR_FORMAT_NOT_SUPPORTED)
        {
            TCU_THROW(NotSupportedError, "Image format is not supported for required usage");
        }

        result =
            vki.getPhysicalDeviceImageFormatProperties(physDevice, caseDef.viewFormat, getImageType(caseDef.imageType),
                                                       VK_IMAGE_TILING_OPTIMAL, imageUsage, 0, &properties);

        if (result == VK_ERROR_FORMAT_NOT_SUPPORTED)
        {
            TCU_THROW(NotSupportedError, "Image view format is not supported for required usage");
        }
    }

    const VkSurfaceCapabilitiesKHR capabilities = getPhysicalDeviceSurfaceCapabilities(vki, physDevice, *surface);

    if (caseDef.numLayers > capabilities.maxImageArrayLayers)
        caseDef.numLayers = capabilities.maxImageArrayLayers;

    // Check support for requested formats by swapchain surface
    const vector<VkSurfaceFormatKHR> surfaceFormats = getPhysicalDeviceSurfaceFormats(vki, physDevice, *surface);

    const VkSurfaceFormatKHR *surfaceFormat = nullptr;
    const VkFormat *viewFormat              = nullptr;

    for (vector<VkSurfaceFormatKHR>::size_type i = 0; i < surfaceFormats.size(); i++)
    {
        if (surfaceFormats[i].format == caseDef.imageFormat)
            surfaceFormat = &surfaceFormats[i];

        if (surfaceFormats[i].format == caseDef.viewFormat)
            viewFormat = &surfaceFormats[i].format;
    }

    if (surfaceFormat == nullptr)
        TCU_THROW(NotSupportedError, "Image format is not supported by swapchain.");

    if (viewFormat == nullptr)
        TCU_THROW(NotSupportedError, "Image view format is not supported by swapchain.");

    if ((capabilities.supportedUsageFlags & imageUsage) != imageUsage)
        TCU_THROW(NotSupportedError, "Image usage request not supported by swapchain.");

    const Unique<VkSwapchainKHR> swapchain(makeSwapchain(vk, device, caseDef.wsiType, *surface, capabilities,
                                                         *surfaceFormat, caseDef.viewFormat, caseDef.numLayers,
                                                         imageUsage, desiredSize, 2));
    const vector<VkImage> swapchainImages = getSwapchainImages(vk, device, *swapchain);

    // Create a color buffer for host-inspection of results
    // For the Copy download method, this is the target of the download, for other
    // download methods, pixel data will be copied to this buffer from the download
    // target
    const VkDeviceSize colorBufferSize = caseDef.size.x() * caseDef.size.y() * caseDef.size.z() * caseDef.numLayers *
                                         tcu::getPixelSize(mapVkFormat(caseDef.imageFormat));
    const Unique<VkBuffer> colorBuffer(makeBuffer(vk, device, colorBufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT));
    const UniquePtr<Allocation> colorBufferAlloc(
        bindBuffer(vk, device, allocator, *colorBuffer, MemoryRequirement::HostVisible));
    deMemset(colorBufferAlloc->getHostPtr(), 0, static_cast<std::size_t>(colorBufferSize));
    flushAlloc(vk, device, *colorBufferAlloc);

    // Execute the test
    UploadDownloadExecutor executor(context, false, vk, device, devHelper.queue, devHelper.queueFamilyIndex, caseDef);

    executor.runSwapchain(context, *colorBuffer, swapchainImages[0]);

    // Verify results
    {
        invalidateAlloc(vk, device, *colorBufferAlloc);

        // For verification purposes, we use the format of the upload to generate the expected image
        const VkFormat format =
            caseDef.upload == UPLOAD_CLEAR || caseDef.upload == UPLOAD_COPY ? caseDef.imageFormat : caseDef.viewFormat;
        const tcu::TextureFormat tcuFormat = mapVkFormat(format);
        const bool isIntegerFormat         = isUintFormat(format) || isIntFormat(format);
        const tcu::ConstPixelBufferAccess resultImage(tcuFormat, caseDef.size.x(), caseDef.size.y(), caseDef.numLayers,
                                                      colorBufferAlloc->getHostPtr());
        tcu::TextureLevel textureLevel(tcuFormat, caseDef.size.x(), caseDef.size.y(), caseDef.numLayers);
        const tcu::PixelBufferAccess expectedImage = textureLevel.getAccess();
        generateExpectedImage(expectedImage, caseDef);

        bool ok;
        if (isIntegerFormat)
            ok = tcu::intThresholdCompare(context.getTestContext().getLog(), "Image comparison", "", expectedImage,
                                          resultImage, tcu::UVec4(1), tcu::COMPARE_LOG_RESULT);
        else
            ok = tcu::floatThresholdCompare(context.getTestContext().getLog(), "Image comparison", "", expectedImage,
                                            resultImage, tcu::Vec4(0.01f), tcu::COMPARE_LOG_RESULT);
        return ok ? tcu::TestStatus::pass("Pass") : tcu::TestStatus::fail("Fail");
    }
}

tcu::TestCaseGroup *createSwapchainImageMutableTests(TestContext &testCtx)
{
    de::MovePtr<TestCaseGroup> testGroup(new TestCaseGroup(testCtx, "swapchain_mutable"));

    for (int typeNdx = 0; typeNdx < vk::wsi::TYPE_LAST; ++typeNdx)
    {
        const vk::wsi::Type wsiType = (vk::wsi::Type)typeNdx;

        de::MovePtr<TestCaseGroup> testGroupWsi(new TestCaseGroup(testCtx, getName(wsiType)));

        for (int textureNdx = 0; textureNdx < DE_LENGTH_OF_ARRAY(s_textures); ++textureNdx)
        {
            const Texture &texture = s_textures[textureNdx];
            de::MovePtr<tcu::TestCaseGroup> groupByImageViewType(
                new tcu::TestCaseGroup(testCtx, getImageTypeName(texture.type()).c_str()));

            for (int imageFormatNdx = 0; imageFormatNdx < DE_LENGTH_OF_ARRAY(s_swapchainFormats); ++imageFormatNdx)
                for (int viewFormatNdx = 0; viewFormatNdx < DE_LENGTH_OF_ARRAY(s_swapchainFormats); ++viewFormatNdx)
                {
                    if (imageFormatNdx != viewFormatNdx &&
                        formatsAreCompatible(s_swapchainFormats[imageFormatNdx], s_swapchainFormats[viewFormatNdx]))
                    {
                        for (int upload = 0; upload < UPLOAD_LAST; upload++)
                        {
                            if (upload == UPLOAD_STORE &&
                                !isFormatImageLoadStoreCapable(s_swapchainFormats[viewFormatNdx]))
                                continue;

                            for (int download = 0; download < DOWNLOAD_LAST; download++)
                            {
                                if ((download == DOWNLOAD_LOAD || download == DOWNLOAD_TEXTURE) &&
                                    !isFormatImageLoadStoreCapable(s_swapchainFormats[viewFormatNdx]))
                                    continue;

                                CaseDef caseDef = {
                                    texture.type(),
                                    texture.layerSize(),
                                    static_cast<uint32_t>(texture.numLayers()),
                                    s_swapchainFormats[imageFormatNdx],
                                    s_swapchainFormats[viewFormatNdx],
                                    static_cast<enum Upload>(upload),
                                    static_cast<enum Download>(download),
                                    true, // isFormatListTest;
                                    wsiType,
                                    ResolveAttachmentTestType::RA_TEST_NONE, // resolveAttachmentTestType
                                    false                                    // isLoadOpClearTest
                                };

                                std::string caseName = getFormatShortString(s_swapchainFormats[imageFormatNdx]) + "_" +
                                                       getFormatShortString(s_swapchainFormats[viewFormatNdx]) + "_" +
                                                       getUploadString(upload) + "_" + getDownloadString(download) +
                                                       "_format_list";

                                addFunctionCaseWithPrograms(groupByImageViewType.get(), caseName, checkSupport,
                                                            initPrograms, testSwapchainMutable, caseDef);
                            }
                        }
                    }
                }

            testGroupWsi->addChild(groupByImageViewType.release());
        }

        testGroup->addChild(testGroupWsi.release());
    }
    return testGroup.release();
}

} // namespace image
} // namespace vkt
