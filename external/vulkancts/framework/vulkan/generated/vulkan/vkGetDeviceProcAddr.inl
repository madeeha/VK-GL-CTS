/* WARNING: This is auto-generated file. Do not modify, since changes will
 * be lost! Modify the generating script instead.
 * This file was generated by /scripts/gen_framework.py
 */

#include "tcuCommandLine.hpp"
#include "vktTestCase.hpp"
#include "vkPlatform.hpp"
#include "vkDeviceUtil.hpp"
#include "vkQueryUtil.hpp"
#include "vktCustomInstancesDevices.hpp"
#include "vktTestCase.hpp"
#include "vktTestCaseUtil.hpp"

namespace vkt
{

using namespace vk;

tcu::TestStatus        testGetDeviceProcAddr        (Context& context)
{
    tcu::TestLog&                                log                        (context.getTestContext().getLog());
    const PlatformInterface&                    platformInterface = context.getPlatformInterface();
    const auto                                    validationEnabled = context.getTestContext().getCommandLine().isValidationEnabled();
    const CustomInstance                        instance                (createCustomInstanceFromContext(context));
    const InstanceDriver&                        instanceDriver = instance.getDriver();
    const VkPhysicalDevice                        physicalDevice = chooseDevice(instanceDriver, instance, context.getTestContext().getCommandLine());
    const uint32_t                                queueFamilyIndex = 0;
    const uint32_t                                queueCount = 1;
    const float                                    queuePriority = 1.0f;
    const std::vector<VkQueueFamilyProperties>    queueFamilyProperties = getPhysicalDeviceQueueFamilyProperties(instanceDriver, physicalDevice);

    const VkDeviceQueueCreateInfo            deviceQueueCreateInfo =
    {
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, //  VkStructureType sType;
        nullptr, //  const void* pNext;
        (VkDeviceQueueCreateFlags)0u, //  VkDeviceQueueCreateFlags flags;
        queueFamilyIndex, //  uint32_t queueFamilyIndex;
        queueCount, //  uint32_t queueCount;
        &queuePriority, //  const float* pQueuePriorities;
    };

    const VkDeviceCreateInfo                deviceCreateInfo =
    {
        VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, //  VkStructureType sType;
        nullptr, //  const void* pNext;
        (VkDeviceCreateFlags)0u, //  VkDeviceCreateFlags flags;
        1u, //  uint32_t queueCreateInfoCount;
        &deviceQueueCreateInfo, //  const VkDeviceQueueCreateInfo* pQueueCreateInfos;
        0u, //  uint32_t enabledLayerCount;
        nullptr, //  const char* const* ppEnabledLayerNames;
        0u, //  uint32_t enabledExtensionCount;
        nullptr, //  const char* const* ppEnabledExtensionNames;
        nullptr, //  const VkPhysicalDeviceFeatures* pEnabledFeatures;
    };
    const Unique<VkDevice>                    device            (createCustomDevice(validationEnabled, platformInterface, instance, instanceDriver, physicalDevice, &deviceCreateInfo));
    const DeviceDriver                        deviceDriver    (platformInterface, instance, device.get(), context.getUsedApiVersion(), context.getTestContext().getCommandLine());

    const std::vector<std::string> functions{
		"vkDestroySurfaceKHR",
		"vkGetPhysicalDeviceSurfaceSupportKHR",
		"vkGetPhysicalDeviceSurfaceCapabilitiesKHR",
		"vkGetPhysicalDeviceSurfaceFormatsKHR",
		"vkGetPhysicalDeviceSurfacePresentModesKHR",
		"vkCreateSwapchainKHR",
		"vkDestroySwapchainKHR",
		"vkGetSwapchainImagesKHR",
		"vkAcquireNextImageKHR",
		"vkQueuePresentKHR",
		"vkGetDeviceGroupPresentCapabilitiesKHR",
		"vkGetDeviceGroupSurfacePresentModesKHR",
		"vkGetPhysicalDevicePresentRectanglesKHR",
		"vkAcquireNextImage2KHR",
		"vkGetPhysicalDeviceDisplayPropertiesKHR",
		"vkGetPhysicalDeviceDisplayPlanePropertiesKHR",
		"vkGetDisplayPlaneSupportedDisplaysKHR",
		"vkGetDisplayModePropertiesKHR",
		"vkCreateDisplayModeKHR",
		"vkGetDisplayPlaneCapabilitiesKHR",
		"vkCreateDisplayPlaneSurfaceKHR",
		"vkCreateSharedSwapchainsKHR",
		"vkCreateXlibSurfaceKHR",
		"vkGetPhysicalDeviceXlibPresentationSupportKHR",
		"vkCreateXcbSurfaceKHR",
		"vkGetPhysicalDeviceXcbPresentationSupportKHR",
		"vkCreateWaylandSurfaceKHR",
		"vkGetPhysicalDeviceWaylandPresentationSupportKHR",
		"vkCreateAndroidSurfaceKHR",
		"vkCreateWin32SurfaceKHR",
		"vkGetPhysicalDeviceWin32PresentationSupportKHR",
		"vkCreateDebugReportCallbackEXT",
		"vkDestroyDebugReportCallbackEXT",
		"vkDebugReportMessageEXT",
		"vkDebugMarkerSetObjectTagEXT",
		"vkDebugMarkerSetObjectNameEXT",
		"vkCmdDebugMarkerBeginEXT",
		"vkCmdDebugMarkerEndEXT",
		"vkCmdDebugMarkerInsertEXT",
		"vkGetPhysicalDeviceVideoCapabilitiesKHR",
		"vkGetPhysicalDeviceVideoFormatPropertiesKHR",
		"vkCreateVideoSessionKHR",
		"vkDestroyVideoSessionKHR",
		"vkGetVideoSessionMemoryRequirementsKHR",
		"vkBindVideoSessionMemoryKHR",
		"vkCreateVideoSessionParametersKHR",
		"vkUpdateVideoSessionParametersKHR",
		"vkDestroyVideoSessionParametersKHR",
		"vkCmdBeginVideoCodingKHR",
		"vkCmdEndVideoCodingKHR",
		"vkCmdControlVideoCodingKHR",
		"vkCmdDecodeVideoKHR",
		"vkCmdBindTransformFeedbackBuffersEXT",
		"vkCmdBeginTransformFeedbackEXT",
		"vkCmdEndTransformFeedbackEXT",
		"vkCmdBeginQueryIndexedEXT",
		"vkCmdEndQueryIndexedEXT",
		"vkCmdDrawIndirectByteCountEXT",
		"vkCreateCuModuleNVX",
		"vkCreateCuFunctionNVX",
		"vkDestroyCuModuleNVX",
		"vkDestroyCuFunctionNVX",
		"vkCmdCuLaunchKernelNVX",
		"vkGetImageViewHandleNVX",
		"vkGetImageViewHandle64NVX",
		"vkGetImageViewAddressNVX",
		"vkCmdDrawIndirectCountAMD",
		"vkCmdDrawIndexedIndirectCountAMD",
		"vkGetShaderInfoAMD",
		"vkCmdBeginRenderingKHR",
		"vkCmdEndRenderingKHR",
		"vkCreateStreamDescriptorSurfaceGGP",
		"vkGetPhysicalDeviceExternalImageFormatPropertiesNV",
		"vkGetMemoryWin32HandleNV",
		"vkGetPhysicalDeviceFeatures2KHR",
		"vkGetPhysicalDeviceProperties2KHR",
		"vkGetPhysicalDeviceFormatProperties2KHR",
		"vkGetPhysicalDeviceImageFormatProperties2KHR",
		"vkGetPhysicalDeviceQueueFamilyProperties2KHR",
		"vkGetPhysicalDeviceMemoryProperties2KHR",
		"vkGetPhysicalDeviceSparseImageFormatProperties2KHR",
		"vkGetDeviceGroupPeerMemoryFeaturesKHR",
		"vkCmdSetDeviceMaskKHR",
		"vkCmdDispatchBaseKHR",
		"vkGetDeviceGroupPresentCapabilitiesKHR",
		"vkGetDeviceGroupSurfacePresentModesKHR",
		"vkGetPhysicalDevicePresentRectanglesKHR",
		"vkAcquireNextImage2KHR",
		"vkCreateViSurfaceNN",
		"vkTrimCommandPoolKHR",
		"vkEnumeratePhysicalDeviceGroupsKHR",
		"vkGetPhysicalDeviceExternalBufferPropertiesKHR",
		"vkGetMemoryWin32HandleKHR",
		"vkGetMemoryWin32HandlePropertiesKHR",
		"vkGetMemoryFdKHR",
		"vkGetMemoryFdPropertiesKHR",
		"vkGetPhysicalDeviceExternalSemaphorePropertiesKHR",
		"vkImportSemaphoreWin32HandleKHR",
		"vkGetSemaphoreWin32HandleKHR",
		"vkImportSemaphoreFdKHR",
		"vkGetSemaphoreFdKHR",
		"vkCmdPushDescriptorSetKHR",
		"vkCmdPushDescriptorSetWithTemplateKHR",
		"vkCmdBeginConditionalRenderingEXT",
		"vkCmdEndConditionalRenderingEXT",
		"vkCreateDescriptorUpdateTemplateKHR",
		"vkDestroyDescriptorUpdateTemplateKHR",
		"vkUpdateDescriptorSetWithTemplateKHR",
		"vkCmdPushDescriptorSetWithTemplateKHR",
		"vkCmdSetViewportWScalingNV",
		"vkReleaseDisplayEXT",
		"vkAcquireXlibDisplayEXT",
		"vkGetRandROutputDisplayEXT",
		"vkGetPhysicalDeviceSurfaceCapabilities2EXT",
		"vkDisplayPowerControlEXT",
		"vkRegisterDeviceEventEXT",
		"vkRegisterDisplayEventEXT",
		"vkGetSwapchainCounterEXT",
		"vkGetRefreshCycleDurationGOOGLE",
		"vkGetPastPresentationTimingGOOGLE",
		"vkCmdSetDiscardRectangleEXT",
		"vkCmdSetDiscardRectangleEnableEXT",
		"vkCmdSetDiscardRectangleModeEXT",
		"vkSetHdrMetadataEXT",
		"vkCreateRenderPass2KHR",
		"vkCmdBeginRenderPass2KHR",
		"vkCmdNextSubpass2KHR",
		"vkCmdEndRenderPass2KHR",
		"vkGetSwapchainStatusKHR",
		"vkGetPhysicalDeviceExternalFencePropertiesKHR",
		"vkImportFenceWin32HandleKHR",
		"vkGetFenceWin32HandleKHR",
		"vkImportFenceFdKHR",
		"vkGetFenceFdKHR",
		"vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR",
		"vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR",
		"vkAcquireProfilingLockKHR",
		"vkReleaseProfilingLockKHR",
		"vkGetPhysicalDeviceSurfaceCapabilities2KHR",
		"vkGetPhysicalDeviceSurfaceFormats2KHR",
		"vkGetPhysicalDeviceDisplayProperties2KHR",
		"vkGetPhysicalDeviceDisplayPlaneProperties2KHR",
		"vkGetDisplayModeProperties2KHR",
		"vkGetDisplayPlaneCapabilities2KHR",
		"vkCreateIOSSurfaceMVK",
		"vkCreateMacOSSurfaceMVK",
		"vkSetDebugUtilsObjectNameEXT",
		"vkSetDebugUtilsObjectTagEXT",
		"vkQueueBeginDebugUtilsLabelEXT",
		"vkQueueEndDebugUtilsLabelEXT",
		"vkQueueInsertDebugUtilsLabelEXT",
		"vkCmdBeginDebugUtilsLabelEXT",
		"vkCmdEndDebugUtilsLabelEXT",
		"vkCmdInsertDebugUtilsLabelEXT",
		"vkCreateDebugUtilsMessengerEXT",
		"vkDestroyDebugUtilsMessengerEXT",
		"vkSubmitDebugUtilsMessageEXT",
		"vkGetAndroidHardwareBufferPropertiesANDROID",
		"vkGetMemoryAndroidHardwareBufferANDROID",
		"vkCreateExecutionGraphPipelinesAMDX",
		"vkGetExecutionGraphPipelineScratchSizeAMDX",
		"vkGetExecutionGraphPipelineNodeIndexAMDX",
		"vkCmdInitializeGraphScratchMemoryAMDX",
		"vkCmdDispatchGraphAMDX",
		"vkCmdDispatchGraphIndirectAMDX",
		"vkCmdDispatchGraphIndirectCountAMDX",
		"vkCmdSetSampleLocationsEXT",
		"vkGetPhysicalDeviceMultisamplePropertiesEXT",
		"vkGetImageMemoryRequirements2KHR",
		"vkGetBufferMemoryRequirements2KHR",
		"vkGetImageSparseMemoryRequirements2KHR",
		"vkCreateAccelerationStructureKHR",
		"vkDestroyAccelerationStructureKHR",
		"vkCmdBuildAccelerationStructuresKHR",
		"vkCmdBuildAccelerationStructuresIndirectKHR",
		"vkBuildAccelerationStructuresKHR",
		"vkCopyAccelerationStructureKHR",
		"vkCopyAccelerationStructureToMemoryKHR",
		"vkCopyMemoryToAccelerationStructureKHR",
		"vkWriteAccelerationStructuresPropertiesKHR",
		"vkCmdCopyAccelerationStructureKHR",
		"vkCmdCopyAccelerationStructureToMemoryKHR",
		"vkCmdCopyMemoryToAccelerationStructureKHR",
		"vkGetAccelerationStructureDeviceAddressKHR",
		"vkCmdWriteAccelerationStructuresPropertiesKHR",
		"vkGetDeviceAccelerationStructureCompatibilityKHR",
		"vkGetAccelerationStructureBuildSizesKHR",
		"vkCmdTraceRaysKHR",
		"vkCreateRayTracingPipelinesKHR",
		"vkGetRayTracingShaderGroupHandlesKHR",
		"vkGetRayTracingCaptureReplayShaderGroupHandlesKHR",
		"vkCmdTraceRaysIndirectKHR",
		"vkGetRayTracingShaderGroupStackSizeKHR",
		"vkCmdSetRayTracingPipelineStackSizeKHR",
		"vkCreateSamplerYcbcrConversionKHR",
		"vkDestroySamplerYcbcrConversionKHR",
		"vkBindBufferMemory2KHR",
		"vkBindImageMemory2KHR",
		"vkGetImageDrmFormatModifierPropertiesEXT",
		"vkCreateValidationCacheEXT",
		"vkDestroyValidationCacheEXT",
		"vkMergeValidationCachesEXT",
		"vkGetValidationCacheDataEXT",
		"vkCmdBindShadingRateImageNV",
		"vkCmdSetViewportShadingRatePaletteNV",
		"vkCmdSetCoarseSampleOrderNV",
		"vkCreateAccelerationStructureNV",
		"vkDestroyAccelerationStructureNV",
		"vkGetAccelerationStructureMemoryRequirementsNV",
		"vkBindAccelerationStructureMemoryNV",
		"vkCmdBuildAccelerationStructureNV",
		"vkCmdCopyAccelerationStructureNV",
		"vkCmdTraceRaysNV",
		"vkCreateRayTracingPipelinesNV",
		"vkGetRayTracingShaderGroupHandlesNV",
		"vkGetAccelerationStructureHandleNV",
		"vkCmdWriteAccelerationStructuresPropertiesNV",
		"vkCompileDeferredNV",
		"vkGetDescriptorSetLayoutSupportKHR",
		"vkCmdDrawIndirectCountKHR",
		"vkCmdDrawIndexedIndirectCountKHR",
		"vkGetMemoryHostPointerPropertiesEXT",
		"vkCmdWriteBufferMarkerAMD",
		"vkCmdWriteBufferMarker2AMD",
		"vkGetPhysicalDeviceCalibrateableTimeDomainsEXT",
		"vkGetCalibratedTimestampsEXT",
		"vkCmdDrawMeshTasksNV",
		"vkCmdDrawMeshTasksIndirectNV",
		"vkCmdDrawMeshTasksIndirectCountNV",
		"vkCmdSetExclusiveScissorEnableNV",
		"vkCmdSetExclusiveScissorNV",
		"vkCmdSetCheckpointNV",
		"vkGetQueueCheckpointDataNV",
		"vkGetQueueCheckpointData2NV",
		"vkGetSemaphoreCounterValueKHR",
		"vkWaitSemaphoresKHR",
		"vkSignalSemaphoreKHR",
		"vkInitializePerformanceApiINTEL",
		"vkUninitializePerformanceApiINTEL",
		"vkCmdSetPerformanceMarkerINTEL",
		"vkCmdSetPerformanceStreamMarkerINTEL",
		"vkCmdSetPerformanceOverrideINTEL",
		"vkAcquirePerformanceConfigurationINTEL",
		"vkReleasePerformanceConfigurationINTEL",
		"vkQueueSetPerformanceConfigurationINTEL",
		"vkGetPerformanceParameterINTEL",
		"vkSetLocalDimmingAMD",
		"vkCreateImagePipeSurfaceFUCHSIA",
		"vkCreateMetalSurfaceEXT",
		"vkGetPhysicalDeviceFragmentShadingRatesKHR",
		"vkCmdSetFragmentShadingRateKHR",
		"vkCmdSetRenderingAttachmentLocationsKHR",
		"vkCmdSetRenderingInputAttachmentIndicesKHR",
		"vkGetBufferDeviceAddressEXT",
		"vkGetPhysicalDeviceToolPropertiesEXT",
		"vkWaitForPresentKHR",
		"vkGetPhysicalDeviceCooperativeMatrixPropertiesNV",
		"vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV",
		"vkGetPhysicalDeviceSurfacePresentModes2EXT",
		"vkAcquireFullScreenExclusiveModeEXT",
		"vkReleaseFullScreenExclusiveModeEXT",
		"vkGetDeviceGroupSurfacePresentModes2EXT",
		"vkCreateHeadlessSurfaceEXT",
		"vkGetBufferDeviceAddressKHR",
		"vkGetBufferOpaqueCaptureAddressKHR",
		"vkGetDeviceMemoryOpaqueCaptureAddressKHR",
		"vkCmdSetLineStippleEXT",
		"vkResetQueryPoolEXT",
		"vkCmdSetCullModeEXT",
		"vkCmdSetFrontFaceEXT",
		"vkCmdSetPrimitiveTopologyEXT",
		"vkCmdSetViewportWithCountEXT",
		"vkCmdSetScissorWithCountEXT",
		"vkCmdBindVertexBuffers2EXT",
		"vkCmdSetDepthTestEnableEXT",
		"vkCmdSetDepthWriteEnableEXT",
		"vkCmdSetDepthCompareOpEXT",
		"vkCmdSetDepthBoundsTestEnableEXT",
		"vkCmdSetStencilTestEnableEXT",
		"vkCmdSetStencilOpEXT",
		"vkCreateDeferredOperationKHR",
		"vkDestroyDeferredOperationKHR",
		"vkGetDeferredOperationMaxConcurrencyKHR",
		"vkGetDeferredOperationResultKHR",
		"vkDeferredOperationJoinKHR",
		"vkGetPipelineExecutablePropertiesKHR",
		"vkGetPipelineExecutableStatisticsKHR",
		"vkGetPipelineExecutableInternalRepresentationsKHR",
		"vkCopyMemoryToImageEXT",
		"vkCopyImageToMemoryEXT",
		"vkCopyImageToImageEXT",
		"vkTransitionImageLayoutEXT",
		"vkGetImageSubresourceLayout2EXT",
		"vkMapMemory2KHR",
		"vkUnmapMemory2KHR",
		"vkReleaseSwapchainImagesEXT",
		"vkGetGeneratedCommandsMemoryRequirementsNV",
		"vkCmdPreprocessGeneratedCommandsNV",
		"vkCmdExecuteGeneratedCommandsNV",
		"vkCmdBindPipelineShaderGroupNV",
		"vkCreateIndirectCommandsLayoutNV",
		"vkDestroyIndirectCommandsLayoutNV",
		"vkCmdSetDepthBias2EXT",
		"vkAcquireDrmDisplayEXT",
		"vkGetDrmDisplayEXT",
		"vkCreatePrivateDataSlotEXT",
		"vkDestroyPrivateDataSlotEXT",
		"vkSetPrivateDataEXT",
		"vkGetPrivateDataEXT",
		"vkGetPhysicalDeviceVideoEncodeQualityLevelPropertiesKHR",
		"vkGetEncodedVideoSessionParametersKHR",
		"vkCmdEncodeVideoKHR",
		"vkCreateCudaModuleNV",
		"vkGetCudaModuleCacheNV",
		"vkCreateCudaFunctionNV",
		"vkDestroyCudaModuleNV",
		"vkDestroyCudaFunctionNV",
		"vkCmdCudaLaunchKernelNV",
		"vkCmdDispatchTileQCOM",
		"vkCmdBeginPerTileExecutionQCOM",
		"vkCmdEndPerTileExecutionQCOM",
		"vkExportMetalObjectsEXT",
		"vkCmdSetEvent2KHR",
		"vkCmdResetEvent2KHR",
		"vkCmdWaitEvents2KHR",
		"vkCmdPipelineBarrier2KHR",
		"vkCmdWriteTimestamp2KHR",
		"vkQueueSubmit2KHR",
		"vkGetDescriptorSetLayoutSizeEXT",
		"vkGetDescriptorSetLayoutBindingOffsetEXT",
		"vkGetDescriptorEXT",
		"vkCmdBindDescriptorBuffersEXT",
		"vkCmdSetDescriptorBufferOffsetsEXT",
		"vkCmdBindDescriptorBufferEmbeddedSamplersEXT",
		"vkGetBufferOpaqueCaptureDescriptorDataEXT",
		"vkGetImageOpaqueCaptureDescriptorDataEXT",
		"vkGetImageViewOpaqueCaptureDescriptorDataEXT",
		"vkGetSamplerOpaqueCaptureDescriptorDataEXT",
		"vkGetAccelerationStructureOpaqueCaptureDescriptorDataEXT",
		"vkCmdSetFragmentShadingRateEnumNV",
		"vkCmdDrawMeshTasksEXT",
		"vkCmdDrawMeshTasksIndirectEXT",
		"vkCmdDrawMeshTasksIndirectCountEXT",
		"vkCmdCopyBuffer2KHR",
		"vkCmdCopyImage2KHR",
		"vkCmdCopyBufferToImage2KHR",
		"vkCmdCopyImageToBuffer2KHR",
		"vkCmdBlitImage2KHR",
		"vkCmdResolveImage2KHR",
		"vkGetImageSubresourceLayout2EXT",
		"vkGetDeviceFaultInfoEXT",
		"vkAcquireWinrtDisplayNV",
		"vkGetWinrtDisplayNV",
		"vkCmdSetVertexInputEXT",
		"vkGetMemoryZirconHandleFUCHSIA",
		"vkGetMemoryZirconHandlePropertiesFUCHSIA",
		"vkImportSemaphoreZirconHandleFUCHSIA",
		"vkGetSemaphoreZirconHandleFUCHSIA",
		"vkCreateBufferCollectionFUCHSIA",
		"vkSetBufferCollectionImageConstraintsFUCHSIA",
		"vkSetBufferCollectionBufferConstraintsFUCHSIA",
		"vkDestroyBufferCollectionFUCHSIA",
		"vkGetBufferCollectionPropertiesFUCHSIA",
		"vkGetDeviceSubpassShadingMaxWorkgroupSizeHUAWEI",
		"vkCmdSubpassShadingHUAWEI",
		"vkCmdBindInvocationMaskHUAWEI",
		"vkGetMemoryRemoteAddressNV",
		"vkGetPipelinePropertiesEXT",
		"vkCmdSetPatchControlPointsEXT",
		"vkCmdSetRasterizerDiscardEnableEXT",
		"vkCmdSetDepthBiasEnableEXT",
		"vkCmdSetLogicOpEXT",
		"vkCmdSetPrimitiveRestartEnableEXT",
		"vkCreateScreenSurfaceQNX",
		"vkGetPhysicalDeviceScreenPresentationSupportQNX",
		"vkCmdSetColorWriteEnableEXT",
		"vkCmdTraceRaysIndirect2KHR",
		"vkCmdDrawMultiEXT",
		"vkCmdDrawMultiIndexedEXT",
		"vkCreateMicromapEXT",
		"vkDestroyMicromapEXT",
		"vkCmdBuildMicromapsEXT",
		"vkBuildMicromapsEXT",
		"vkCopyMicromapEXT",
		"vkCopyMicromapToMemoryEXT",
		"vkCopyMemoryToMicromapEXT",
		"vkWriteMicromapsPropertiesEXT",
		"vkCmdCopyMicromapEXT",
		"vkCmdCopyMicromapToMemoryEXT",
		"vkCmdCopyMemoryToMicromapEXT",
		"vkCmdWriteMicromapsPropertiesEXT",
		"vkGetDeviceMicromapCompatibilityEXT",
		"vkGetMicromapBuildSizesEXT",
		"vkCmdDrawClusterHUAWEI",
		"vkCmdDrawClusterIndirectHUAWEI",
		"vkSetDeviceMemoryPriorityEXT",
		"vkGetDeviceBufferMemoryRequirementsKHR",
		"vkGetDeviceImageMemoryRequirementsKHR",
		"vkGetDeviceImageSparseMemoryRequirementsKHR",
		"vkGetDescriptorSetLayoutHostMappingInfoVALVE",
		"vkGetDescriptorSetHostMappingVALVE",
		"vkCmdCopyMemoryIndirectNV",
		"vkCmdCopyMemoryToImageIndirectNV",
		"vkCmdDecompressMemoryNV",
		"vkCmdDecompressMemoryIndirectCountNV",
		"vkGetPipelineIndirectMemoryRequirementsNV",
		"vkCmdUpdatePipelineIndirectBufferNV",
		"vkGetPipelineIndirectDeviceAddressNV",
		"vkCmdSetDepthClampEnableEXT",
		"vkCmdSetPolygonModeEXT",
		"vkCmdSetRasterizationSamplesEXT",
		"vkCmdSetSampleMaskEXT",
		"vkCmdSetAlphaToCoverageEnableEXT",
		"vkCmdSetAlphaToOneEnableEXT",
		"vkCmdSetLogicOpEnableEXT",
		"vkCmdSetColorBlendEnableEXT",
		"vkCmdSetColorBlendEquationEXT",
		"vkCmdSetColorWriteMaskEXT",
		"vkCmdSetTessellationDomainOriginEXT",
		"vkCmdSetRasterizationStreamEXT",
		"vkCmdSetConservativeRasterizationModeEXT",
		"vkCmdSetExtraPrimitiveOverestimationSizeEXT",
		"vkCmdSetDepthClipEnableEXT",
		"vkCmdSetSampleLocationsEnableEXT",
		"vkCmdSetColorBlendAdvancedEXT",
		"vkCmdSetProvokingVertexModeEXT",
		"vkCmdSetLineRasterizationModeEXT",
		"vkCmdSetLineStippleEnableEXT",
		"vkCmdSetDepthClipNegativeOneToOneEXT",
		"vkCmdSetViewportWScalingEnableNV",
		"vkCmdSetViewportSwizzleNV",
		"vkCmdSetCoverageToColorEnableNV",
		"vkCmdSetCoverageToColorLocationNV",
		"vkCmdSetCoverageModulationModeNV",
		"vkCmdSetCoverageModulationTableEnableNV",
		"vkCmdSetCoverageModulationTableNV",
		"vkCmdSetShadingRateImageEnableNV",
		"vkCmdSetRepresentativeFragmentTestEnableNV",
		"vkCmdSetCoverageReductionModeNV",
		"vkCreateTensorARM",
		"vkDestroyTensorARM",
		"vkCreateTensorViewARM",
		"vkDestroyTensorViewARM",
		"vkGetTensorMemoryRequirementsARM",
		"vkBindTensorMemoryARM",
		"vkGetDeviceTensorMemoryRequirementsARM",
		"vkCmdCopyTensorARM",
		"vkGetPhysicalDeviceExternalTensorPropertiesARM",
		"vkGetTensorOpaqueCaptureDescriptorDataARM",
		"vkGetTensorViewOpaqueCaptureDescriptorDataARM",
		"vkGetShaderModuleIdentifierEXT",
		"vkGetShaderModuleCreateInfoIdentifierEXT",
		"vkGetPhysicalDeviceOpticalFlowImageFormatsNV",
		"vkCreateOpticalFlowSessionNV",
		"vkDestroyOpticalFlowSessionNV",
		"vkBindOpticalFlowSessionImageNV",
		"vkCmdOpticalFlowExecuteNV",
		"vkCmdBindIndexBuffer2KHR",
		"vkGetRenderingAreaGranularityKHR",
		"vkGetDeviceImageSubresourceLayoutKHR",
		"vkGetImageSubresourceLayout2KHR",
		"vkAntiLagUpdateAMD",
		"vkWaitForPresent2KHR",
		"vkCreateShadersEXT",
		"vkDestroyShaderEXT",
		"vkGetShaderBinaryDataEXT",
		"vkCmdBindShadersEXT",
		"vkCmdSetCullModeEXT",
		"vkCmdSetFrontFaceEXT",
		"vkCmdSetPrimitiveTopologyEXT",
		"vkCmdSetViewportWithCountEXT",
		"vkCmdSetScissorWithCountEXT",
		"vkCmdBindVertexBuffers2EXT",
		"vkCmdSetDepthTestEnableEXT",
		"vkCmdSetDepthWriteEnableEXT",
		"vkCmdSetDepthCompareOpEXT",
		"vkCmdSetDepthBoundsTestEnableEXT",
		"vkCmdSetStencilTestEnableEXT",
		"vkCmdSetStencilOpEXT",
		"vkCmdSetVertexInputEXT",
		"vkCmdSetPatchControlPointsEXT",
		"vkCmdSetRasterizerDiscardEnableEXT",
		"vkCmdSetDepthBiasEnableEXT",
		"vkCmdSetLogicOpEXT",
		"vkCmdSetPrimitiveRestartEnableEXT",
		"vkCmdSetTessellationDomainOriginEXT",
		"vkCmdSetDepthClampEnableEXT",
		"vkCmdSetPolygonModeEXT",
		"vkCmdSetRasterizationSamplesEXT",
		"vkCmdSetSampleMaskEXT",
		"vkCmdSetAlphaToCoverageEnableEXT",
		"vkCmdSetAlphaToOneEnableEXT",
		"vkCmdSetLogicOpEnableEXT",
		"vkCmdSetColorBlendEnableEXT",
		"vkCmdSetColorBlendEquationEXT",
		"vkCmdSetColorWriteMaskEXT",
		"vkCmdSetRasterizationStreamEXT",
		"vkCmdSetConservativeRasterizationModeEXT",
		"vkCmdSetExtraPrimitiveOverestimationSizeEXT",
		"vkCmdSetDepthClipEnableEXT",
		"vkCmdSetSampleLocationsEnableEXT",
		"vkCmdSetColorBlendAdvancedEXT",
		"vkCmdSetProvokingVertexModeEXT",
		"vkCmdSetLineRasterizationModeEXT",
		"vkCmdSetLineStippleEnableEXT",
		"vkCmdSetDepthClipNegativeOneToOneEXT",
		"vkCmdSetViewportWScalingEnableNV",
		"vkCmdSetViewportSwizzleNV",
		"vkCmdSetCoverageToColorEnableNV",
		"vkCmdSetCoverageToColorLocationNV",
		"vkCmdSetCoverageModulationModeNV",
		"vkCmdSetCoverageModulationTableEnableNV",
		"vkCmdSetCoverageModulationTableNV",
		"vkCmdSetShadingRateImageEnableNV",
		"vkCmdSetRepresentativeFragmentTestEnableNV",
		"vkCmdSetCoverageReductionModeNV",
		"vkCmdSetDepthClampRangeEXT",
		"vkCreatePipelineBinariesKHR",
		"vkDestroyPipelineBinaryKHR",
		"vkGetPipelineKeyKHR",
		"vkGetPipelineBinaryDataKHR",
		"vkReleaseCapturedPipelineDataKHR",
		"vkGetFramebufferTilePropertiesQCOM",
		"vkGetDynamicRenderingTilePropertiesQCOM",
		"vkReleaseSwapchainImagesKHR",
		"vkGetPhysicalDeviceCooperativeVectorPropertiesNV",
		"vkConvertCooperativeVectorMatrixNV",
		"vkCmdConvertCooperativeVectorMatrixNV",
		"vkSetLatencySleepModeNV",
		"vkLatencySleepNV",
		"vkSetLatencyMarkerNV",
		"vkGetLatencyTimingsNV",
		"vkQueueNotifyOutOfBandNV",
		"vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR",
		"vkCreateDataGraphPipelinesARM",
		"vkCreateDataGraphPipelineSessionARM",
		"vkGetDataGraphPipelineSessionBindPointRequirementsARM",
		"vkGetDataGraphPipelineSessionMemoryRequirementsARM",
		"vkBindDataGraphPipelineSessionMemoryARM",
		"vkDestroyDataGraphPipelineSessionARM",
		"vkCmdDispatchDataGraphARM",
		"vkGetDataGraphPipelineAvailablePropertiesARM",
		"vkGetDataGraphPipelinePropertiesARM",
		"vkGetPhysicalDeviceQueueFamilyDataGraphPropertiesARM",
		"vkGetPhysicalDeviceQueueFamilyDataGraphProcessingEnginePropertiesARM",
		"vkCmdSetAttachmentFeedbackLoopEnableEXT",
		"vkGetScreenBufferPropertiesQNX",
		"vkCmdSetLineStippleKHR",
		"vkGetPhysicalDeviceCalibrateableTimeDomainsKHR",
		"vkGetCalibratedTimestampsKHR",
		"vkCmdBindDescriptorSets2KHR",
		"vkCmdPushConstants2KHR",
		"vkCmdPushDescriptorSet2KHR",
		"vkCmdPushDescriptorSetWithTemplate2KHR",
		"vkCmdSetDescriptorBufferOffsets2EXT",
		"vkCmdBindDescriptorBufferEmbeddedSamplers2EXT",
		"vkCmdBindTileMemoryQCOM",
		"vkCreateExternalComputeQueueNV",
		"vkDestroyExternalComputeQueueNV",
		"vkGetExternalComputeQueueDataNV",
		"vkGetClusterAccelerationStructureBuildSizesNV",
		"vkCmdBuildClusterAccelerationStructureIndirectNV",
		"vkGetPartitionedAccelerationStructuresBuildSizesNV",
		"vkCmdBuildPartitionedAccelerationStructuresNV",
		"vkGetGeneratedCommandsMemoryRequirementsEXT",
		"vkCmdPreprocessGeneratedCommandsEXT",
		"vkCmdExecuteGeneratedCommandsEXT",
		"vkCreateIndirectCommandsLayoutEXT",
		"vkDestroyIndirectCommandsLayoutEXT",
		"vkCreateIndirectExecutionSetEXT",
		"vkDestroyIndirectExecutionSetEXT",
		"vkUpdateIndirectExecutionSetPipelineEXT",
		"vkUpdateIndirectExecutionSetShaderEXT",
		"vkCmdSetDepthClampRangeEXT",
		"vkCreateSurfaceOHOS",
		"vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV",
		"vkGetMemoryMetalHandleEXT",
		"vkGetMemoryMetalHandlePropertiesEXT",
		"vkCmdEndRendering2EXT",
    };

    bool fail = false;
    for (const auto& function : functions)
    {
        if (deviceDriver.getDeviceProcAddr(device.get(), function.c_str()) != nullptr)
        {
            fail = true;
            log << tcu::TestLog::Message << "Function " << function << " is not NULL" << tcu::TestLog::EndMessage;
        }
    }
    if (fail)
        return tcu::TestStatus::fail("Fail");
    return tcu::TestStatus::pass("All functions are NULL");
}

void addGetDeviceProcAddrTests (tcu::TestCaseGroup* testGroup)
{
	addFunctionCase(testGroup, "non_enabled", testGetDeviceProcAddr);
}

}

