/* WARNING: This is auto-generated file. Do not modify, since changes will
 * be lost! Modify the generating script instead.
 * This file was generated by /scripts/gen_framework.py
 */

virtual void		destroyInstance														(VkInstance instance, const VkAllocationCallbacks* pAllocator) const = 0;
virtual VkResult	enumeratePhysicalDevices											(VkInstance instance, uint32_t* pPhysicalDeviceCount, VkPhysicalDevice* pPhysicalDevices) const = 0;
virtual void		getPhysicalDeviceProperties											(VkPhysicalDevice physicalDevice, VkPhysicalDeviceProperties* pProperties) const = 0;
virtual void		getPhysicalDeviceQueueFamilyProperties								(VkPhysicalDevice physicalDevice, uint32_t* pQueueFamilyPropertyCount, VkQueueFamilyProperties* pQueueFamilyProperties) const = 0;
virtual void		getPhysicalDeviceMemoryProperties									(VkPhysicalDevice physicalDevice, VkPhysicalDeviceMemoryProperties* pMemoryProperties) const = 0;
virtual void		getPhysicalDeviceFeatures											(VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures* pFeatures) const = 0;
virtual void		getPhysicalDeviceFormatProperties									(VkPhysicalDevice physicalDevice, VkFormat format, VkFormatProperties* pFormatProperties) const = 0;
virtual VkResult	getPhysicalDeviceImageFormatProperties								(VkPhysicalDevice physicalDevice, VkFormat format, VkImageType type, VkImageTiling tiling, VkImageUsageFlags usage, VkImageCreateFlags flags, VkImageFormatProperties* pImageFormatProperties) const = 0;
virtual VkResult	createDevice														(VkPhysicalDevice physicalDevice, const VkDeviceCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDevice* pDevice) const = 0;
virtual VkResult	enumerateDeviceLayerProperties										(VkPhysicalDevice physicalDevice, uint32_t* pPropertyCount, VkLayerProperties* pProperties) const = 0;
virtual VkResult	enumerateDeviceExtensionProperties									(VkPhysicalDevice physicalDevice, const char* pLayerName, uint32_t* pPropertyCount, VkExtensionProperties* pProperties) const = 0;
virtual void		getPhysicalDeviceSparseImageFormatProperties						(VkPhysicalDevice physicalDevice, VkFormat format, VkImageType type, VkSampleCountFlagBits samples, VkImageUsageFlags usage, VkImageTiling tiling, uint32_t* pPropertyCount, VkSparseImageFormatProperties* pProperties) const = 0;
virtual VkResult	createAndroidSurfaceKHR												(VkInstance instance, const VkAndroidSurfaceCreateInfoKHR* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface) const = 0;
virtual VkResult	createSurfaceOHOS													(VkInstance instance, const VkSurfaceCreateInfoOHOS* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface) const = 0;
virtual VkResult	getPhysicalDeviceDisplayPropertiesKHR								(VkPhysicalDevice physicalDevice, uint32_t* pPropertyCount, VkDisplayPropertiesKHR* pProperties) const = 0;
virtual VkResult	getPhysicalDeviceDisplayPlanePropertiesKHR							(VkPhysicalDevice physicalDevice, uint32_t* pPropertyCount, VkDisplayPlanePropertiesKHR* pProperties) const = 0;
virtual VkResult	getDisplayPlaneSupportedDisplaysKHR									(VkPhysicalDevice physicalDevice, uint32_t planeIndex, uint32_t* pDisplayCount, VkDisplayKHR* pDisplays) const = 0;
virtual VkResult	getDisplayModePropertiesKHR											(VkPhysicalDevice physicalDevice, VkDisplayKHR display, uint32_t* pPropertyCount, VkDisplayModePropertiesKHR* pProperties) const = 0;
virtual VkResult	createDisplayModeKHR												(VkPhysicalDevice physicalDevice, VkDisplayKHR display, const VkDisplayModeCreateInfoKHR* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDisplayModeKHR* pMode) const = 0;
virtual VkResult	getDisplayPlaneCapabilitiesKHR										(VkPhysicalDevice physicalDevice, VkDisplayModeKHR mode, uint32_t planeIndex, VkDisplayPlaneCapabilitiesKHR* pCapabilities) const = 0;
virtual VkResult	createDisplayPlaneSurfaceKHR										(VkInstance instance, const VkDisplaySurfaceCreateInfoKHR* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface) const = 0;
virtual void		destroySurfaceKHR													(VkInstance instance, VkSurfaceKHR surface, const VkAllocationCallbacks* pAllocator) const = 0;
virtual VkResult	getPhysicalDeviceSurfaceSupportKHR									(VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex, VkSurfaceKHR surface, VkBool32* pSupported) const = 0;
virtual VkResult	getPhysicalDeviceSurfaceCapabilitiesKHR								(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, VkSurfaceCapabilitiesKHR* pSurfaceCapabilities) const = 0;
virtual VkResult	getPhysicalDeviceSurfaceFormatsKHR									(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, uint32_t* pSurfaceFormatCount, VkSurfaceFormatKHR* pSurfaceFormats) const = 0;
virtual VkResult	getPhysicalDeviceSurfacePresentModesKHR								(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, uint32_t* pPresentModeCount, VkPresentModeKHR* pPresentModes) const = 0;
virtual VkResult	createViSurfaceNN													(VkInstance instance, const VkViSurfaceCreateInfoNN* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface) const = 0;
virtual VkResult	createWaylandSurfaceKHR												(VkInstance instance, const VkWaylandSurfaceCreateInfoKHR* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface) const = 0;
virtual VkBool32	getPhysicalDeviceWaylandPresentationSupportKHR						(VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex, pt::WaylandDisplayPtr display) const = 0;
virtual VkResult	createWin32SurfaceKHR												(VkInstance instance, const VkWin32SurfaceCreateInfoKHR* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface) const = 0;
virtual VkBool32	getPhysicalDeviceWin32PresentationSupportKHR						(VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex) const = 0;
virtual VkResult	createXlibSurfaceKHR												(VkInstance instance, const VkXlibSurfaceCreateInfoKHR* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface) const = 0;
virtual VkBool32	getPhysicalDeviceXlibPresentationSupportKHR							(VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex, pt::XlibDisplayPtr dpy, pt::XlibVisualID visualID) const = 0;
virtual VkResult	createXcbSurfaceKHR													(VkInstance instance, const VkXcbSurfaceCreateInfoKHR* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface) const = 0;
virtual VkBool32	getPhysicalDeviceXcbPresentationSupportKHR							(VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex, pt::XcbConnectionPtr connection, pt::XcbVisualid visual_id) const = 0;
virtual VkResult	createImagePipeSurfaceFUCHSIA										(VkInstance instance, const VkImagePipeSurfaceCreateInfoFUCHSIA* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface) const = 0;
virtual VkResult	createStreamDescriptorSurfaceGGP									(VkInstance instance, const VkStreamDescriptorSurfaceCreateInfoGGP* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface) const = 0;
virtual VkResult	createScreenSurfaceQNX												(VkInstance instance, const VkScreenSurfaceCreateInfoQNX* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface) const = 0;
virtual VkBool32	getPhysicalDeviceScreenPresentationSupportQNX						(VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex, pt::QNXScreenWindowPtr window) const = 0;
virtual VkResult	createDebugReportCallbackEXT										(VkInstance instance, const VkDebugReportCallbackCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugReportCallbackEXT* pCallback) const = 0;
virtual void		destroyDebugReportCallbackEXT										(VkInstance instance, VkDebugReportCallbackEXT callback, const VkAllocationCallbacks* pAllocator) const = 0;
virtual void		debugReportMessageEXT												(VkInstance instance, VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objectType, uint64_t object, size_t location, int32_t messageCode, const char* pLayerPrefix, const char* pMessage) const = 0;
virtual VkResult	getPhysicalDeviceExternalImageFormatPropertiesNV					(VkPhysicalDevice physicalDevice, VkFormat format, VkImageType type, VkImageTiling tiling, VkImageUsageFlags usage, VkImageCreateFlags flags, VkExternalMemoryHandleTypeFlagsNV externalHandleType, VkExternalImageFormatPropertiesNV* pExternalImageFormatProperties) const = 0;
virtual void		getPhysicalDeviceFeatures2											(VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures2* pFeatures) const = 0;
virtual void		getPhysicalDeviceProperties2										(VkPhysicalDevice physicalDevice, VkPhysicalDeviceProperties2* pProperties) const = 0;
virtual void		getPhysicalDeviceFormatProperties2									(VkPhysicalDevice physicalDevice, VkFormat format, VkFormatProperties2* pFormatProperties) const = 0;
virtual VkResult	getPhysicalDeviceImageFormatProperties2								(VkPhysicalDevice physicalDevice, const VkPhysicalDeviceImageFormatInfo2* pImageFormatInfo, VkImageFormatProperties2* pImageFormatProperties) const = 0;
virtual void		getPhysicalDeviceQueueFamilyProperties2								(VkPhysicalDevice physicalDevice, uint32_t* pQueueFamilyPropertyCount, VkQueueFamilyProperties2* pQueueFamilyProperties) const = 0;
virtual void		getPhysicalDeviceMemoryProperties2									(VkPhysicalDevice physicalDevice, VkPhysicalDeviceMemoryProperties2* pMemoryProperties) const = 0;
virtual void		getPhysicalDeviceSparseImageFormatProperties2						(VkPhysicalDevice physicalDevice, const VkPhysicalDeviceSparseImageFormatInfo2* pFormatInfo, uint32_t* pPropertyCount, VkSparseImageFormatProperties2* pProperties) const = 0;
virtual void		getPhysicalDeviceExternalBufferProperties							(VkPhysicalDevice physicalDevice, const VkPhysicalDeviceExternalBufferInfo* pExternalBufferInfo, VkExternalBufferProperties* pExternalBufferProperties) const = 0;
virtual void		getPhysicalDeviceExternalSemaphoreProperties						(VkPhysicalDevice physicalDevice, const VkPhysicalDeviceExternalSemaphoreInfo* pExternalSemaphoreInfo, VkExternalSemaphoreProperties* pExternalSemaphoreProperties) const = 0;
virtual void		getPhysicalDeviceExternalFenceProperties							(VkPhysicalDevice physicalDevice, const VkPhysicalDeviceExternalFenceInfo* pExternalFenceInfo, VkExternalFenceProperties* pExternalFenceProperties) const = 0;
virtual VkResult	releaseDisplayEXT													(VkPhysicalDevice physicalDevice, VkDisplayKHR display) const = 0;
virtual VkResult	acquireXlibDisplayEXT												(VkPhysicalDevice physicalDevice, pt::XlibDisplayPtr dpy, VkDisplayKHR display) const = 0;
virtual VkResult	getRandROutputDisplayEXT											(VkPhysicalDevice physicalDevice, pt::XlibDisplayPtr dpy, pt::RROutput rrOutput, VkDisplayKHR* pDisplay) const = 0;
virtual VkResult	acquireWinrtDisplayNV												(VkPhysicalDevice physicalDevice, VkDisplayKHR display) const = 0;
virtual VkResult	getWinrtDisplayNV													(VkPhysicalDevice physicalDevice, uint32_t deviceRelativeId, VkDisplayKHR* pDisplay) const = 0;
virtual VkResult	getPhysicalDeviceSurfaceCapabilities2EXT							(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, VkSurfaceCapabilities2EXT* pSurfaceCapabilities) const = 0;
virtual VkResult	enumeratePhysicalDeviceGroups										(VkInstance instance, uint32_t* pPhysicalDeviceGroupCount, VkPhysicalDeviceGroupProperties* pPhysicalDeviceGroupProperties) const = 0;
virtual VkResult	getPhysicalDevicePresentRectanglesKHR								(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, uint32_t* pRectCount, VkRect2D* pRects) const = 0;
virtual VkResult	createIOSSurfaceMVK													(VkInstance instance, const VkIOSSurfaceCreateInfoMVK* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface) const = 0;
virtual VkResult	createMacOSSurfaceMVK												(VkInstance instance, const VkMacOSSurfaceCreateInfoMVK* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface) const = 0;
virtual VkResult	createMetalSurfaceEXT												(VkInstance instance, const VkMetalSurfaceCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface) const = 0;
virtual void		getPhysicalDeviceMultisamplePropertiesEXT							(VkPhysicalDevice physicalDevice, VkSampleCountFlagBits samples, VkMultisamplePropertiesEXT* pMultisampleProperties) const = 0;
virtual VkResult	getPhysicalDeviceSurfaceCapabilities2KHR							(VkPhysicalDevice physicalDevice, const VkPhysicalDeviceSurfaceInfo2KHR* pSurfaceInfo, VkSurfaceCapabilities2KHR* pSurfaceCapabilities) const = 0;
virtual VkResult	getPhysicalDeviceSurfaceFormats2KHR									(VkPhysicalDevice physicalDevice, const VkPhysicalDeviceSurfaceInfo2KHR* pSurfaceInfo, uint32_t* pSurfaceFormatCount, VkSurfaceFormat2KHR* pSurfaceFormats) const = 0;
virtual VkResult	getPhysicalDeviceDisplayProperties2KHR								(VkPhysicalDevice physicalDevice, uint32_t* pPropertyCount, VkDisplayProperties2KHR* pProperties) const = 0;
virtual VkResult	getPhysicalDeviceDisplayPlaneProperties2KHR							(VkPhysicalDevice physicalDevice, uint32_t* pPropertyCount, VkDisplayPlaneProperties2KHR* pProperties) const = 0;
virtual VkResult	getDisplayModeProperties2KHR										(VkPhysicalDevice physicalDevice, VkDisplayKHR display, uint32_t* pPropertyCount, VkDisplayModeProperties2KHR* pProperties) const = 0;
virtual VkResult	getDisplayPlaneCapabilities2KHR										(VkPhysicalDevice physicalDevice, const VkDisplayPlaneInfo2KHR* pDisplayPlaneInfo, VkDisplayPlaneCapabilities2KHR* pCapabilities) const = 0;
virtual VkResult	getPhysicalDeviceCalibrateableTimeDomainsKHR						(VkPhysicalDevice physicalDevice, uint32_t* pTimeDomainCount, VkTimeDomainKHR* pTimeDomains) const = 0;
virtual VkResult	createDebugUtilsMessengerEXT										(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pMessenger) const = 0;
virtual void		destroyDebugUtilsMessengerEXT										(VkInstance instance, VkDebugUtilsMessengerEXT messenger, const VkAllocationCallbacks* pAllocator) const = 0;
virtual void		submitDebugUtilsMessageEXT											(VkInstance instance, VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageTypes, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData) const = 0;
virtual VkResult	getPhysicalDeviceCooperativeMatrixPropertiesNV						(VkPhysicalDevice physicalDevice, uint32_t* pPropertyCount, VkCooperativeMatrixPropertiesNV* pProperties) const = 0;
virtual VkResult	getPhysicalDeviceSurfacePresentModes2EXT							(VkPhysicalDevice physicalDevice, const VkPhysicalDeviceSurfaceInfo2KHR* pSurfaceInfo, uint32_t* pPresentModeCount, VkPresentModeKHR* pPresentModes) const = 0;
virtual VkResult	enumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR		(VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex, uint32_t* pCounterCount, VkPerformanceCounterKHR* pCounters, VkPerformanceCounterDescriptionKHR* pCounterDescriptions) const = 0;
virtual void		getPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR				(VkPhysicalDevice physicalDevice, const VkQueryPoolPerformanceCreateInfoKHR* pPerformanceQueryCreateInfo, uint32_t* pNumPasses) const = 0;
virtual VkResult	createHeadlessSurfaceEXT											(VkInstance instance, const VkHeadlessSurfaceCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface) const = 0;
virtual VkResult	getPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV		(VkPhysicalDevice physicalDevice, uint32_t* pCombinationCount, VkFramebufferMixedSamplesCombinationNV* pCombinations) const = 0;
virtual VkResult	getPhysicalDeviceToolProperties										(VkPhysicalDevice physicalDevice, uint32_t* pToolCount, VkPhysicalDeviceToolProperties* pToolProperties) const = 0;
virtual VkResult	getPhysicalDeviceFragmentShadingRatesKHR							(VkPhysicalDevice physicalDevice, uint32_t* pFragmentShadingRateCount, VkPhysicalDeviceFragmentShadingRateKHR* pFragmentShadingRates) const = 0;
virtual VkResult	getPhysicalDeviceVideoCapabilitiesKHR								(VkPhysicalDevice physicalDevice, const VkVideoProfileInfoKHR* pVideoProfile, VkVideoCapabilitiesKHR* pCapabilities) const = 0;
virtual VkResult	getPhysicalDeviceVideoFormatPropertiesKHR							(VkPhysicalDevice physicalDevice, const VkPhysicalDeviceVideoFormatInfoKHR* pVideoFormatInfo, uint32_t* pVideoFormatPropertyCount, VkVideoFormatPropertiesKHR* pVideoFormatProperties) const = 0;
virtual VkResult	getPhysicalDeviceVideoEncodeQualityLevelPropertiesKHR				(VkPhysicalDevice physicalDevice, const VkPhysicalDeviceVideoEncodeQualityLevelInfoKHR* pQualityLevelInfo, VkVideoEncodeQualityLevelPropertiesKHR* pQualityLevelProperties) const = 0;
virtual VkResult	acquireDrmDisplayEXT												(VkPhysicalDevice physicalDevice, int32_t drmFd, VkDisplayKHR display) const = 0;
virtual VkResult	getDrmDisplayEXT													(VkPhysicalDevice physicalDevice, int32_t drmFd, uint32_t connectorId, VkDisplayKHR* display) const = 0;
virtual VkResult	getPhysicalDeviceOpticalFlowImageFormatsNV							(VkPhysicalDevice physicalDevice, const VkOpticalFlowImageFormatInfoNV* pOpticalFlowImageFormatInfo, uint32_t* pFormatCount, VkOpticalFlowImageFormatPropertiesNV* pImageFormatProperties) const = 0;
virtual VkResult	getPhysicalDeviceCooperativeMatrixPropertiesKHR						(VkPhysicalDevice physicalDevice, uint32_t* pPropertyCount, VkCooperativeMatrixPropertiesKHR* pProperties) const = 0;
virtual VkResult	getPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV	(VkPhysicalDevice physicalDevice, uint32_t* pPropertyCount, VkCooperativeMatrixFlexibleDimensionsPropertiesNV* pProperties) const = 0;
virtual VkResult	getPhysicalDeviceCooperativeVectorPropertiesNV						(VkPhysicalDevice physicalDevice, uint32_t* pPropertyCount, VkCooperativeVectorPropertiesNV* pProperties) const = 0;
virtual void		getPhysicalDeviceExternalTensorPropertiesARM						(VkPhysicalDevice physicalDevice, const VkPhysicalDeviceExternalTensorInfoARM* pExternalTensorInfo, VkExternalTensorPropertiesARM* pExternalTensorProperties) const = 0;
virtual VkResult	getPhysicalDeviceQueueFamilyDataGraphPropertiesARM					(VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex, uint32_t* pQueueFamilyDataGraphPropertyCount, VkQueueFamilyDataGraphPropertiesARM* pQueueFamilyDataGraphProperties) const = 0;
virtual void		getPhysicalDeviceQueueFamilyDataGraphProcessingEnginePropertiesARM	(VkPhysicalDevice physicalDevice, const VkPhysicalDeviceQueueFamilyDataGraphProcessingEngineInfoARM* pQueueFamilyDataGraphProcessingEngineInfo, VkQueueFamilyDataGraphProcessingEnginePropertiesARM* pQueueFamilyDataGraphProcessingEngineProperties) const = 0;
