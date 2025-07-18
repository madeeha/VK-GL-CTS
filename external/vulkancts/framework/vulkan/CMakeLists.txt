# vk - Vulkan utilites
# Split into two libraries - one that depends on spirv/glslang libraries
# which have long build times, and one that can build in parallel with those.

include(CheckIncludeFileCXX)
check_include_file_cxx(
	${CMAKE_SOURCE_DIR}/external/renderdoc/src/renderdoc_app.h
	HAVE_RENDERDOC_APP_H)
if(HAVE_RENDERDOC_APP_H)
	set(VKRENDERDOC_SRC vkRenderDocUtil.cpp)
else()
	set(VKRENDERDOC_SRC vkNoRenderDocUtil.cpp)
endif(HAVE_RENDERDOC_APP_H)


# Build inl files
set(DEQP_VULKAN_INL_GEN_INPUT ${PROJECT_SOURCE_DIR}/external/vulkan-docs/src/xml/vk.xml)
set(DEQP_VULKAN_INL_GEN_OUTPUTS_DIR ${PROJECT_SOURCE_DIR}/external/vulkancts/framework/vulkan)
set(DEQP_VULKAN_INL_GEN_OUTPUTS
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkApiExtensionDependencyInfo.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkBasicTypes.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkConcreteDeviceInterface.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkConcreteInstanceInterface.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkConcretePlatformInterface.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkCoreFunctionalities.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkDeviceDriverImpl.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkDeviceExtensions.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkDeviceFeatures.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkDeviceFeatures2.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkDeviceFeaturesForContextDecl.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkDeviceFeaturesForContextDefs.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkDeviceFeaturesForDefaultDeviceDefs.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkDeviceFunctionPointers.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkDeviceProperties.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkDevicePropertiesForContextDecl.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkDevicePropertiesForContextDefs.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkDevicePropertiesForDefaultDeviceDefs.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkExtensionFunctions.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkFunctionPointerTypes.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkGetDeviceProcAddr.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkGetStructureTypeImpl.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkHandleType.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkInitDeviceFunctionPointers.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkInitInstanceFunctionPointers.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkInitPlatformFunctionPointers.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkInstanceDriverImpl.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkInstanceExtensions.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkInstanceFunctionPointers.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkKnownDriverIds.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkMandatoryFeatures.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkNullDriverImpl.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkObjTypeImpl.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkPlatformDriverImpl.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkPlatformFunctionPointers.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkRefUtil.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkRefUtilImpl.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkStrUtil.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkStrUtilImpl.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkStructTypes.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkSupportedExtensions.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkTypeUtil.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkVirtualDeviceInterface.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkVirtualInstanceInterface.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkVirtualPlatformInterface.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkEntryPointValidation.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkProfileTests.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkKnownConformanceVersions.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/createFeatureStructFromEnum.inl
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkDeviceFeaturesVariantDecl.inl
	# NOTE: when new file is generated by gen_framework it should be added to this list.
	${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}/vkVulkan_c.inl)

# If .inl files are generated to build directory add custom command that will build them.
if (DEQP_VULKAN_INL_GEN_OUTPUTS_DIR MATCHES ^${PROJECT_BINARY_DIR})
	add_custom_command(
		# Remove all outputs to help detect missing outputs.
		COMMAND cmake -E remove ${DEQP_VULKAN_INL_GEN_OUTPUTS}

		# Generate all inl files.
		COMMAND ${PYTHON_EXECUTABLE} ARGS ${PROJECT_SOURCE_DIR}/external/vulkancts/scripts/gen_framework.py -o ${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}
		COMMAND ${PYTHON_EXECUTABLE} ARGS ${PROJECT_SOURCE_DIR}/external/vulkancts/scripts/gen_framework_c.py -o ${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR}

		# Check all outputs exist, as CMake does not do this.
		COMMAND cmake -E md5sum ${DEQP_VULKAN_INL_GEN_OUTPUTS}

		DEPENDS ${DEQP_VULKAN_INL_GEN_INPUT}
		OUTPUT ${DEQP_VULKAN_INL_GEN_OUTPUTS})

	add_custom_target(
		deqp-vk-inl ALL
		DEPENDS ${DEQP_VULKAN_INL_GEN_OUTPUTS})
else()
	add_custom_target(deqp-vk-inl)
endif()

set(VKUTILNOSHADER_SRCS
	vkApiVersion.cpp
	vkApiVersion.hpp
	vkAppParamsUtil.cpp
	vkAppParamsUtil.hpp
	vkBuilderUtil.cpp
	vkBuilderUtil.hpp
	vkBarrierUtil.cpp
	vkBarrierUtil.hpp
	vkCmdUtil.cpp
	vkCmdUtil.hpp
	vkDefs.cpp
	vkDefs.hpp
	vkRef.cpp
	vkRef.hpp
	vkRefUtil.cpp
	vkRefUtil.hpp
	vkPlatform.cpp
	vkPlatform.hpp
	vkStrUtil.cpp
	vkStrUtil.hpp
	vkQueryUtil.cpp
	vkQueryUtil.hpp
	vkMemUtil.cpp
	vkMemUtil.hpp
	vkDeviceUtil.cpp
	vkDeviceUtil.hpp
	vkBinaryRegistry.cpp
	vkBinaryRegistry.hpp
	vkNullDriver.cpp
	vkNullDriver.hpp
	vkImageUtil.cpp
	vkImageUtil.hpp
	vkTypeUtil.cpp
	vkTypeUtil.hpp
	vkAllocationCallbackUtil.cpp
	vkAllocationCallbackUtil.hpp
	vkWsiPlatform.cpp
	vkWsiPlatform.hpp
	vkWsiUtil.cpp
	vkWsiUtil.hpp
	vkDebugReportUtil.cpp
	vkDebugReportUtil.hpp
	vkBufferWithMemory.cpp
	vkBufferWithMemory.hpp
	vkImageWithMemory.cpp
	vkImageWithMemory.hpp
	vkImageWithMemory.cpp
	vkImageWithMemory.hpp
	vkShaderProgram.cpp
	vkShaderProgram.hpp
	vkValidatorOptions.hpp
	vkYCbCrImageWithMemory.cpp
	vkYCbCrImageWithMemory.hpp
	vkObjUtil.cpp
	vkObjUtil.hpp
	${VKRENDERDOC_SRC}
	vkRenderDocUtil.hpp
	vkDeviceFeatures.hpp
	vkDeviceFeatures.cpp
	vkDeviceProperties.hpp
	vkDeviceProperties.cpp
	vkRayTracingUtil.hpp
	vkRayTracingUtil.cpp
	vkPipelineBinaryUtil.hpp
	vkPipelineBinaryUtil.cpp
	vkPipelineConstructionUtil.hpp
	vkPipelineConstructionUtil.cpp
	vkSafetyCriticalUtil.hpp
	vkSafetyCriticalUtil.cpp
	vkResourceInterface.hpp
	vkResourceInterface.cpp
	vkMd5Sum.hpp
	vkMd5Sum.cpp
	vkComputePipelineConstructionUtil.hpp
	vkComputePipelineConstructionUtil.cpp
	vkShaderObjectUtil.hpp
	vkShaderObjectUtil.cpp
	)

set(VKUTILNOSHADER_INLS
	generated/vulkan/vkDeviceFeatures.inl
	generated/vulkan/vkDeviceFeatureTest.inl
	generated/vulkan/vkDeviceProperties.inl
	)

set(VKSCUTILNOSHADER_INLS
	generated/vulkansc/vkDeviceFeatures.inl
	generated/vulkansc/vkDeviceFeatureTest.inl
	generated/vulkansc/vkDeviceProperties.inl
	generated/vulkansc/vulkan_json_data.hpp
	generated/vulkansc/vulkan_json_parser.hpp
	)

set(VKUTIL_SRCS
	vkPrograms.cpp
	vkPrograms.hpp
	vkShaderToSpirV.cpp
	vkShaderToSpirV.hpp
	vkSpirVAsm.hpp
	vkSpirVAsm.cpp
	vkSpirVProgram.hpp
	vkSpirVProgram.cpp
	)

set(VKUTILNOSHADER_LIBS
	glutil
	tcutil
	)

include_directories(${GLSLANG_INCLUDE_PATH})
include_directories(${DEQP_VULKAN_INL_GEN_OUTPUTS_DIR})

# \note Code interfacing with glslang needs to include third-party headers
#       that cause all sorts of warnings to appear.
if (DE_COMPILER_IS_GCC OR DE_COMPILER_IS_CLANG)
	set_source_files_properties(
		FILES vkShaderToSpirV.cpp
		PROPERTIES COMPILE_FLAGS "${DE_3RD_PARTY_CXX_FLAGS}")
endif ()


# \note Code interfacing with spirv-opt needs to include third-party headers
#       that cause all sorts of warnings to appear.
if (DE_COMPILER_IS_GCC OR DE_COMPILER_IS_CLANG)
	set_source_files_properties(
		FILES vkPrograms.cpp
		PROPERTIES COMPILE_FLAGS "${DE_3RD_PARTY_CXX_FLAGS}")
endif ()

if(DEQP_HAVE_RENDERDOC_HEADER)
	add_definitions(-DDEQP_HAVE_RENDERDOC_HEADER=1)
endif()

add_definitions(-DDISABLE_SHADERCACHE_IPC)

PCH(VKUTILNOSHADER_SRCS ../../modules/vulkan/pch.cpp)
PCH(VKUTIL_SRCS ../../modules/vulkan/pch.cpp)
PCH(VKUTILNOSHADER_INLS ../../modules/vulkan/pch.cpp)
PCH(VKSCUTILNOSHADER_INLS ../../modules/vulkan/pch.cpp)

add_library(vkutilnoshader STATIC ${VKUTILNOSHADER_SRCS} ${VKUTILNOSHADER_INLS})
target_compile_definitions(vkutilnoshader PUBLIC CTS_USES_VULKAN)
target_include_directories(vkutilnoshader PUBLIC generated/vulkan)
target_link_libraries(vkutilnoshader PUBLIC ${VKUTILNOSHADER_LIBS})
add_dependencies(vkutilnoshader deqp-vk-inl)

add_library(vkscutilnoshader STATIC ${VKUTILNOSHADER_SRCS} ${VKSCUTILNOSHADER_INLS})
target_compile_definitions(vkscutilnoshader PUBLIC CTS_USES_VULKANSC)
target_include_directories(vkscutilnoshader BEFORE PUBLIC generated/vulkansc)
target_include_directories(vkscutilnoshader BEFORE PUBLIC ${PROJECT_SOURCE_DIR}/external/vulkancts/vkscserver)
target_link_libraries(vkscutilnoshader PUBLIC ${VKUTILNOSHADER_LIBS} jsoncpp_static vksc-server-util)

if (DEQP_USE_X11)
	find_package(X11 REQUIRED)

	target_compile_definitions(vkutilnoshader PUBLIC DEQP_SUPPORT_X11=1)
	target_link_libraries(vkutilnoshader PUBLIC ${X11_LIBRARIES})

	target_compile_definitions(vkscutilnoshader PUBLIC DEQP_SUPPORT_X11=1)
	target_link_libraries(vkscutilnoshader PUBLIC ${X11_LIBRARIES})

	if (DEQP_USE_XCB)
		find_package(XCB REQUIRED)

		target_compile_definitions(vkutilnoshader PUBLIC DEQP_SUPPORT_XCB=1)
		target_link_libraries(vkutilnoshader PUBLIC ${XCB_LIBRARIES})

		target_compile_definitions(vkscutilnoshader PUBLIC DEQP_SUPPORT_XCB=1)
		target_link_libraries(vkscutilnoshader PUBLIC ${XCB_LIBRARIES})

	endif ()
endif()

add_library(vkutil STATIC ${VKUTIL_SRCS})
target_include_directories(vkutil PUBLIC ${GLSLANG_INCLUDE_PATH} ${spirv-tools_SOURCE_DIR}/include ${spirv-tools_SOURCE_DIR}/external/include )
target_link_libraries(vkutil PUBLIC vkutilnoshader ${GLSLANG_LIBRARIES} SPIRV-Tools SPIRV-Tools-opt)

add_library(vkscutil STATIC ${VKUTIL_SRCS})
target_include_directories(vkscutil PUBLIC ${GLSLANG_INCLUDE_PATH} ${spirv-tools_SOURCE_DIR}/include ${spirv-tools_SOURCE_DIR}/external/include )
target_link_libraries(vkscutil PUBLIC vkscutilnoshader ${GLSLANG_LIBRARIES} SPIRV-Tools SPIRV-Tools-opt)
