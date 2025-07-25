/* WARNING: This is auto-generated file. Do not modify, since changes will
 * be lost! Modify the generating script instead.
 * This file was generated by /scripts/gen_framework.py
 */


void getCoreDeviceExtensionsImpl (uint32_t coreVersion, ::std::vector<const char*>& dst)
{
    if (coreVersion >= VK_API_VERSION_1_2)
    {
        dst.push_back("VK_EXT_descriptor_indexing");
        dst.push_back("VK_EXT_host_query_reset");
        dst.push_back("VK_EXT_sampler_filter_minmax");
        dst.push_back("VK_EXT_scalar_block_layout");
        dst.push_back("VK_EXT_separate_stencil_usage");
        dst.push_back("VK_EXT_shader_viewport_index_layer");
        dst.push_back("VK_KHR_8bit_storage");
        dst.push_back("VK_KHR_buffer_device_address");
        dst.push_back("VK_KHR_create_renderpass2");
        dst.push_back("VK_KHR_depth_stencil_resolve");
        dst.push_back("VK_KHR_draw_indirect_count");
        dst.push_back("VK_KHR_driver_properties");
        dst.push_back("VK_KHR_image_format_list");
        dst.push_back("VK_KHR_imageless_framebuffer");
        dst.push_back("VK_KHR_sampler_mirror_clamp_to_edge");
        dst.push_back("VK_KHR_separate_depth_stencil_layouts");
        dst.push_back("VK_KHR_shader_atomic_int64");
        dst.push_back("VK_KHR_shader_float16_int8");
        dst.push_back("VK_KHR_shader_float_controls");
        dst.push_back("VK_KHR_shader_subgroup_extended_types");
        dst.push_back("VK_KHR_spirv_1_4");
        dst.push_back("VK_KHR_timeline_semaphore");
        dst.push_back("VK_KHR_uniform_buffer_standard_layout");
        dst.push_back("VK_KHR_vulkan_memory_model");
    }
    if (coreVersion >= VK_API_VERSION_1_3)
    {
        dst.push_back("VK_EXT_image_robustness");
        dst.push_back("VK_EXT_inline_uniform_block");
        dst.push_back("VK_EXT_pipeline_creation_cache_control");
        dst.push_back("VK_EXT_pipeline_creation_feedback");
        dst.push_back("VK_EXT_private_data");
        dst.push_back("VK_EXT_shader_demote_to_helper_invocation");
        dst.push_back("VK_EXT_subgroup_size_control");
        dst.push_back("VK_EXT_texture_compression_astc_hdr");
        dst.push_back("VK_EXT_tooling_info");
        dst.push_back("VK_KHR_copy_commands2");
        dst.push_back("VK_KHR_dynamic_rendering");
        dst.push_back("VK_KHR_format_feature_flags2");
        dst.push_back("VK_KHR_maintenance4");
        dst.push_back("VK_KHR_shader_integer_dot_product");
        dst.push_back("VK_KHR_shader_non_semantic_info");
        dst.push_back("VK_KHR_shader_terminate_invocation");
        dst.push_back("VK_KHR_synchronization2");
        dst.push_back("VK_KHR_zero_initialize_workgroup_memory");
    }
    if (coreVersion >= VK_API_VERSION_1_1)
    {
        dst.push_back("VK_KHR_16bit_storage");
        dst.push_back("VK_KHR_bind_memory2");
        dst.push_back("VK_KHR_dedicated_allocation");
        dst.push_back("VK_KHR_descriptor_update_template");
        dst.push_back("VK_KHR_device_group");
        dst.push_back("VK_KHR_external_fence");
        dst.push_back("VK_KHR_external_memory");
        dst.push_back("VK_KHR_external_semaphore");
        dst.push_back("VK_KHR_get_memory_requirements2");
        dst.push_back("VK_KHR_maintenance1");
        dst.push_back("VK_KHR_maintenance2");
        dst.push_back("VK_KHR_maintenance3");
        dst.push_back("VK_KHR_multiview");
        dst.push_back("VK_KHR_relaxed_block_layout");
        dst.push_back("VK_KHR_sampler_ycbcr_conversion");
        dst.push_back("VK_KHR_shader_draw_parameters");
        dst.push_back("VK_KHR_storage_buffer_storage_class");
        dst.push_back("VK_KHR_variable_pointers");
    }
    if (coreVersion >= VK_API_VERSION_1_4)
    {
        dst.push_back("VK_EXT_host_image_copy");
        dst.push_back("VK_EXT_pipeline_protected_access");
        dst.push_back("VK_EXT_pipeline_robustness");
        dst.push_back("VK_KHR_dynamic_rendering_local_read");
        dst.push_back("VK_KHR_global_priority");
        dst.push_back("VK_KHR_index_type_uint8");
        dst.push_back("VK_KHR_line_rasterization");
        dst.push_back("VK_KHR_load_store_op_none");
        dst.push_back("VK_KHR_maintenance5");
        dst.push_back("VK_KHR_maintenance6");
        dst.push_back("VK_KHR_map_memory2");
        dst.push_back("VK_KHR_push_descriptor");
        dst.push_back("VK_KHR_shader_expect_assume");
        dst.push_back("VK_KHR_shader_float_controls2");
        dst.push_back("VK_KHR_shader_subgroup_rotate");
        dst.push_back("VK_KHR_vertex_attribute_divisor");
    }
}

void getCoreInstanceExtensionsImpl (uint32_t coreVersion, ::std::vector<const char*>& dst)
{
    if (coreVersion >= VK_API_VERSION_1_1)
    {
        dst.push_back("VK_KHR_device_group_creation");
        dst.push_back("VK_KHR_external_fence_capabilities");
        dst.push_back("VK_KHR_external_memory_capabilities");
        dst.push_back("VK_KHR_external_semaphore_capabilities");
        dst.push_back("VK_KHR_get_physical_device_properties2");
    }
}

