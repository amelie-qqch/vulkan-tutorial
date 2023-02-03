use std::collections::HashSet;
use std::ffi::{c_void, CStr};
use std::mem::size_of;
use std::time::Instant;

use std::ptr::copy_nonoverlapping as memcpy;
use anyhow::{anyhow, Error};
use log::{debug, error, trace, warn};
use nalgebra_glm as glm;
use vulkanalia::{Device, Entry, Instance, vk};
use vulkanalia::loader::{LibloadingLoader, LIBRARY};
use vulkanalia::vk::{DeviceV1_0, EntryV1_0, ExtDebugUtilsExtension, Handle, HasBuilder, InstanceV1_0, KhrSurfaceExtension, KhrSwapchainExtension};
use vulkanalia::window as vk_window;
use winit::window::Window;

use crate::{AppData, create_framebuffers, create_sync_objects, Error};
use crate::commandbuffer::{create_command_buffers, create_command_pools};
use crate::depthbuffer::create_depth_objects;
use crate::descriptor::{create_descriptor_pool, create_descriptor_set_layout, create_descriptor_sets};
use crate::device::{create_logical_device, pick_physical_device};
use crate::image::create_color_objects;
use crate::models::load_models;
use crate::pipeline::create_pipeline;
use crate::renderpass::create_render_pass;
use crate::swpachain::{create_swapchain, create_swapchain_image_views};
use crate::texture::{create_texture_image, create_texture_image_view, create_texture_sampler};
use crate::ubfo::{create_uniform_buffers, UniformBufferObject};
use crate::vertex::{create_index_buffer, create_vertex_buffer};

/// Whether the validation layers should be enabled.
pub const VALIDATION_ENABLED: bool = cfg!(debug_assertions);

/// The name of the validation layers.
pub const VALIDATION_LAYER: vk::ExtensionName = vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");

pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

#[derive(Clone, Debug)]
pub struct App {
    entry: Entry,
    instance: Instance,
    data: AppData,
    pub(crate) logical_device: Device,
    frame: usize,
    pub(crate) resized: bool,
    start: Instant,
    pub(crate) models: usize,
}

#[derive(Debug, Error)]
#[error("Missing {0}.")]
pub struct SuitabilityError(pub &'static str);

extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    type_: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> vk::Bool32 {
    let data = unsafe { *data };
    let message = unsafe { CStr::from_ptr(data.message) }.to_string_lossy();

    if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
        error!("({:?}) {}", type_, message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::WARNING {
        warn!("({:?}) {}", type_, message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::INFO {
        debug!("({:?}) {}", type_, message);
    } else {
        trace!("({:?}) {}", type_, message);
    }

    vk::FALSE
}

unsafe fn create_instance(window: &Window,
                          entry: &Entry,
                          data: &mut AppData) -> Result<Instance, Error> {
    let application_info = vk::ApplicationInfo::builder()
        .application_name(b"Vulkan Tutorial\0")
        .application_version(vk::make_version(1, 0, 0))
        .engine_name(b"No engine")
        .engine_version(vk::make_version(1, 0, 0))
        .api_version(vk::make_version(1, 0, 0));

    // Layers

    let available_layers = entry
        .enumerate_instance_layer_properties()?
        .iter()
        .map(|l| l.layer_name)
        .collect::<HashSet<_>>();

    if VALIDATION_ENABLED && !available_layers.contains(&VALIDATION_LAYER) {
        return Err(anyhow!("Validation layer requested but not supported."));
    }

    let layers = if VALIDATION_ENABLED {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        Vec::new()
    };

    let mut extensions = vk_window::get_required_instance_extensions(window)
        .iter()
        .map(|e| e.as_ptr())
        .collect::<Vec<_>>();

    if VALIDATION_ENABLED {
        extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION.name.as_ptr());
    }


    let mut info = vk::InstanceCreateInfo::builder()
        .application_info(&application_info)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions);

    let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
        .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
        .user_callback(Some(debug_callback));

    if VALIDATION_ENABLED {
        info = info.push_next(&mut debug_info);
    }

    let instance = entry.create_instance(&info, None)?;

    if VALIDATION_ENABLED {
        data.messenger = instance.create_debug_utils_messenger_ext(&debug_info, None)?;
    }

    Ok(instance)
}

impl App {
    /// Creates Vulkan app
    pub unsafe fn create(window: &Window) -> Result<Self, Error> {
        let loader = LibloadingLoader::new(LIBRARY)?;
        let entry = Entry::new(loader).map_err(|b| anyhow!("{}", b))?;
        let mut data = AppData::default();

        let instance = create_instance(window, &entry, &mut data)?;

        data.surface = vk_window::create_surface(&instance, window)?;

        pick_physical_device(&instance, &mut data)?;
        let logical_device = create_logical_device(&instance, &mut data)?;

        create_swapchain(window, &instance, &logical_device, &mut data)?;
        create_swapchain_image_views(&logical_device, &mut data)?;

        create_render_pass(&instance, &logical_device, &mut data)?;

        create_descriptor_set_layout(&logical_device, &mut data)?;
        create_pipeline(&logical_device, &mut data)?;
        create_command_pools(&instance, &logical_device, &mut data)?;

        create_color_objects(&instance, &logical_device, &mut data)?;
        create_depth_objects(&instance, &logical_device, &mut data)?;
        create_framebuffers(&logical_device, &mut data)?;

        create_texture_image(&instance, &logical_device, &mut data)?;
        create_texture_image_view(&logical_device, &mut data)?;
        create_texture_sampler(&logical_device, &mut data)?;

        load_models(&mut data)?;
        create_vertex_buffer(&instance, &logical_device, &mut data)?;
        create_index_buffer(&instance, &logical_device, &mut data)?;

        create_uniform_buffers(&instance, &logical_device, &mut data)?;
        create_descriptor_pool(&logical_device, &mut data)?;
        create_descriptor_sets(&logical_device, &mut data)?;

        create_command_buffers(&logical_device, &mut data)?;

        create_sync_objects(&logical_device, &mut data)?;

        Ok(Self { entry, instance, data, logical_device, frame: 0, resized: false, start: Instant::now(), models: 1 })
    }

    /// Renders a frame for Vulkan app
    pub unsafe fn render(&mut self, window: &Window) -> Result<(), Error> {
        self.logical_device.wait_for_fences(
            &[self.data.in_flight_fences[self.frame]],
            true,
            u64::MAX,
        )?;

        let result = self
            .logical_device
            .acquire_next_image_khr(
                self.data.swapchain,
                u64::MAX,
                //Les objets de synchro qui devront être signalé quand la partie pres à finit d'utiliser les images
                self.data.image_available_semaphores[self.frame],
                vk::Fence::null(),
            );

        //Récupération de l'index d'une image disponnible
        let image_index = match result {
            Ok((image_index, _)) => image_index as usize,
            Err(vk::ErrorCode::OUT_OF_DATE_KHR) => return self.recreate_swapchain(window),
            Err(e) => return Err(anyhow!(e)),
        };

        if !self.data.images_in_flight[image_index as usize].is_null() {
            self.logical_device.wait_for_fences(
                &[self.data.images_in_flight[image_index as usize]],
                true,
                u64::MAX,
            )?;
        }

        self.data.images_in_flight[image_index as usize] = self.data.in_flight_fences[self.frame];

        self.update_command_buffer(image_index)?;
        self.update_uniform_buffer(image_index)?;

        //Spécifique quelle sémaphore il faut attendre avant que l'execution ne commence
        let wait_semaphores = &[self.data.image_available_semaphores[self.frame]];
        //On souhaite attendre de pouvoir appliquer les couleurs,
        // donc que l'image soit dispo pour la stage qui écrit sur le color_attachment (si j'ai bien compris)
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];

        let command_buffers = &[self.data.command_buffers[image_index]];
        //Les sémaphore à signaler quand le.s command_buffer a finit de s'éxecuter
        let signal_semaphores = &[self.data.render_finished_semaphores[self.frame]];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);

        self.logical_device.reset_fences(&[self.data.in_flight_fences[self.frame]])?;

        self.logical_device.queue_submit(
            self.data.graphics_queue,
            &[submit_info],
            self.data.in_flight_fences[self.frame],
        )?;

        //PRESENTATION
        let swapchains = &[self.data.swapchain];
        let image_indices = &[image_index as u32];
        let presentation_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);

        let result = self.logical_device.queue_present_khr(
            self.data.prensentation_queue, &presentation_info,
        );
        let changed = result == Ok(vk::SuccessCode::SUBOPTIMAL_KHR)
            || result == Err(vk::ErrorCode::OUT_OF_DATE_KHR);

        if self.resized || changed {
            self.recreate_swapchain(window)?;
        } else if let Err(e) = result {
            return Err(anyhow!(e));
        }

        self.frame = (self.frame + 1) % MAX_FRAMES_IN_FLIGHT;

        Ok(())
    }

    unsafe fn update_uniform_buffer(
        &self,
        image_index: usize,
    ) -> Result<(), Error> {
        let time = self.start.elapsed().as_secs_f32();

        let model = glm::rotate(
            &glm::identity(),
            time * glm::radians(&glm::vec1(90.0))[0],
            &glm::vec3(0.0, 0.0, 1.0),
        );

        let view = glm::look_at(
            &glm::vec3(6.0, 2.0, 2.0),
            &glm::vec3(0.0, 0.0, 0.0),
            &glm::vec3(0.0, 0.0, 1.0),
        );

        let mut proj = glm::perspective_rh_zo(
            self.data.swapchain_extent.width as f32 / self.data.swapchain_extent.height as f32,
            glm::radians(&glm::vec1(45.0))[0],
            0.1,
            10.0,
        );

        proj[(1, 1)] *= -1.0;

        let ubo = UniformBufferObject { view, proj };

        let memory = self.logical_device.map_memory(
            self.data.uniform_buffers_memory[image_index],
            0,
            size_of::<UniformBufferObject>() as u64,
            vk::MemoryMapFlags::empty(),
        )?;

        memcpy(&ubo, memory.cast(), 1);

        self.logical_device.unmap_memory(self.data.uniform_buffers_memory[image_index]);

        Ok(())
    }

    unsafe fn update_command_buffer(&mut self, image_index: usize) -> Result<(), Error> {
        let command_pool = self.data.command_pools[image_index];
        self.logical_device.reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())?;

        let command_buffer = self.data.command_buffers[image_index];

        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        self.logical_device.begin_command_buffer(command_buffer, &info)?;

        let render_area = vk::Rect2D::builder()
            .offset(vk::Offset2D::default())
            .extent(self.data.swapchain_extent);

        let color_clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };

        let depth_clear_value = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        };

        let clear_values = &[color_clear_value, depth_clear_value];
        let info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.data.render_pass)
            .framebuffer(self.data.framebuffers[image_index])
            .render_area(render_area)
            .clear_values(clear_values);

        self.logical_device.cmd_begin_render_pass(
            command_buffer,
            &info,
            vk::SubpassContents::SECONDARY_COMMAND_BUFFERS,
        );

        let secondary_command_buffer = (0..self.models)
            .map(|i| self.update_secondary_command_buffer(image_index, i))
            .collect::<Result<Vec<_>, _>>()?;
        self.logical_device.cmd_execute_commands(command_buffer, &secondary_command_buffer[..]);

        self.logical_device.cmd_end_render_pass(command_buffer);
        self.logical_device.end_command_buffer(command_buffer)?;

        Ok(())
    }

    unsafe fn update_secondary_command_buffer(
        &mut self,
        image_index: usize,
        model_index: usize,
    ) -> Result<vk::CommandBuffer, Error> {
        self.data.secondary_command_buffers.resize_with(image_index + 1, Vec::new);

        let command_buffers = &mut self.data.secondary_command_buffers[image_index];

        while model_index >= command_buffers.len() {
            let allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(self.data.command_pools[image_index])
                .level(vk::CommandBufferLevel::SECONDARY)
                .command_buffer_count(1);

            let command_buffer = self.logical_device.allocate_command_buffers(&allocate_info)?[0];
            command_buffers.push(command_buffer);
        }

        let command_buffer = command_buffers[model_index];

        let y = (((model_index % 2) as f32) * 2.5) - 1.25;
        let z = (((model_index / 2) as f32) * -2.0) + 1.0;

        let model = glm::translate(
            &glm::identity(),
            &glm::vec3(0.0, y, z),
        );

        let time = self.start.elapsed().as_secs_f32();

        let model = glm::rotate(
            &model,
            time * glm::radians(&glm::vec1(90.0))[0],
            &glm::vec3(0.0, 0.0, 1.0),
        );

        let (_, model_bytes, _) = model.as_slice().align_to::<u8>();

        let opacity = (model_index + 1) as f32 * 0.25;
        let opacity_bytes = &opacity.to_ne_bytes()[..];

        let inheritance_info = vk::CommandBufferInheritanceInfo::builder()
            .render_pass(self.data.render_pass)
            .subpass(0)
            .framebuffer(self.data.framebuffers[image_index]);
        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE)
            .inheritance_info(&inheritance_info);

        self.logical_device.begin_command_buffer(command_buffer, &info)?;

        self.logical_device.cmd_bind_pipeline(
            command_buffer, vk::PipelineBindPoint::GRAPHICS, self.data.pipeline,
        );

        self.logical_device.cmd_bind_vertex_buffers(
            command_buffer,
            0,
            &[self.data.vertex_buffer],
            &[0],
        );
        self.logical_device.cmd_bind_index_buffer(
            command_buffer,
            self.data.index_buffer,
            0,
            vk::IndexType::UINT32,
        );
        self.logical_device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.data.pipeline_layout,
            0,
            &[self.data.descriptor_sets[image_index]],
            &[],
        );

        //Pour matrice model
        self.logical_device.cmd_push_constants(
            command_buffer,
            self.data.pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            model_bytes,
        );

        //Pour opacity
        self.logical_device.cmd_push_constants(
            command_buffer,
            self.data.pipeline_layout,
            vk::ShaderStageFlags::FRAGMENT,
            64,
            opacity_bytes, //opacité de 0.2
        );

        self.logical_device.cmd_draw_indexed(
            command_buffer,
            self.data.indices.len() as u32,
            1,
            0,
            0,
            0,
        );

        self.logical_device.end_command_buffer(command_buffer)?;

        Ok(command_buffer)
    }

    unsafe fn recreate_swapchain(&mut self, window: &Window) -> Result<(), Error> {
        self.logical_device.device_wait_idle()?;
        self.destroy_swapchain();

        create_swapchain(window, &self.instance, &self.logical_device, &mut self.data)?;
        create_swapchain_image_views(&self.logical_device, &mut self.data)?;
        create_render_pass(&self.instance, &self.logical_device, &mut self.data)?;
        create_pipeline(&self.logical_device, &mut self.data)?;

        create_color_objects(&self.instance, &self.logical_device, &mut self.data)?;
        create_depth_objects(&self.instance, &self.logical_device, &mut self.data)?;

        create_framebuffers(&self.logical_device, &mut self.data)?;
        create_uniform_buffers(&self.instance, &self.logical_device, &mut self.data)?;
        create_descriptor_pool(&self.logical_device, &mut self.data)?;
        create_descriptor_sets(&self.logical_device, &mut self.data)?;

        create_command_buffers(&self.logical_device, &mut self.data)?;

        self.data
            .images_in_flight
            .resize(self.data.swapchain_images.len(), vk::Fence::null())
        ;

        Ok(())
    }

    unsafe fn destroy_swapchain(&mut self) {
        self.logical_device.destroy_image_view(self.data.color_image_view, None);
        self.logical_device.free_memory(self.data.color_image_memory, None);
        self.logical_device.destroy_image(self.data.color_image, None);

        self.logical_device.destroy_image_view(self.data.depth_image_view, None);
        self.logical_device.free_memory(self.data.depth_image_memory, None);
        self.logical_device.destroy_image(self.data.depth_image, None);

        self.logical_device.destroy_descriptor_pool(self.data.descriptor_pool, None);

        self.data.uniform_buffers
            .iter()
            .for_each(|b| self.logical_device.destroy_buffer(*b, None));
        self.data.uniform_buffers_memory
            .iter()
            .for_each(|m| self.logical_device.free_memory(*m, None));

        self.data.framebuffers
            .iter()
            .for_each(|f| self.logical_device.destroy_framebuffer(*f, None));

        self.logical_device.destroy_pipeline(self.data.pipeline, None);
        self.logical_device.destroy_pipeline_layout(self.data.pipeline_layout, None);
        self.logical_device.destroy_render_pass(self.data.render_pass, None);
        self.data.swapchain_image_views
            .iter()
            .for_each(|v| self.logical_device.destroy_image_view(*v, None));
        self.logical_device.destroy_swapchain_khr(self.data.swapchain, None);
    }

    /// Destroys Vulkan app
    #[rustfmt::skip]
    pub(crate) unsafe fn destroy(&mut self) {
        self.destroy_swapchain();

        self.data.command_pools
            .iter()
            .for_each(|p| self.logical_device.destroy_command_pool(*p, None));
        self.logical_device.destroy_sampler(self.data.texture_sampler, None);
        self.logical_device.destroy_image_view(self.data.texture_image_view, None);
        self.logical_device.destroy_image(self.data.texture_image, None);
        self.logical_device.free_memory(self.data.texture_image_memory, None);

        self.logical_device.destroy_descriptor_set_layout(self.data.descriptor_set_layout, None);
        self.logical_device.destroy_buffer(self.data.index_buffer, None);
        self.logical_device.free_memory(self.data.index_buffer_memory, None);
        self.logical_device.destroy_buffer(self.data.vertex_buffer, None);
        self.logical_device.free_memory(self.data.vertex_buffer_memory, None);

        self.data.in_flight_fences
            .iter()
            .for_each(|f| self.logical_device.destroy_fence(*f, None));
        self.data.render_finished_semaphores
            .iter()
            .for_each(|s| self.logical_device.destroy_semaphore(*s, None));
        self.data.image_available_semaphores
            .iter()
            .for_each(|s| self.logical_device.destroy_semaphore(*s, None));

        self.logical_device.destroy_command_pool(self.data.command_pool, None);
        self.logical_device.destroy_device(None);
        self.instance.destroy_surface_khr(self.data.surface, None);

        if VALIDATION_ENABLED {
            self.instance.destroy_debug_utils_messenger_ext(self.data.messenger, None);
        }

        self.instance.destroy_instance(None);
    }
}
