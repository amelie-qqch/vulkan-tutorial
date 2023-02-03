use std::mem::size_of;
use anyhow::Error;
use nalgebra_glm as glm;
use vulkanalia::{Device, Instance, vk};
use crate::AppData;
use crate::buffers::create_buffer;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct UniformBufferObject {
    pub(crate) view: glm::Mat4,
    pub(crate) proj: glm::Mat4,
}

pub unsafe fn create_uniform_buffers(
    instance: &Instance,
    logical_device: &Device,
    data: &mut AppData,
) -> Result<(), Error> {
    data.uniform_buffers.clear();
    data.uniform_buffers_memory.clear();

    for _ in 0..data.swapchain_images.len() {
        let (uniform_buffer, uniform_buffer_memory) = create_buffer(
            instance,
            logical_device,
            data,
            size_of::<UniformBufferObject>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;

        data.uniform_buffers.push(uniform_buffer);
        data.uniform_buffers_memory.push(uniform_buffer_memory);
    }

    Ok(())
}