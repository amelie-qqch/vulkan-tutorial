use anyhow::Error;
use vulkanalia::{Device, Instance, vk};
use vulkanalia::vk::{DeviceV1_0, Handle, HasBuilder};
use crate::AppData;
use crate::queuefamily::QueueFamilyIndices;

pub unsafe fn begin_single_time_commands(
    logical_device: &Device,
    data: &AppData,
) -> Result<vk::CommandBuffer, Error> {
    let info = vk::CommandBufferAllocateInfo::builder()
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(data.command_pool)
        .command_buffer_count(1);

    let command_buffer = logical_device.allocate_command_buffers(&info)?[0];

    let info = vk::CommandBufferBeginInfo::builder()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    logical_device.begin_command_buffer(command_buffer, &info)?;

    Ok(command_buffer)
}

pub unsafe fn end_single_time_commands(
    logical_device: &Device,
    data: &AppData,
    command_buffer: vk::CommandBuffer,
) -> Result<(), Error> {
    logical_device.end_command_buffer(command_buffer)?;

    let command_buffers = &[command_buffer];
    let info = vk::SubmitInfo::builder()
        .command_buffers(command_buffers);

    logical_device.queue_submit(data.graphics_queue, &[info], vk::Fence::null())?;
    logical_device.queue_wait_idle(data.graphics_queue)?;

    logical_device.free_command_buffers(data.command_pool, &[command_buffer]);

    Ok(())
}

// Les command_pool gèrent la mémoire utilisée pour stocker les buffers,
// et les command_buffer sont alloués à partir de ça.
pub unsafe fn create_command_pools(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<(), Error> {
    data.command_pool = create_command_pool(instance, device, data)?;

    let num_images = data.swapchain_images.len();
    for _ in 0..num_images {
        let command_pool = create_command_pool(instance, device, data)?;
        data.command_pools.push(command_pool);
    }

    Ok(())
}
pub unsafe fn create_command_pool(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<vk::CommandPool, Error> {
    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;

    let info = vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::TRANSIENT)
        .queue_family_index(indices.graphics);

    Ok(device.create_command_pool(&info, None)?)
}

pub unsafe fn create_command_buffers(device: &Device, data: &mut AppData) -> Result<(), Error> {
    let num_images = data.swapchain_images.len();

    for image_index in 0..num_images {
        let allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(data.command_pools[image_index])
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffer = device.allocate_command_buffers(&allocate_info)?[0];
        data.command_buffers.push(command_buffer);
    }

    data.secondary_command_buffers = vec![vec![]; data.swapchain_images.len()];

    Ok(())
}