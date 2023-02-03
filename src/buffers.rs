use anyhow::Error;
use vulkanalia::{Device, Instance, vk};
use vulkanalia::vk::{DeviceV1_0, HasBuilder};
use crate::{AppData, get_memory_type_index};
use crate::commandbuffer::{begin_single_time_commands, end_single_time_commands};

pub unsafe fn create_buffer(
    instance: &Instance,
    logical_device: &Device,
    data: &AppData,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory), Error> {
    let buffer_info = vk::BufferCreateInfo::builder()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = logical_device.create_buffer(&buffer_info, None)?;

    let requirements = logical_device.get_buffer_memory_requirements(buffer);

    let memory_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size)
        .memory_type_index(get_memory_type_index(
            instance,
            data,
            properties,
            requirements,
        )?);

    let buffer_memory = logical_device.allocate_memory(&memory_info, None)?;
    logical_device.bind_buffer_memory(buffer, buffer_memory, 0)?;

    Ok((buffer, buffer_memory))
}

pub unsafe fn copy_buffer(
    logical_device: &Device,
    data: &AppData,
    source: vk::Buffer,
    destination: vk::Buffer,
    size: vk::DeviceSize,
) -> Result<(), Error> {
    let command_buffer = begin_single_time_commands(logical_device, data)?;

    let regions = vk::BufferCopy::builder().size(size);
    logical_device.cmd_copy_buffer(command_buffer, source, destination, &[regions]);

    end_single_time_commands(logical_device, data, command_buffer)?;

    Ok(())
}