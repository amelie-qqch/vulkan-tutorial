use std::fs::File;
use anyhow::Error;
use vulkanalia::{Device, Instance, vk};
use vulkanalia::vk::{DeviceV1_0, HasBuilder};
use crate::{AppData};
use crate::buffers::create_buffer;
use crate::image::{copy_buffer_to_image, create_image, create_image_view, transition_image_layout};
use crate::mipmaps::generate_mipmaps;
use std::ptr::copy_nonoverlapping as memcpy;

pub unsafe fn create_texture_image_view(logical_device: &Device, data: &mut AppData) -> Result<(), Error> {
    data.texture_image_view = create_image_view(
        logical_device,
        data.texture_image,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageAspectFlags::COLOR,
        data.mip_levels,
    )?;

    Ok(())
}

pub unsafe fn create_texture_image(
    instance: &Instance,
    logical_device: &Device,
    data: &mut AppData,
) -> Result<(), Error> {
    let image = File::open("resources/viking_room.png")?;

    let decoder = png::Decoder::new(image);
    let mut reader = decoder.read_info()?;

    let mut pixels = vec![0; reader.1.info().raw_bytes()];
    reader.1.next_frame(&mut pixels)?;

    let size = reader.1.info().raw_bytes() as u64;
    let (width, height) = reader.1.info().size();

    data.mip_levels = (width.max(height) as f32).log2().floor() as u32 + 1;

    let (staging_buffer, staging_buffer_memory) = create_buffer(
        instance,
        logical_device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE
    )?;

    let memory = logical_device.map_memory(
        staging_buffer_memory,
        0,
        size,
        vk::MemoryMapFlags::empty(),
    )?;

    memcpy(pixels.as_ptr(), memory.cast(), pixels.len());

    logical_device.unmap_memory(staging_buffer_memory);

    let (texture_image, texture_image_memory) = create_image(
        instance,
        logical_device,
        data,
        width,
        height,
        data.mip_levels,
        vk::SampleCountFlags::_1,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::SAMPLED |
            vk::ImageUsageFlags::TRANSFER_DST |
            vk::ImageUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    data.texture_image = texture_image;
    data.texture_image_memory = texture_image_memory;

    transition_image_layout(
        logical_device,
        data,
        data.texture_image,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        data.mip_levels,
    )?;

    copy_buffer_to_image(
        logical_device,
        data,
        staging_buffer,
        data.texture_image,
        width,
        height,
    )?;

    logical_device.destroy_buffer(staging_buffer, None);
    logical_device.free_memory(staging_buffer_memory, None);

    generate_mipmaps(
        instance,
        logical_device,
        data,
        data.texture_image,
        vk::Format::R8G8B8A8_SRGB,
        width,
        height,
        data.mip_levels,
    )?;

    Ok(())
}

pub unsafe fn create_texture_sampler(
    logical_device: &Device,
    data: &mut AppData
) -> Result<(), Error> {
    let info = vk::SamplerCreateInfo::builder()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT)
        .anisotropy_enable(true)
        .max_anisotropy(16.0)
        .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
        .unnormalized_coordinates(false)
        .compare_enable(false)
        .compare_op(vk::CompareOp::ALWAYS)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .mip_lod_bias(0.0)
        .min_lod(0.0)
        .max_lod(data.mip_levels as f32)
        .mip_lod_bias(0.0);

    data.texture_sampler = logical_device.create_sampler(&info, None)?;

    Ok(())
}