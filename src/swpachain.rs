use anyhow::Error;
use vulkanalia::{Device, Instance, vk};
use vulkanalia::vk::{Handle, HasBuilder, KhrSurfaceExtension, KhrSwapchainExtension};
use winit::window::Window;

use crate::AppData;
use crate::image::create_image_view;
use crate::queuefamily::QueueFamilyIndices;

#[derive(Clone, Debug)]
pub struct SwapchainSupport {
    capabilities: vk::SurfaceCapabilitiesKHR,
    pub(crate) formats: Vec<vk::SurfaceFormatKHR>,
    pub(crate) presentation_modes: Vec<vk::PresentModeKHR>,
}

/////// SWAPCHAIN ///////
impl SwapchainSupport {
    pub unsafe fn get(
        instance: &Instance,
        data: &AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self, Error> {
        Ok(Self {
            capabilities: instance
                .get_physical_device_surface_capabilities_khr(
                    physical_device,
                    data.surface
                )?,
            formats: instance
                .get_physical_device_surface_formats_khr(
                    physical_device,
                    data.surface
                )?,
            presentation_modes: instance
                .get_physical_device_surface_present_modes_khr(
                    physical_device,
                    data.surface
                )?,
        })
    }

}


pub unsafe fn create_swapchain(
    window: &Window,
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<(), Error> {
    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;
    let support = SwapchainSupport::get(instance, data, data.physical_device)?;

    let surface_format = get_swapchain_surface_format(&support.formats);
    let presentation_mode = get_swapchain_presentation_mode(&support.presentation_modes);
    let extent = get_swapchain_extent(window, support.capabilities);

    data.swapchain_format = surface_format.format;
    data.swapchain_extent = extent;

    let mut image_count = support.capabilities.min_image_count + 1;
    if support.capabilities.max_image_count != 0
        && image_count > support.capabilities.max_image_count
    {
        image_count = support.capabilities.max_image_count;
    }

    let mut queue_family_indices = vec![];
    let image_sharing_mode = if indices.graphics != indices.presentation {
        queue_family_indices.push(indices.graphics);
        queue_family_indices.push(indices.presentation);
        vk::SharingMode::CONCURRENT
    } else {
        vk::SharingMode::EXCLUSIVE
    };

    let info = vk::SwapchainCreateInfoKHR::builder()
        .surface(data.surface)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(image_sharing_mode)
        .queue_family_indices(&queue_family_indices)
        // Veut dire qu'on ne souhaite pas appliquer de transformation
        .pre_transform(support.capabilities.current_transform)
        //Spécifie si le channel alpha (aucune idée de ce que c'est)
        // doit être utilisé pour se fondre avec les autres fenêtre
        // Ici non
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(presentation_mode)
        .clipped(true)
        //Si on est amené à recréer la swapchain, par exemple parce qu'on resize la fenêtre,
        // il faut recréer la swapchain de zéro et on doit donc spécifier l'ancienne
        // Pour l'instant on ne fait rien
        .old_swapchain(vk::SwapchainKHR::null());

    data.swapchain = device.create_swapchain_khr(&info, None)?;
    data.swapchain_images = device.get_swapchain_images_khr(data.swapchain)?;


    Ok(())
}

pub unsafe fn create_swapchain_image_views(
    device: &Device,
    data: &mut AppData,
) -> Result<(), Error> {

    data.swapchain_image_views = data
        .swapchain_images
        .iter()
        .map(|i| {
            create_image_view(
                device,
                *i,
                data.swapchain_format,
                vk::ImageAspectFlags::COLOR,
                1,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(())
}

fn get_swapchain_surface_format(
    formats: &[vk::SurfaceFormatKHR],
) -> vk::SurfaceFormatKHR {
    formats
        .iter()
        .cloned()
        .find(|f| {
            f.format == vk::Format::B8G8R8A8_SRGB
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .unwrap_or_else(|| formats[0])
}

fn get_swapchain_presentation_mode(
    presentation_mode: &[vk::PresentModeKHR],
) -> vk::PresentModeKHR {
    presentation_mode
        .iter()
        .cloned()
        .find(|m| *m == vk::PresentModeKHR::MAILBOX)
        .unwrap_or(vk::PresentModeKHR::FIFO)
}

fn get_swapchain_extent(
    window: &Window,
    capabilities: vk::SurfaceCapabilitiesKHR,
) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::max_value() {
        capabilities.current_extent
    } else {
        let size = window.inner_size();
        let clamp = |min: u32, max: u32, v: u32| min.max(max.min(v));
        vk::Extent2D::builder()
            .width(clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
                size.width,
            ))
            .height(clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
                size.height
            ))
            .build()
    }
}
