use anyhow::{anyhow, Error};
use vulkanalia::{Instance, vk};
use vulkanalia::vk::{InstanceV1_0, KhrSurfaceExtension};
use crate::AppData;
use crate::app::{SuitabilityError};

#[derive(Copy, Clone, Debug)]
pub struct QueueFamilyIndices {
    pub(crate) graphics: u32,
    pub(crate) presentation: u32,
}

impl QueueFamilyIndices {
    pub(crate) unsafe fn get(
        instance: &Instance,
        data: &AppData,
        physical_device: vk::PhysicalDevice
    ) -> Result<Self, Error> {
        let properties = instance.get_physical_device_queue_family_properties(physical_device);
        let graphics = properties
            .iter()
            .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|i| i as u32);

        let mut presentation = None;
        for (index, properties) in properties.iter().enumerate() {
            if instance.get_physical_device_surface_support_khr(
                physical_device,
                index as u32,
                data.surface
            )? {
                presentation = Some(index as u32);
                break;
            }
        }

        if let (Some(graphics), Some(presentation)) = (graphics, presentation) {
            Ok(Self { graphics, presentation })
        } else {
            Err(anyhow!(SuitabilityError("Missing required queue families.")))
        }
    }
}