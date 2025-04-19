use std::sync::atomic::AtomicU64;

use ash::vk;

use crate::{NanoSecond, ScopeId};

pub trait VulkanBuffer {
    fn mapped_slice(&self) -> &[u8];
    fn raw(&self) -> ash::vk::Buffer;
}

pub trait VulkanBackend {
    type Buffer: VulkanBuffer;

    fn create_query_result_buffer(&mut self, bytes: usize) -> Self::Buffer;
    fn timestamp_period(&self) -> f32;
}

pub struct VulkanProfilerFrame<Buffer: VulkanBuffer> {
    buffer: Buffer,
    query_pool: vk::QueryPool,
    next_query_idx: std::sync::atomic::AtomicU32,
    query_scope_ids: Box<[AtomicU64]>,
    timestamp_period: f32,
}

const MAX_QUERY_COUNT: usize = 1024;
type DurationRange = [u64; 2];

pub struct VulkanActiveScope {
    query_id: u32,
}

impl<Buffer: VulkanBuffer> VulkanProfilerFrame<Buffer> {
    pub fn new(device: &ash::Device, mut backend: impl VulkanBackend<Buffer = Buffer>) -> Self {
        let buffer = backend
            .create_query_result_buffer(MAX_QUERY_COUNT * std::mem::size_of::<DurationRange>());

        let pool_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(MAX_QUERY_COUNT as u32 * 2);

        Self {
            buffer,
            query_pool: unsafe { device.create_query_pool(&pool_info, None) }
                .expect("create_query_pool"),
            next_query_idx: Default::default(),
            query_scope_ids: (1..MAX_QUERY_COUNT)
                .map(|_| AtomicU64::new(ScopeId::invalid().as_u64()))
                .collect::<Vec<AtomicU64>>()
                .into(),
            timestamp_period: backend.timestamp_period(),
        }
    }

    pub fn begin_scope(
        &self,
        device: &ash::Device,
        cb: ash::vk::CommandBuffer,
        scope_id: ScopeId,
    ) -> VulkanActiveScope {
        let query_id = self
            .next_query_idx
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        self.query_scope_ids[query_id as usize]
            .store(scope_id.as_u64(), std::sync::atomic::Ordering::Relaxed);

        unsafe {
            device.cmd_write_timestamp(
                cb,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.query_pool,
                query_id * 2,
            );
        }

        VulkanActiveScope { query_id }
    }

    pub fn end_scope(
        &self,
        device: &ash::Device,
        cb: ash::vk::CommandBuffer,
        active_scope: VulkanActiveScope,
    ) {
        unsafe {
            device.cmd_write_timestamp(
                cb,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.query_pool,
                active_scope.query_id * 2 + 1,
            );
        }
    }

    /// Call this before recording any profiling scopes in the frame
    pub fn begin_frame(&self, device: &ash::Device, cmd: vk::CommandBuffer) {
        self.report_durations();

        unsafe {
            device.cmd_reset_query_pool(cmd, self.query_pool, 0, MAX_QUERY_COUNT as u32 * 2);
        }

        self.next_query_idx
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// Call this after recording all profiling scopes in the frame
    pub fn end_frame(&self, device: &ash::Device, cmd: vk::CommandBuffer) {
        let valid_query_count = self
            .next_query_idx
            .load(std::sync::atomic::Ordering::Relaxed);

        unsafe {
            device.cmd_copy_query_pool_results(
                cmd,
                self.query_pool,
                0,
                valid_query_count * 2,
                self.buffer.raw(),
                0,
                8,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            );
        }
    }

    fn report_durations(&self) {
        let previous_results = self.retrieve_previous_results();
        let ns_per_tick = self.timestamp_period as f64;

        crate::profiler().report_durations(previous_results.into_iter().map(
            |(scope_id, duration_range)| {
                let duration_ticks = duration_range[1].saturating_sub(duration_range[0]);
                let duration =
                    NanoSecond::from_raw_ns((duration_ticks as f64 * ns_per_tick) as u64);

                (scope_id, duration)
            },
        ));
    }

    fn retrieve_previous_results(&self) -> Vec<(ScopeId, DurationRange)> {
        let valid_query_count = self
            .next_query_idx
            .load(std::sync::atomic::Ordering::Relaxed) as usize;

        let mapped_slice = self.buffer.mapped_slice();

        assert_eq!(mapped_slice.len() % std::mem::size_of::<DurationRange>(), 0);
        assert!(mapped_slice.len() / std::mem::size_of::<DurationRange>() >= valid_query_count);

        let durations = unsafe {
            std::slice::from_raw_parts(
                mapped_slice.as_ptr() as *const DurationRange,
                valid_query_count,
            )
        };

        let scopes = self.query_scope_ids[0..valid_query_count]
            .iter()
            .map(|val| ScopeId::from_u64(val.load(std::sync::atomic::Ordering::Relaxed)));

        let mut result: Vec<(ScopeId, [u64; 2])> =
            scopes.zip(durations.into_iter().copied()).collect();

        result.sort_unstable_by_key(|(scope, _)| *scope);

        result
    }
}
