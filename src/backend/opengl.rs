mod types {
    use std::os::raw::{c_int, c_uint};

    pub type GLuint = c_uint;
    pub type GLenum = c_uint;
    pub type GLint = c_int;
    pub type GLuint64 = u64;
    pub type GLsizei = c_int;
}

use std::collections::VecDeque;

use types::*;

mod consts {
    #![allow(non_upper_case_globals)]

    use super::types::*;

    pub const QUERY_RESULT_AVAILABLE: GLenum = 0x8867;
    pub const QUERY_RESULT: GLenum = 0x8866;
    pub const TIME_ELAPSED: GLenum = 0x88BF;
}

use consts::*;

use crate::{NanoSecond, ScopeId};

#[allow(non_snake_case)]
pub trait GlBackend {
    fn GetQueryObjectiv(&mut self, id: GLuint, pname: GLenum, params: *mut GLint);
    fn GetQueryObjectui64v(&mut self, id: GLuint, pname: GLenum, params: *mut GLuint64);
    fn GenQueries(&mut self, n: GLsizei, ids: *mut GLuint);
    fn BeginQuery(&mut self, target: GLenum, id: GLuint);
    fn EndQuery(&mut self, target: GLenum);
}

const MAX_QUERY_COUNT: usize = 1024;

struct GlProfilerFrame {
    query_handles: Vec<GLuint>,
    next_query_idx: usize,
    query_scope_ids: Vec<ScopeId>,
    results_buffer: Vec<u64>,
}

pub struct GlActiveScope {
    query_handle: GLuint,
}

impl GlProfilerFrame {
    pub fn new(backend: &mut impl GlBackend) -> Self {
        let mut queries = vec![0; MAX_QUERY_COUNT];
        backend.GenQueries(MAX_QUERY_COUNT as _, queries.as_mut_ptr());
        Self {
            query_handles: queries,
            next_query_idx: 0,
            query_scope_ids: vec![ScopeId::invalid(); MAX_QUERY_COUNT],
            results_buffer: vec![0; MAX_QUERY_COUNT],
        }
    }

    pub fn begin_scope(
        &mut self,
        backend: &mut impl GlBackend,
        scope_id: ScopeId,
    ) -> GlActiveScope {
        let query_id = self.next_query_idx;
        self.next_query_idx += 1;

        self.query_scope_ids[query_id as usize] = scope_id;

        let query_handle = self.query_handles[query_id];

        backend.BeginQuery(TIME_ELAPSED, query_handle);

        GlActiveScope { query_handle }
    }

    pub fn end_scope(&self, backend: &mut impl GlBackend, active_scope: GlActiveScope) {
        assert!(active_scope.query_handle == self.query_handles[self.next_query_idx - 1]);
        backend.EndQuery(TIME_ELAPSED);
    }

    fn read_results(&mut self, backend: &mut impl GlBackend) -> Option<(&[ScopeId], &[u64])> {
        let result_count = self.next_query_idx;

        let results_available = self.query_handles[0..result_count]
            .iter()
            .all(|&query_handle| {
                let mut available: i32 = 0;
                backend.GetQueryObjectiv(query_handle, QUERY_RESULT_AVAILABLE, &mut available);
                available != 0
            });

        if results_available {
            for (&handle, result_nanos) in self
                .query_handles
                .iter()
                .zip(self.results_buffer.iter_mut())
                .take(result_count)
            {
                backend.GetQueryObjectui64v(handle, QUERY_RESULT, result_nanos);
            }

            Some((
                &self.query_scope_ids[0..result_count],
                &self.results_buffer[0..result_count],
            ))
        } else {
            None
        }
    }

    fn reset(&mut self) {
        self.next_query_idx = 0;
    }
}

#[derive(Default)]
pub struct GlProfiler {
    current_frame: Option<GlProfilerFrame>,
    waiting_frames: VecDeque<GlProfilerFrame>,
    frame_pool: Vec<GlProfilerFrame>,
}

impl GlProfiler {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn begin_frame(&mut self, backend: &mut impl GlBackend) {
        assert!(self.current_frame.is_none(), "begin_frame called twice");

        self.try_get_results(backend);

        crate::profiler().begin_frame();

        self.current_frame = Some(if let Some(frame) = self.frame_pool.pop() {
            frame
        } else {
            GlProfilerFrame::new(backend)
        });
    }

    pub fn end_frame(&mut self) {
        self.waiting_frames.push_back(
            self.current_frame
                .take()
                .expect("end_frame called before begin_frame"),
        );

        // Sanity check
        assert!(
            self.waiting_frames.len() < 10,
            "OpenGL queries failed to become available"
        );

        crate::profiler().end_frame();
    }

    pub fn begin_scope(
        &mut self,
        backend: &mut impl GlBackend,
        scope_id: ScopeId,
    ) -> GlActiveScope {
        self.current_frame
            .as_mut()
            .expect("begin_scope called before begin_frame")
            .begin_scope(backend, scope_id)
    }

    pub fn end_scope(&mut self, backend: &mut impl GlBackend, active_scope: GlActiveScope) {
        self.current_frame
            .as_mut()
            .expect("end_scope called before begin_frame")
            .end_scope(backend, active_scope)
    }

    fn try_get_results(&mut self, backend: &mut impl GlBackend) {
        while let Some(frame) = self.waiting_frames.front_mut() {
            if let Some((scopes, durations)) = frame.read_results(backend) {
                crate::profiler().report_durations(
                    scopes
                        .iter()
                        .zip(durations.iter())
                        .map(|(&scope, &duration)| (scope, NanoSecond::from_raw_ns(duration))),
                );

                let mut frame = self.waiting_frames.pop_front().unwrap();
                frame.reset();
                self.frame_pool.push(frame);
            } else {
                break;
            }
        }
    }
}
