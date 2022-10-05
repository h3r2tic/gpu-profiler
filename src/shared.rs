use std::sync::Mutex;

const MAX_FRAMES_IN_FLIGHT: usize = 4;

pub fn profiler() -> std::sync::MutexGuard<'static, GpuProfiler> {
    use once_cell::sync::Lazy;
    static GLOBAL_PROFILER: Lazy<Mutex<GpuProfiler>> = Lazy::new(Default::default);
    GLOBAL_PROFILER.lock().unwrap()
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ScopeId {
    frame: u32,
    scope: u32,
}

impl ScopeId {
    pub fn invalid() -> Self {
        Self {
            frame: !0,
            scope: !0,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct FrameScopeId {
    scope: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct NanoSecond(u64);

impl NanoSecond {
    pub fn from_raw_ns(ns: u64) -> Self {
        Self(ns)
    }

    pub fn raw_ns(self) -> u64 {
        self.0
    }

    pub fn ms(self) -> f64 {
        self.0 as f64 / 1_000_000.0
    }
}

#[derive(Clone)]
struct Scope {
    name: String,
}

#[derive(Default, Clone, Copy)]
enum FrameState {
    #[default]
    Invalid,
    Begin {
        index: u32,
    },
    End {
        index: u32,
    },
    Reported,
}

#[derive(Default, Clone)]
struct Frame {
    state: FrameState,
    scopes: Vec<Scope>,
}

pub struct GpuProfiler {
    frames: Vec<Frame>,
    frame_idx: u32,

    last_report: Option<TimedFrame>,
}

#[derive(Clone)]
pub struct TimedScope {
    pub name: String,
    pub duration: NanoSecond,
}

#[derive(Default, Clone)]
pub struct TimedFrame {
    pub scopes: Vec<TimedScope>,
}

impl Default for GpuProfiler {
    fn default() -> Self {
        Self {
            frames: vec![Default::default(); MAX_FRAMES_IN_FLIGHT],
            frame_idx: Default::default(),
            last_report: Default::default(),
        }
    }
}

impl GpuProfiler {
    pub fn begin_frame(&mut self) {
        let frame_idx = self.frame_idx;
        let frame = self.frame_mut();

        assert!(
            !matches!(frame.state, FrameState::Begin { .. }),
            "begin_frame called twice"
        );

        frame.state = FrameState::Begin { index: frame_idx };
        frame.scopes.clear();
    }

    pub fn end_frame(&mut self) {
        let frame = self.frame_mut();
        frame.state = match frame.state {
            FrameState::Invalid | FrameState::Reported => {
                panic!("end_frame called without begin_frame")
            }
            FrameState::Begin { index } => FrameState::End { index },
            FrameState::End { .. } => panic!("end_frame called twice"),
        };

        self.frame_idx += 1;
    }

    pub fn create_scope(&mut self, name: impl Into<String>) -> ScopeId {
        let frame = self.frame_mut();
        let next_scope_id = frame.scopes.len() as _;

        frame.scopes.push(Scope { name: name.into() });

        ScopeId {
            frame: self.frame_idx as _,
            scope: next_scope_id,
        }
    }

    pub fn report_durations(&mut self, mut durations: impl Iterator<Item = (ScopeId, NanoSecond)>) {
        self.last_report = durations.next().map(|(scope_id, duration)| {
            // TODO: assert on the frame being in the valid range
            let first_scope_frame_idx = scope_id.frame;
            let frame_count = self.frames.len();

            let frame = &mut self.frames[first_scope_frame_idx as usize % frame_count];

            frame.state = match frame.state {
                FrameState::End { index } => {
                    assert!(index == first_scope_frame_idx);
                    FrameState::Reported
                }
                FrameState::Reported => {
                    panic!("report_durations called twice");
                }
                _ => {
                    panic!("report_durations called before end_frame");
                }
            };

            let timed_frame = std::iter::once(TimedScope {
                name: std::mem::take(&mut frame.scopes[scope_id.scope as usize].name),
                duration,
            })
            .chain(durations.map(|(scope_id, duration)| {
                assert!(scope_id.frame == first_scope_frame_idx);

                TimedScope {
                    name: std::mem::take(&mut frame.scopes[scope_id.scope as usize].name),
                    duration,
                }
            }));

            let scopes = timed_frame.collect();

            TimedFrame { scopes }
        });
    }

    fn frame_mut(&mut self) -> &mut Frame {
        let frame_count = self.frames.len();
        &mut self.frames[self.frame_idx as usize % frame_count]
    }
}

impl GpuProfiler {
    pub fn last_report(&self) -> Option<&TimedFrame> {
        self.last_report.as_ref()
    }

    pub fn take_last_report(&mut self) -> Option<TimedFrame> {
        self.last_report.take()
    }
}

impl TimedFrame {
    pub fn send_to_puffin(&self, gpu_frame_start_ns: puffin::NanoSecond) {
        let mut stream = puffin::Stream::default();
        let mut gpu_time_accum: puffin::NanoSecond = 0;
        let mut puffin_scope_count = 0;
        let main_gpu_scope_offset = stream.begin_scope(gpu_frame_start_ns, "frame", "", "");
        puffin_scope_count += 1;
        puffin_scope_count += self.scopes.len();
        for TimedScope { name, duration } in &self.scopes {
            let ns = duration.raw_ns() as puffin::NanoSecond;
            let offset = stream.begin_scope(gpu_frame_start_ns + gpu_time_accum, name, "", "");
            gpu_time_accum += ns;
            stream.end_scope(offset, gpu_frame_start_ns + gpu_time_accum);
        }
        stream.end_scope(main_gpu_scope_offset, gpu_frame_start_ns + gpu_time_accum);
        puffin::global_reporter(
            puffin::ThreadInfo {
                start_time_ns: None,
                name: "gpu".to_owned(),
            },
            &puffin::StreamInfo {
                num_scopes: puffin_scope_count,
                stream,
                depth: 1,
                range_ns: (gpu_frame_start_ns, gpu_frame_start_ns + gpu_time_accum),
            }
            .as_stream_into_ref(),
        );
    }
}
