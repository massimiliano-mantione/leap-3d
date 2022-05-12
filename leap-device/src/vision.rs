use crate::device::{LeapDevice, LeapDeviceResult, LeapMode, LeapStream};
use lib::nalgebra::{Vector2, Vector4};
use std::f32::consts::PI;

pub struct LeapVision<'a> {
    mode: LeapMode,
    stream: LeapStream<'a>,
}

impl<'a> LeapVision<'a> {
    pub fn mode(&self) -> &LeapMode {
        &self.mode
    }

    pub fn stream(&mut self) -> &LeapStream {
        &mut self.stream
    }

    pub fn new(device: &'a LeapDevice, buf_count: u32) -> LeapDeviceResult<Self> {
        let mode = *device.mode();
        let stream = device.stream(buf_count)?;
        Ok(Self { mode, stream })
    }
}

pub type Point = Vector2<f32>;

#[derive(Debug, Clone, Copy)]
pub struct PolarPoint {
    angle: f32,
    distance: f32,
}

impl PolarPoint {
    pub const ORIGIN: Self = Self {
        angle: 0.0,
        distance: 0.0,
    };

    pub fn apply_barrel_distortion(&mut self, barrel_distortion: f32) {
        // k x^2 + x - y = 0
        // x = (-b +- sqrt(b^2 -4ac)) / (2a)
        // a = k
        // b = 1
        // c = -1
        // x = (-1 +- sqrt(1 + 4k)) / 2k
        // x = (sqrt(1 + 4k) -1) / 2k
        self.distance =
            ((1.0 + (4.0 * barrel_distortion)) - 1.0).sqrt() / (2.0 + barrel_distortion);
    }

    pub fn remove_barrel_distortion(&mut self, barrel_distortion: f32) {
        // y = x + k x^2
        self.distance = self.distance + (barrel_distortion * self.distance * self.distance);
    }

    pub fn with_barrel_distortion(self, barrel_distortion: f32) -> Self {
        let mut result = self;
        result.apply_barrel_distortion(barrel_distortion);
        result
    }

    pub fn without_barrel_distortion(self, barrel_distortion: f32) -> Self {
        let mut result = self;
        result.remove_barrel_distortion(barrel_distortion);
        result
    }
}

impl From<Point> for PolarPoint {
    fn from(p: Point) -> Self {
        let distance = p.magnitude();
        if distance == 0.0 {
            PolarPoint::ORIGIN
        } else {
            let cos = p.x / distance;
            let angle = if p.y > 0.0 { cos.acos() } else { -cos.acos() };
            Self { angle, distance }
        }
    }
}

impl Into<Point> for PolarPoint {
    fn into(self) -> Point {
        if self.distance == 0.0 {
            Point::new(0.0, 0.0)
        } else {
            let x = self.angle.cos() * self.distance;
            let y = self.angle.sin() * self.distance;
            Point::new(x, y)
        }
    }
}

pub const LEAP_ASPECT_RATIO_LARGE: f32 = 752.0 / 480.0;
pub const LEAP_ASPECT_RATIO_SMALL: f32 = 640.0 / 480.0;
pub const LEAP_Y_APERTURE: f32 = 150.0 * 2.0 * PI / 360.0;
pub const LEAP_Y_APERTURE_MAX: f32 = 175.0 * 2.0 * PI / 360.0;
pub const LEAP_Y_APERTURE_MIN: f32 = 120.0 * 2.0 * PI / 360.0;
pub const LEAP_BARREL_DISTORTION: f32 = 0.15;
pub const LEAP_BARREL_DISTORTION_MAX: f32 = 1.0;
pub const LEAP_BARREL_DISTORTION_MIN: f32 = 0.01;
pub const LEAP_Y_OFFSET_MAX: i32 = 20;
pub const LEAP_Y_OFFSET_MIN: i32 = -LEAP_Y_OFFSET_MAX;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LensModelParamsData {
    pub y_aperture: f32,
    pub barrel_distortion: f32,
    pub y_left_offset: i32,
    pub y_right_offset: i32,
}

impl Default for LensModelParamsData {
    fn default() -> Self {
        Self {
            y_aperture: LEAP_Y_APERTURE,
            barrel_distortion: LEAP_BARREL_DISTORTION,
            y_left_offset: 0,
            y_right_offset: 0,
        }
    }
}

impl LensModelParamsData {
    pub fn fix(self) -> Self {
        Self {
            y_aperture: self
                .y_aperture
                .min(LEAP_Y_APERTURE_MAX)
                .max(LEAP_Y_APERTURE_MIN),
            barrel_distortion: self
                .barrel_distortion
                .min(LEAP_BARREL_DISTORTION_MAX)
                .max(LEAP_BARREL_DISTORTION_MIN),
            y_left_offset: self
                .y_left_offset
                .min(LEAP_Y_OFFSET_MAX)
                .max(LEAP_Y_OFFSET_MIN),
            y_right_offset: self
                .y_right_offset
                .min(LEAP_Y_OFFSET_MAX)
                .max(LEAP_Y_OFFSET_MIN),
        }
    }

    pub fn with_y_aperture(self, y_aperture: f32) -> Self {
        Self { y_aperture, ..self }.fix()
    }

    pub fn with_barrel_distortion(self, barrel_distortion: f32) -> Self {
        Self {
            barrel_distortion,
            ..self
        }
        .fix()
    }

    pub fn with_y_left_offset(self, y_left_offset: i32) -> Self {
        Self {
            y_left_offset,
            ..self
        }
        .fix()
    }

    pub fn with_y_right_offset(self, y_right_offset: i32) -> Self {
        Self {
            y_right_offset,
            ..self
        }
        .fix()
    }
}

#[derive(Debug, Clone)]
pub struct LensModelParams {
    pub x_frame_resolution: usize,
    pub y_frame_resolution: usize,
    pub y_left_offset: i32,
    pub y_right_offset: i32,
    pub x_half_frame_resolution: f32,
    pub y_half_frame_resolution: f32,
    pub aspect_ratio: f32,
    pub x_aperture: f32,
    pub y_aperture: f32,
    pub x_unit_arc: f32,
    pub y_unit_arc: f32,
    pub barrel_distortion: f32,
}

impl LensModelParams {
    fn compute_x_aperture(y_aperture: f32, aspect_ratio: f32) -> f32 {
        y_aperture * aspect_ratio
    }

    pub fn new(mode: LeapMode, data: LensModelParamsData) -> Self {
        let x_frame_resolution = mode.x_res() as usize;
        let y_frame_resolution = mode.y_res() as usize;
        let aspect_ratio = if x_frame_resolution == 752 {
            LEAP_ASPECT_RATIO_LARGE
        } else {
            LEAP_ASPECT_RATIO_SMALL
        };
        let x_aperture = Self::compute_x_aperture(LEAP_Y_APERTURE, aspect_ratio);
        let x_unit_arc = x_aperture / (x_frame_resolution as f32);
        let y_unit_arc = data.y_aperture / (y_frame_resolution as f32);

        Self {
            x_frame_resolution,
            y_frame_resolution,
            y_left_offset: data.y_left_offset,
            y_right_offset: data.y_right_offset,
            x_half_frame_resolution: (x_frame_resolution / 2) as f32,
            y_half_frame_resolution: (y_frame_resolution / 2) as f32,
            aspect_ratio,
            x_aperture,
            y_aperture: data.y_aperture,
            x_unit_arc,
            y_unit_arc,
            barrel_distortion: data.barrel_distortion,
        }
    }

    pub fn data(&self) -> LensModelParamsData {
        LensModelParamsData {
            y_aperture: self.y_aperture,
            barrel_distortion: self.barrel_distortion,
            y_left_offset: self.y_left_offset,
            y_right_offset: self.y_right_offset,
        }
    }

    pub fn frame_to_point(&self, x: u32, y: u32) -> Point {
        let x = ((x as f32) - self.x_half_frame_resolution) * self.x_unit_arc;
        let y = (-(y as f32) + self.y_half_frame_resolution) * self.y_unit_arc;
        Point::new(x + (self.x_unit_arc / 2.0), y + (self.y_unit_arc / 2.0))
    }

    pub fn point_to_frame(&self, p: &Point) -> (u32, u32) {
        let x = p.x - (self.x_unit_arc / 2.0);
        let y = p.y - (self.y_unit_arc / 2.0);
        let x = x + self.x_half_frame_resolution;
        let y = -y + self.y_half_frame_resolution;
        (x as u32, y as u32)
    }

    pub fn apply_barrel_distortion(&self, p: Point) -> Point {
        PolarPoint::from(p)
            .with_barrel_distortion(self.barrel_distortion)
            .into()
    }

    pub fn remove_barrel_distortion(&self, p: Point) -> Point {
        PolarPoint::from(p)
            .without_barrel_distortion(self.barrel_distortion)
            .into()
    }

    pub fn line_frame_size(&self) -> usize {
        self.x_frame_resolution * 2
    }

    pub fn get_left_pixel(&self, frame: &[u8], x: u32, y: u32) -> u8 {
        frame[(y as usize * self.line_frame_size()) + (x as usize) + 0]
    }

    pub fn get_right_pixel(&self, frame: &[u8], x: u32, y: u32) -> u8 {
        frame[(y as usize * self.line_frame_size()) + (x as usize) + 1]
    }

    fn get_line_at_channel(&self, frame: &[u8], line: &mut [u8], y: u32, channel: usize) {
        let line_start = y as usize * self.line_frame_size();
        let line_end = line_start + self.line_frame_size();
        let frame_line = &frame[line_start..line_end];
        frame_line
            .chunks_exact(2)
            .zip(line.iter_mut())
            .for_each(|(channels, line_pixel)| {
                *line_pixel = channels[channel];
            });
    }

    pub fn get_left_line(&self, frame: &[u8], line: &mut [u8], y: u32) {
        self.get_line_at_channel(frame, line, y, 0)
    }

    pub fn get_right_line(&self, frame: &[u8], line: &mut [u8], y: u32) {
        self.get_line_at_channel(frame, line, y, 1)
    }

    fn compute_elevation_jump(&self, x: usize, y: usize, y_level: f32) -> Option<usize> {
        let y_unsafe_area = self.y_left_offset.abs().max(self.y_right_offset.abs()) + 1;
        let mut jump = 0;
        let mut y = y as i32;
        let mut p = self.remove_barrel_distortion(self.frame_to_point(x as u32, y as u32));
        while y_level > (p.y + self.y_unit_arc) {
            if y < y_unsafe_area {
                return None;
            } else {
                y -= 1;
                jump += 1;
                p = self.remove_barrel_distortion(self.frame_to_point(x as u32, y as u32));
            }
        }
        Some(jump)
    }

    pub fn build_line_descriptor(&self, index_from_center: usize) -> Option<LineDescriptor> {
        let lines_count = self.y_frame_resolution / 2;
        let half_line_length = self.x_frame_resolution / 2;

        if index_from_center >= lines_count {
            return None;
        }

        let mut y_in_frame = lines_count - (index_from_center + 1);
        let start_with_barrel = self.frame_to_point(0, y_in_frame as u32);
        let start_without_barrel = self.remove_barrel_distortion(start_with_barrel);
        let y_level = start_without_barrel.y;

        let mut chunk_size = 0;
        let mut chunk_lenghts = Vec::new();

        for x_in_frame in 0..half_line_length {
            match self.compute_elevation_jump(x_in_frame, y_in_frame, y_level) {
                Some(mut elevation) => {
                    if elevation > 0 {
                        y_in_frame -= elevation;
                        chunk_lenghts.push(chunk_size as u16);
                        chunk_size = 0;
                        while elevation > 1 {
                            chunk_lenghts.push(0);
                            elevation -= 1;
                        }
                    }
                    chunk_size += 1;
                }
                None => {
                    return None;
                }
            }
        }

        if chunk_size > 0 {
            chunk_lenghts.push(chunk_size as u16);
        }

        Some(LineDescriptor {
            index_from_center,
            chunk_lenghts,
        })
    }
}

#[derive(Debug, Clone)]
pub struct LineDescriptor {
    pub index_from_center: usize,
    pub chunk_lenghts: Vec<u16>,
}

impl LineDescriptor {
    fn read_line_with_offsets(
        &self,
        frame: &[u8],
        line: &mut [u8],
        line_frame_size: usize,
        start_y: i32,
        line_jump_offset: i32,
        channel_offset: usize,
    ) {
        let mut frame_index = start_y * (line_frame_size as i32);
        let mut line_index = 0;

        for chunk_length in self.chunk_lenghts.iter().map(|l| *l as i32) {
            if chunk_length > 0 {
                let next_frame_index = frame_index + (chunk_length * 2);
                let next_line_index = line_index + (chunk_length as usize);

                let frame_chunk = &frame[(frame_index as usize)..(next_frame_index as usize)];
                let line_chunk = &mut line[line_index..next_line_index];
                frame_chunk
                    .chunks_exact(2)
                    .zip(line_chunk.iter_mut())
                    .for_each(|(channels, line_pixel)| {
                        *line_pixel = channels[channel_offset];
                    });
                frame_index = next_frame_index;
                line_index = next_line_index;
            }
            frame_index -= line_jump_offset;
        }

        frame_index += line_jump_offset;

        for chunk_length in self.chunk_lenghts.iter().rev().map(|l| *l as i32) {
            if chunk_length > 0 {
                let next_frame_index = frame_index + (chunk_length * 2);
                let next_line_index = line_index + (chunk_length as usize);
                let frame_chunk = &frame[(frame_index as usize)..(next_frame_index as usize)];
                let line_chunk = &mut line[line_index..next_line_index];
                frame_chunk
                    .chunks_exact(2)
                    .zip(line_chunk.iter_mut())
                    .for_each(|(channels, line_pixel)| {
                        *line_pixel = channels[channel_offset];
                    });
                frame_index = next_frame_index;
                line_index = next_line_index;
            }
            frame_index += line_jump_offset;
        }
    }

    fn read_upper_line(
        &self,
        frame: &[u8],
        line: &mut [u8],
        params: &LensModelParams,
        channel_offset: usize,
        y_offset: i32,
    ) {
        let line_frame_size = params.line_frame_size();
        let start_y =
            (params.y_frame_resolution as i32 / 2) - (self.index_from_center as i32 + 1) + y_offset;
        self.read_line_with_offsets(
            frame,
            line,
            line_frame_size,
            start_y,
            line_frame_size as i32,
            channel_offset,
        )
    }

    fn read_lower_line(
        &self,
        frame: &[u8],
        line: &mut [u8],
        params: &LensModelParams,
        channel_offset: usize,
        y_offset: i32,
    ) {
        let line_frame_size = params.line_frame_size();
        let start_y =
            (params.y_frame_resolution as i32 / 2) + (self.index_from_center as i32) + y_offset;
        self.read_line_with_offsets(
            frame,
            line,
            line_frame_size,
            start_y,
            -(line_frame_size as i32),
            channel_offset,
        )
    }

    pub fn read_upper_left_line(&self, frame: &[u8], line: &mut [u8], params: &LensModelParams) {
        self.read_upper_line(frame, line, params, 0, params.y_left_offset)
    }

    pub fn read_lower_left_line(&self, frame: &[u8], line: &mut [u8], params: &LensModelParams) {
        self.read_lower_line(frame, line, params, 0, params.y_left_offset)
    }

    pub fn read_upper_right_line(&self, frame: &[u8], line: &mut [u8], params: &LensModelParams) {
        self.read_upper_line(frame, line, params, 1, params.y_right_offset)
    }

    pub fn read_lower_right_line(&self, frame: &[u8], line: &mut [u8], params: &LensModelParams) {
        self.read_lower_line(frame, line, params, 1, params.y_right_offset)
    }
}

pub struct LineReader {
    params: LensModelParams,
    lines: Vec<LineDescriptor>,
}

impl LineReader {
    pub fn new(params: LensModelParams) -> Self {
        let mut lines = Vec::new();

        for index_from_center in 0..(params.y_frame_resolution / 2) {
            if let Some(line_descriptor) = params.build_line_descriptor(index_from_center) {
                lines.push(line_descriptor);
            } else {
                break;
            }
        }
        let lines = lines.into_iter().collect();
        Self { params, lines }
    }

    pub fn params(&self) -> &LensModelParams {
        &self.params
    }

    pub fn get_left_pixel_direct(&self, frame: &[u8], x: u32, y: u32) -> u8 {
        self.params.get_left_pixel(frame, x, y)
    }

    pub fn get_right_pixel_direct(&self, frame: &[u8], x: u32, y: u32) -> u8 {
        self.params.get_right_pixel(frame, x, y)
    }

    pub fn get_left_line_direct(&self, frame: &[u8], line: &mut [u8], y: u32) {
        self.params.get_left_line(frame, line, y)
    }

    pub fn get_right_line_direct(&self, frame: &[u8], line: &mut [u8], y: u32) {
        self.params.get_right_line(frame, line, y)
    }

    fn center_index(&self) -> usize {
        self.params.y_frame_resolution / 2
    }

    pub fn corrected_y_min(&self) -> usize {
        self.center_index() - self.lines.len()
    }

    pub fn corrected_y_max(&self) -> usize {
        self.center_index() + self.lines.len() - 1
    }

    fn get_line_descriptor(&self, y: u32) -> Option<(&LineDescriptor, bool)> {
        let y = y as usize;
        let center_index = self.center_index();
        if y > self.params.y_frame_resolution {
            None
        } else {
            let (index, upper) = if y >= center_index {
                (y - center_index, false)
            } else {
                (center_index - (y + 1), true)
            };
            self.lines.get(index).map(|d| (d, upper))
        }
    }

    pub fn get_left_line_corrected(&self, frame: &[u8], line: &mut [u8], y: u32) {
        match self.get_line_descriptor(y) {
            Some((descriptor, upper)) => {
                if upper {
                    descriptor.read_upper_left_line(frame, line, self.params());
                } else {
                    descriptor.read_lower_left_line(frame, line, self.params());
                }
            }
            None => {
                line.fill(0);
            }
        }
    }

    pub fn get_right_line_corrected(&self, frame: &[u8], line: &mut [u8], y: u32) {
        match self.get_line_descriptor(y) {
            Some((descriptor, upper)) => {
                if upper {
                    descriptor.read_upper_right_line(frame, line, self.params());
                } else {
                    descriptor.read_lower_right_line(frame, line, self.params());
                }
            }
            None => {
                line.fill(0);
            }
        }
    }
}

pub type ProfileValue = i16;
pub type Profile = Vector4<ProfileValue>;
pub const PROFILE_D1_THRESHOLD: i16 = 7;
pub const PROFILE_D3_THRESHOLD: i16 = 7;
pub const PROFILE_MATCHING_THRESHOLD: i16 = 20;
pub const LINE_FEATURES_MAX: usize = 40;

pub const LINE_SCANNER_WINDOW_MAX: usize = 19;
pub const LINE_SCANNER_WINDOW_MIN: usize = 3;
pub const LINE_SCANNER_WINDOW_DEFAULT: usize = 7;
pub const LINE_NOISE_REDUCTION_WINDOW_MAX: usize = 19;
pub const LINE_NOISE_REDUCTION_WINDOW_DEFAULT: usize = 5;

//pub const LINE_SCANNER_WINDOW_CENTER: usize = LINE_SCANNER_WINDOW / 2;
//pub const LINE_SCANNER_WINDOW_POST_START: usize = LINE_SCANNER_WINDOW + 3;
//pub const LINE_SCANNER_WINDOW_CENTER_TO_LAST: usize = LINE_SCANNER_WINDOW_CENTER - 1;

pub const LINE_SCANNER_VALUE_FACTOR: ProfileValue = 8;

#[derive(Debug, Clone)]
pub struct LineScannerParameters {
    pub d1_threshold: i16,
    pub d3_threshold: i16,
    pub matching_threshold: i16,
    pub features_max: usize,
    pub line_scanner_window: usize,
    pub line_noise_reduction_window: usize,
}

impl LineScannerParameters {
    pub fn fix(&mut self) {
        if self.d1_threshold < 0 {
            self.d1_threshold = 0;
        }
        if self.d3_threshold < 0 {
            self.d3_threshold = 0;
        }
        if self.matching_threshold < 0 {
            self.matching_threshold = 0;
        }

        if self.line_scanner_window % 2 == 0 {
            self.line_scanner_window += 1;
        }
        if self.line_scanner_window > LINE_SCANNER_WINDOW_MAX {
            self.line_scanner_window = LINE_SCANNER_WINDOW_MAX;
        } else if self.line_scanner_window < LINE_SCANNER_WINDOW_MIN {
            self.line_scanner_window = LINE_SCANNER_WINDOW_MIN;
        }

        if self.line_noise_reduction_window % 2 == 0 {
            self.line_noise_reduction_window += 1;
        }
        if self.line_noise_reduction_window > LINE_NOISE_REDUCTION_WINDOW_MAX {
            self.line_noise_reduction_window = LINE_NOISE_REDUCTION_WINDOW_MAX;
        }
    }

    pub fn line_scanner_window_center(&self) -> usize {
        self.line_scanner_window / 2
    }
    pub fn line_scanner_window_post_start(&self) -> usize {
        self.line_scanner_window + 3
    }
    pub fn line_scanner_window_center_to_last(&self) -> usize {
        self.line_scanner_window_center() + 1
    }
}

impl Default for LineScannerParameters {
    fn default() -> Self {
        Self {
            d1_threshold: PROFILE_D1_THRESHOLD,
            d3_threshold: PROFILE_D3_THRESHOLD,
            matching_threshold: PROFILE_MATCHING_THRESHOLD,
            features_max: LINE_FEATURES_MAX,
            line_scanner_window: LINE_SCANNER_WINDOW_DEFAULT,
            line_noise_reduction_window: LINE_NOISE_REDUCTION_WINDOW_DEFAULT,
        }
    }
}

#[inline]
pub fn profiles_match(p1: Profile, p2: Profile) -> bool {
    let diff = p1 - p2;
    let prod = diff.dot(&diff);
    prod < PROFILE_MATCHING_THRESHOLD
}

#[derive(Debug, Clone, Copy)]
pub enum LineFeatureKind {
    Max,
    Min,
    SlopeUp,
    SlopeDown,
}

#[derive(Debug, Clone, Copy)]
pub struct LineFeature {
    pub profile: Profile,
    pub kind: LineFeatureKind,
    pub position: u16,
    pub matched_position: Option<u16>,
}

impl LineFeature {
    pub fn new(kind: LineFeatureKind, position: usize, profile: Profile) -> Self {
        Self {
            profile,
            kind,
            position: position as u16,
            matched_position: None,
        }
    }

    pub fn value(&self) -> ProfileValue {
        self.profile[0]
    }

    pub fn left_d1(&self) -> ProfileValue {
        self.profile[1]
    }

    pub fn right_d1(&self) -> ProfileValue {
        self.profile[2]
    }

    pub fn d2(&self) -> ProfileValue {
        self.profile[3]
    }
}

pub fn apply_noise_reduction<'a>(
    line: impl Iterator<Item = (usize, u8)> + 'a,
    params: &LineScannerParameters,
) -> impl Iterator<Item = (usize, u8)> + 'a {
    let window_size = params.line_noise_reduction_window;
    let mut line = line;
    let mut window = [0u8; LINE_NOISE_REDUCTION_WINDOW_MAX];

    window
        .iter_mut()
        .take(window_size - 1)
        .for_each(|cell| *cell = line.next().unwrap().1);

    let position_offset = window_size / 2;
    let mut next = window_size - 1;
    let mut sum: u32 = window[0..window_size - 1].iter().map(|v| *v as u32).sum();

    line.map(move |(index, value)| {
        sum -= window[next] as u32;
        window[next] = value;
        sum += value as u32;

        next += 1;
        if next >= window_size {
            next = 0;
        }

        (index - position_offset, (sum / window_size as u32) as u8)
    })
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LineScannerWindowPoint {
    pub val: ProfileValue,
    pub d1: ProfileValue,
    pub d2: ProfileValue,
    pub pos: u16,
}

impl LineScannerWindowPoint {
    pub fn new(val: ProfileValue, d1: ProfileValue, d2: ProfileValue, pos: u16) -> Self {
        Self { val, d1, d2, pos }
    }
}

pub struct LineScanner {
    window: [LineScannerWindowPoint; LINE_SCANNER_WINDOW_MAX],
    window_size: usize,

    first: usize,
    center: usize,
    last: usize,
}

impl LineScanner {
    fn init<'a, L>(line: L, params: &LineScannerParameters) -> (Self, L)
    where
        L: Iterator<Item = (usize, u8)> + 'a,
    {
        let window_size = params.line_scanner_window;
        let mut line = line;
        let mut window = [LineScannerWindowPoint::default(); LINE_SCANNER_WINDOW_MAX];

        let previous1_val = line.next().unwrap().1 as ProfileValue;
        let (previous0_pos, previous0_val) = line.next().unwrap();
        let previous0_val = previous0_val as ProfileValue;
        let previous0_d1 = previous0_val - previous1_val;
        let previous0_d2 = 0;
        let mut previous_point = LineScannerWindowPoint::new(
            previous0_val,
            previous0_d1,
            previous0_d2,
            previous0_pos as u16,
        );

        window.iter_mut().take(window_size - 1).for_each(|cell| {
            let (pos, val) = line.next().unwrap();
            let val = val as ProfileValue;
            let d1 = val - previous_point.val;
            let d2 = d1 - previous_point.d1;
            let point = LineScannerWindowPoint::new(val, d1, d2, pos as u16);
            *cell = point;
            previous_point = point;
        });

        let first = window_size - 1;
        let last = window_size - 2;
        let center = (window_size / 2) - 1;

        (
            Self {
                window,
                window_size,
                first,
                center,
                last,
            },
            line,
        )
    }

    fn next_in_window(&self, index: usize) -> usize {
        let result = index + 1;
        if result < self.window_size {
            result
        } else {
            0
        }
    }

    pub fn update(&mut self, pos: usize, val: u8) {
        let val = val as ProfileValue;

        let last_point = &self.window[self.last];
        let d1 = val - last_point.val;
        let d2 = d1 - last_point.d1;
        let next_point = LineScannerWindowPoint::new(val, d1, d2, pos as u16);

        self.window[self.first] = next_point;

        self.last = self.first;
        self.first = self.next_in_window(self.first);
        self.center = self.next_in_window(self.center);
    }

    fn get_profile(&self) -> Vector4<ProfileValue> {
        Vector4::new(
            self.window[self.center].val,
            self.window[self.center].val - self.window[self.first].val,
            self.window[self.last].val - self.window[self.center].val,
            self.window[self.last].d1 - self.window[self.first].d1,
        ) / LINE_SCANNER_VALUE_FACTOR
    }

    fn d1_delta(&self) -> ProfileValue {
        self.window[self.last].d1 - self.window[self.first].d1
    }

    pub fn get_feature(&self, params: &LineScannerParameters) -> Option<LineFeature> {
        let c1 = self.center;
        let c2 = self.next_in_window(self.center);
        let position = self.window[c1].pos as usize;

        if self.window[c1].d1 >= 0
            && self.window[c2].d1 <= 0
            && self.d1_delta() <= -params.d1_threshold
            && self.window.iter().map(|p| p.d2).all(|d2| d2 <= 0)
        {
            // Max detected
            Some(LineFeature::new(
                LineFeatureKind::Max,
                position,
                self.get_profile(),
            ))
        } else if self.window[c1].d1 <= 0
            && self.window[c2].d1 >= 0
            && self.d1_delta() >= params.d1_threshold
            && self.window.iter().map(|p| p.d2).all(|d2| d2 >= 0)
        {
            // Min detected
            Some(LineFeature::new(
                LineFeatureKind::Min,
                position,
                self.get_profile(),
            ))
        } else {
            None
        }
    }

    pub fn scan<'a, L>(line: L, params: &LineScannerParameters) -> Vec<LineFeature>
    where
        L: Iterator<Item = (usize, u8)> + 'a,
    {
        let (mut window, line) = Self::init(apply_noise_reduction(line, params), params);
        let mut features = Vec::with_capacity(params.features_max);

        line.for_each(|(pos, val)| {
            window.update(pos, val);
            if features.len() < features.capacity() {
                if let Some(feature) = window.get_feature(params) {
                    features.push(feature);
                }
            }
        });

        features
    }

    pub fn point_values<'a, L>(
        line: L,
        params: &LineScannerParameters,
    ) -> impl Iterator<Item = LineScannerWindowPoint> + 'a
    where
        L: Iterator<Item = (usize, u8)> + 'a,
    {
        let (mut window, line) = Self::init(apply_noise_reduction(line, params), params);

        line.map(move |(pos, val)| {
            window.update(pos, val);
            window.window[window.center]
        })
    }
}
