use std::iter::repeat;

pub const MIN_BLUR_SIZE: usize = 1;
pub const MAX_BLUR_SIZE_VALUE: usize = 8;
pub const MAX_BLUR_SIZE_DERIVATIVE: usize = 4;
pub const MIN_DERIVATIVE_SCALING: i16 = 1;
pub const MAX_DERIVATIVE_SCALING: i16 = 64;

#[derive(Debug, Clone)]
pub struct LineProcessingParameters {
    pub blur_size: usize,
    pub blur_size_d1: usize,
    pub blur_size_d2: usize,
    pub derivative_scaling_d1: i16,
    pub derivative_scaling_d2: i16,
}

impl Default for LineProcessingParameters {
    fn default() -> Self {
        Self {
            blur_size: MAX_BLUR_SIZE_VALUE,
            blur_size_d1: MAX_BLUR_SIZE_DERIVATIVE,
            blur_size_d2: MAX_BLUR_SIZE_DERIVATIVE,
            derivative_scaling_d1: 16,
            derivative_scaling_d2: 4,
        }
    }
}

impl LineProcessingParameters {
    pub fn fix(&mut self) {
        if self.blur_size < MIN_BLUR_SIZE {
            self.blur_size = MIN_BLUR_SIZE;
        }
        if self.blur_size_d1 < MIN_BLUR_SIZE {
            self.blur_size_d1 = MIN_BLUR_SIZE;
        }
        if self.blur_size_d1 < MIN_BLUR_SIZE {
            self.blur_size_d2 = MIN_BLUR_SIZE;
        }

        if self.blur_size > MAX_BLUR_SIZE_VALUE {
            self.blur_size = MAX_BLUR_SIZE_VALUE;
        }
        if self.blur_size_d1 > MAX_BLUR_SIZE_DERIVATIVE {
            self.blur_size_d1 = MAX_BLUR_SIZE_DERIVATIVE;
        }
        if self.blur_size_d1 > MAX_BLUR_SIZE_DERIVATIVE {
            self.blur_size_d2 = MAX_BLUR_SIZE_DERIVATIVE;
        }
    }
}

pub struct BlurHalfWindowU8<const SIZE: usize> {
    buffer: u64,
}

impl<const SIZE: usize> BlurHalfWindowU8<SIZE> {
    pub fn new(fill: u8) -> Self {
        let v = fill as u64;
        let buffer: u64 =
            v | (v << 8) | (v << 16) | (v << 24) | (v << 32) | (v << 40) | (v << 48) | (v << 56);
        Self { buffer }
    }

    pub fn slot(&self, index: usize) -> u8 {
        (self.buffer >> (index * 8)) as u8
    }

    pub fn output_slot(&self) -> u8 {
        (self.buffer >> ((SIZE - 1) * 8)) as u8
    }

    pub fn tick_n(&mut self, input: u8, current_sum: i32, size: usize) -> (u8, i32) {
        let output = self.slot(size - 1);
        self.buffer <<= 8;
        self.buffer += input as u64;
        (output, (current_sum + (input as i32)) - (output as i32))
    }

    pub fn tick(&mut self, input: u8, current_sum: i32) -> (u8, i32) {
        let output = self.output_slot();
        self.buffer <<= 8;
        self.buffer += input as u64;
        (output, (current_sum + (input as i32)) - (output as i32))
    }
}

pub struct BlurHalfWindowI16<const SIZE: usize> {
    buffer: u64,
}

impl<const SIZE: usize> BlurHalfWindowI16<SIZE> {
    pub fn new() -> Self {
        Self { buffer: 0 }
    }

    pub fn slot(&self, index: usize) -> i16 {
        (self.buffer >> (index * 16)) as i16
    }

    pub fn output_slot(&self) -> i16 {
        (self.buffer >> ((SIZE - 1) * 16)) as i16
    }

    pub fn tick_n(&mut self, input: i16, current_sum: i32, size: usize) -> (i16, i32) {
        let output = self.slot(size - 1);
        self.buffer <<= 16;
        self.buffer += input as u16 as u64;
        (output, (current_sum + (input as i32)) - (output as i32))
    }

    pub fn tick(&mut self, input: i16, current_sum: i32) -> (i16, i32) {
        let output = self.output_slot();
        self.buffer <<= 16;
        self.buffer += input as u16 as u64;
        (output, (current_sum + (input as i32)) - (output as i32))
    }
}

#[test]
fn test_tick_n() {
    let mut w = BlurHalfWindowU8::<8>::new(0);
    let (mut out, mut sum) = w.tick_n(1, 0, 4);
    assert_eq!(out, 0);
    assert_eq!(sum, 1);
    (out, sum) = w.tick_n(2, sum, 4);
    assert_eq!(out, 0);
    assert_eq!(sum, 3);
    (out, sum) = w.tick_n(3, sum, 4);
    assert_eq!(out, 0);
    assert_eq!(sum, 6);
    (out, sum) = w.tick_n(4, sum, 4);
    assert_eq!(out, 0);
    assert_eq!(sum, 10);
    (out, sum) = w.tick_n(0, sum, 4);
    assert_eq!(out, 1);
    assert_eq!(sum, 9);
    (out, sum) = w.tick_n(0, sum, 4);
    assert_eq!(out, 2);
    assert_eq!(sum, 7);
    (out, sum) = w.tick_n(0, sum, 4);
    assert_eq!(out, 3);
    assert_eq!(sum, 4);
    (out, sum) = w.tick_n(0, sum, 4);
    assert_eq!(out, 4);
    assert_eq!(sum, 0);
}

pub fn blur_n<'a>(input: &'a [u8], blur_size: usize) -> impl Iterator<Item = u8> + 'a {
    let fill = input[0];
    let tail = input[input.len() - 1];
    let mut pre_window = BlurHalfWindowU8::<MAX_BLUR_SIZE_VALUE>::new(fill);
    let mut pre_sum: i32 = (fill as i32) * (blur_size as i32);
    let mut post_window = BlurHalfWindowU8::<MAX_BLUR_SIZE_VALUE>::new(fill);
    let mut post_sum: i32 = (fill as i32) * (blur_size as i32);
    let mut current: u8 = fill;
    let factor: i32 = (blur_size as i32 * 2) + 1;

    input.iter().copied().chain(repeat(tail)).map(move |value| {
        (_, post_sum) = post_window.tick_n(current, post_sum, blur_size);
        (current, pre_sum) = pre_window.tick_n(value, pre_sum, blur_size);
        let current_blurred = current as i32 + pre_sum + post_sum;
        (current_blurred / factor) as u8
    })
}

pub fn blur<'a, const SIZE: usize>(input: &'a [u8]) -> impl Iterator<Item = u8> + 'a {
    let fill = input[0];
    let tail = input[input.len() - 1];
    let mut pre_window = BlurHalfWindowU8::<SIZE>::new(fill);
    let mut pre_sum: i32 = (fill as i32) * (SIZE as i32);
    let mut post_window = BlurHalfWindowU8::<SIZE>::new(fill);
    let mut post_sum: i32 = (fill as i32) * (SIZE as i32);
    let mut current: u8 = fill;
    let factor: i32 = (SIZE as i32 * 2) + 1;

    input.iter().copied().chain(repeat(tail)).map(move |value| {
        (_, post_sum) = post_window.tick(current, post_sum);
        (current, pre_sum) = pre_window.tick(value, pre_sum);
        let current_blurred = current as i32 + pre_sum + post_sum;
        (current_blurred / factor) as u8
    })
}

pub fn blur_n_buffer(input: &[u8], output: &mut [u8], blur_size: usize) {
    blur_n(input, blur_size)
        .skip(blur_size)
        .take(input.len())
        .enumerate()
        .for_each(|(i, v)| output[i] = v);
}

#[test]
fn test_blur_buffer_flat() {
    let input: [u8; 9] = [9, 9, 9, 9, 9, 9, 9, 9, 9];
    let mut output = [0u8; 9];
    let expected: [u8; 9] = [9, 9, 9, 9, 9, 9, 9, 9, 9];
    blur_n_buffer(&input, &mut output, 4);
    assert_eq!(output.to_vec(), expected.to_vec());
}

#[test]
fn test_blur_buffer_spike() {
    let input: [u8; 9] = [9, 9, 9, 9, 18, 9, 9, 9, 9];
    let mut output = [0u8; 9];
    let expected: [u8; 9] = [10, 10, 10, 10, 10, 10, 10, 10, 10];
    blur_n_buffer(&input, &mut output, 4);
    assert_eq!(output.to_vec(), expected.to_vec());
}

pub fn derivative_u8_n(
    input: impl Iterator<Item = u8>,
    blur_size: usize,
    derivative_scaling: i16,
) -> impl Iterator<Item = i16> {
    let mut previous_value: i16 = 0;
    let mut pre_window = BlurHalfWindowI16::<MAX_BLUR_SIZE_DERIVATIVE>::new();
    let mut pre_sum: i32 = 0;
    let mut post_window = BlurHalfWindowI16::<MAX_BLUR_SIZE_DERIVATIVE>::new();
    let mut post_sum: i32 = 0;
    let mut current: i16 = 0;
    let factor: i32 = (blur_size as i32 * 2) + 1;

    input.chain(repeat(0)).map(move |value| {
        let value = value as i16 * derivative_scaling;
        let current_d1 = value - previous_value;
        previous_value = value;
        (_, post_sum) = post_window.tick_n(current, post_sum, blur_size);
        (current, pre_sum) = pre_window.tick_n(current_d1 as i16, pre_sum, blur_size);
        let current_blurred = current as i32 + pre_sum + post_sum;
        (current_blurred / factor) as i16
    })
}

pub fn derivative_i16_n(
    input: impl Iterator<Item = i16>,
    blur_size: usize,
    derivative_scaling: i16,
) -> impl Iterator<Item = i16> {
    let mut previous_value: i16 = 0;
    let mut pre_window = BlurHalfWindowI16::<MAX_BLUR_SIZE_DERIVATIVE>::new();
    let mut pre_sum: i32 = 0;
    let mut post_window = BlurHalfWindowI16::<MAX_BLUR_SIZE_DERIVATIVE>::new();
    let mut post_sum: i32 = 0;
    let mut current: i16 = 0;
    let factor: i32 = (blur_size as i32 * 2) + 1;

    input.chain(repeat(0)).map(move |value| {
        let value = value * derivative_scaling;
        let current_d1 = value - previous_value;
        previous_value = value;
        (_, post_sum) = post_window.tick_n(current, post_sum, blur_size);
        (current, pre_sum) = pre_window.tick_n(current_d1 as i16, pre_sum, blur_size);
        let current_blurred = current as i32 + pre_sum + post_sum;
        (current_blurred / factor) as i16
    })
}

pub fn derivative_1_buffer(input: &[u8], output: &mut [i16], params: &LineProcessingParameters) {
    let blurred = blur_n(input, params.blur_size);
    let derivative_1 = derivative_u8_n(blurred, params.blur_size_d1, params.derivative_scaling_d1);
    derivative_1
        .skip(params.blur_size + params.blur_size_d1)
        .take(input.len())
        .enumerate()
        .for_each(|(i, v)| output[i] = v);
}

pub fn derivative_2_buffer(input: &[u8], output: &mut [i16], params: &LineProcessingParameters) {
    let blurred = blur_n(input, params.blur_size);
    let derivative_1 = derivative_u8_n(blurred, params.blur_size_d1, params.derivative_scaling_d1);
    let derivative_2 = derivative_i16_n(
        derivative_1,
        params.blur_size_d2,
        params.derivative_scaling_d2,
    );
    derivative_2
        .skip(params.blur_size + params.blur_size_d1 + params.blur_size_d2)
        .take(input.len())
        .enumerate()
        .for_each(|(i, v)| output[i] = v);
}

pub fn min_max<'a>(
    input: impl Iterator<Item = (usize, i16)> + 'a,
) -> impl Iterator<Item = (usize, i16)> + 'a {
    let mut increasing: bool = false;
    let mut decreasing: bool = false;
    let mut previous: i16 = 0;
    input.filter_map(move |(index, next)| {
        let current = previous;
        previous = next;

        if next < current {
            decreasing = true;
            if increasing {
                increasing = false;
                Some((index - 1, current))
            } else {
                None
            }
        } else if next > current {
            increasing = true;
            if decreasing {
                decreasing = false;
                Some((index - 1, current))
            } else {
                None
            }
        } else {
            increasing = false;
            decreasing = false;
            None
        }
    })
}

#[test]
fn test_min_max() {
    let input: Vec<i16> = vec![5, 9, 7, 6, 3, 0, -2, -5, 3];
    let expected: Vec<(usize, i16)> = vec![(1, 9), (7, -5)];
    let output: Vec<(usize, i16)> = min_max(input.iter().copied().enumerate()).collect();
    assert_eq!(expected, output);
}

#[derive(Clone, Copy)]
pub struct CurvePoint {
    pub index: usize,
    pub d2: i16,
}

impl CurvePoint {
    pub fn zero(index: usize) -> Self {
        Self { index, d2: 0 }
    }
}

pub fn get_curve_points_blur_n<'a>(
    input: &'a [u8],
    params: &LineProcessingParameters,
) -> impl Iterator<Item = CurvePoint> + 'a {
    let length = input.len();
    let skip = params.blur_size + params.blur_size_d1 + params.blur_size_d2 + 1;
    let blurred = blur_n(input, params.blur_size);
    let d1 = derivative_u8_n(blurred, params.blur_size_d1, params.derivative_scaling_d1);
    let d2 = derivative_i16_n(d1, params.blur_size_d2, params.derivative_scaling_d2);
    let indexed_d2 = d2.skip(skip).enumerate();
    let curve_points = min_max(indexed_d2.filter(move |(i, _)| *i < length).take(length));
    curve_points.map(|(index, d2)| CurvePoint { index, d2 })
}

/*
pub fn get_curve_points_blur<
    'a,
    const BLUR_SIZE: usize,
    const BLUR_SIZE_D1: usize,
    const BLUR_SIZE_D2: usize,
>(
    input: &'a [u8],
) -> impl Iterator<Item = CurvePoint> + 'a {
    let length = input.len();
    const SKIP: usize = (BLUR_SIZE + BLUR_SIZE_D1 + BLUR_SIZE_D2 + 1);
    let blurred = blur_n(input, blur_size);
    let d1 = derivative_u8_n(blurred, blur_size_d1);
    let d2 = derivative_i8_n(d1, blur_size_d2);
    let indexed_d2 = d2.skip(skip).enumerate();
    let curve_points = min_max(indexed_d2.filter(|(i, _)| *i >= 0).take(length));
    curve_points.map(|(index, d2)| CurvePoint {
        index,
        value: input[index],
        d2,
    })
}
*/

pub fn segment_buffer(input: &[u8], output: &mut [u8], params: &LineProcessingParameters) {
    let length = input.len();
    let mut curve_points = Vec::with_capacity(length + 2);
    curve_points.push(CurvePoint::zero(0));
    curve_points.extend(get_curve_points_blur_n(input, params));
    curve_points.push(CurvePoint::zero(length - 1));
    let mut points_iter = curve_points.into_iter();
    let mut current = points_iter.next().unwrap();
    for next in points_iter {
        let current_index = current.index as i32;
        let next_index = next.index as i32;
        let delta_index = next_index - current_index;
        let current_value = input[current.index] as i32;
        let next_value = input[next.index] as i32;
        let delta_value = next_value - current_value;
        for i in 0..delta_index {
            let index = (current_index + i) as usize;
            let value = current_value + (i * delta_value / delta_index);
            if index < output.len() {
                output[index] = value as u8;
            }
        }
        current = next;
    }
}

/*
#[test]
fn test_segment_buffer() {
    let input: Vec<u8> = vec![1, 2, 7, 6, 5, 0, 0, 3, 7];
    let expected: Vec<u8> = vec![1, 2, 7, 6, 5, 0, 0, 3, 7];
    let mut output = [0u8; 9];
    segment_buffer(&input, &mut output, 1, 1, 1);
    assert_eq!(expected, output);
}
*/
