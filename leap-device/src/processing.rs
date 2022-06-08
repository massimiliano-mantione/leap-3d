use std::iter::repeat;

pub const MIN_BLUR_SIZE: usize = 1;
pub const MAX_BLUR_SIZE: usize = 8;

#[derive(Debug, Clone)]
pub struct LineProcessingParameters {
    pub blur_size: usize,
    pub blur_size_d1: usize,
    pub blur_size_d2: usize,
}

impl Default for LineProcessingParameters {
    fn default() -> Self {
        Self {
            blur_size: MAX_BLUR_SIZE,
            blur_size_d1: MAX_BLUR_SIZE,
            blur_size_d2: MAX_BLUR_SIZE,
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

        if self.blur_size > MAX_BLUR_SIZE {
            self.blur_size = MAX_BLUR_SIZE;
        }
        if self.blur_size_d1 > MAX_BLUR_SIZE {
            self.blur_size_d1 = MAX_BLUR_SIZE;
        }
        if self.blur_size_d1 > MAX_BLUR_SIZE {
            self.blur_size_d2 = MAX_BLUR_SIZE;
        }
    }
}

pub struct BlurHalfWindow<const SIZE: usize> {
    buffer: u64,
}

impl<const SIZE: usize> BlurHalfWindow<SIZE> {
    pub fn new_u8(fill: u8) -> Self {
        let v = fill as u64;
        let buffer: u64 =
            v | (v << 8) | (v << 16) | (v << 24) | (v << 32) | (v << 40) | (v << 48) | (v << 56);
        Self { buffer }
    }

    pub fn new_i8(fill: i8) -> Self {
        Self::new_u8(fill as u8)
    }

    pub fn slot_u8(&self, index: usize) -> u8 {
        (self.buffer >> (index * 8)) as u8 & 0xffu8
    }
    pub fn slot_i8(&self, index: usize) -> i8 {
        self.slot_u8(index) as i8
    }

    pub fn output_slot_u8(&self) -> u8 {
        (self.buffer >> ((SIZE - 1) * 8)) as u8 & 0xffu8
    }
    pub fn output_slot_i8(&self) -> i8 {
        self.output_slot_u8() as i8
    }

    pub fn tick_u8_n(&mut self, input: u8, current_sum: i32, size: usize) -> (u8, i32) {
        let output = self.slot_u8(size - 1);
        self.buffer <<= 8;
        self.buffer += input as u64;
        (output, (current_sum + (input as i32)) - (output as i32))
    }
    pub fn tick_i8_n(&mut self, input: i8, current_sum: i32, size: usize) -> (i8, i32) {
        let output = self.slot_i8(size - 1);
        self.buffer <<= 8;
        self.buffer += input as u8 as u64;
        (output, (current_sum + (input as i32)) - (output as i32))
    }

    pub fn tick_u8(&mut self, input: u8, current_sum: i32) -> (u8, i32) {
        let output = self.output_slot_u8();
        self.buffer <<= 8;
        self.buffer += input as u64;
        (output, (current_sum + (input as i32)) - (output as i32))
    }
    pub fn tick_i8(&mut self, input: i8, current_sum: i32) -> (i8, i32) {
        let output = self.output_slot_i8();
        self.buffer <<= 8;
        self.buffer += input as u8 as u64;
        (output, (current_sum + (input as i32)) - (output as i32))
    }
}

#[test]
fn test_tick_n() {
    let mut w = BlurHalfWindow::<8>::new_u8(0);
    let (mut out, mut sum) = w.tick_u8_n(1, 0, 4);
    assert_eq!(out, 0);
    assert_eq!(sum, 1);
    (out, sum) = w.tick_u8_n(2, sum, 4);
    assert_eq!(out, 0);
    assert_eq!(sum, 3);
    (out, sum) = w.tick_u8_n(3, sum, 4);
    assert_eq!(out, 0);
    assert_eq!(sum, 6);
    (out, sum) = w.tick_u8_n(4, sum, 4);
    assert_eq!(out, 0);
    assert_eq!(sum, 10);
    (out, sum) = w.tick_u8_n(0, sum, 4);
    assert_eq!(out, 1);
    assert_eq!(sum, 9);
    (out, sum) = w.tick_u8_n(0, sum, 4);
    assert_eq!(out, 2);
    assert_eq!(sum, 7);
    (out, sum) = w.tick_u8_n(0, sum, 4);
    assert_eq!(out, 3);
    assert_eq!(sum, 4);
    (out, sum) = w.tick_u8_n(0, sum, 4);
    assert_eq!(out, 4);
    assert_eq!(sum, 0);
}

pub fn blur_n<'a>(input: &'a [u8], blur_size: usize) -> impl Iterator<Item = i32> + 'a {
    let fill = input[0];
    let mut pre_window = BlurHalfWindow::<8>::new_u8(fill);
    let mut pre_sum: i32 = (fill as i32) * (blur_size as i32);
    let mut post_window = BlurHalfWindow::<8>::new_u8(fill);
    let mut post_sum: i32 = (fill as i32) * (blur_size as i32);
    let mut current: u8 = fill;
    let factor: i32 = (blur_size as i32 * 2) + 1;

    input.iter().copied().chain(repeat(0)).map(move |value| {
        (_, post_sum) = post_window.tick_u8_n(current, post_sum, blur_size);
        (current, pre_sum) = pre_window.tick_u8_n(value, pre_sum, blur_size);
        let current_blurred = current as i32 + pre_sum + post_sum;
        current_blurred / factor
    })
}

pub fn blur<'a, const SIZE: usize>(input: &'a [u8]) -> impl Iterator<Item = i32> + 'a {
    let fill = input[0];
    let mut pre_window = BlurHalfWindow::<SIZE>::new_u8(fill);
    let mut pre_sum: i32 = (fill as i32) * (SIZE as i32);
    let mut post_window = BlurHalfWindow::<SIZE>::new_u8(fill);
    let mut post_sum: i32 = (fill as i32) * (SIZE as i32);
    let mut current: u8 = fill;
    let factor: i32 = (SIZE as i32 * 2) + 1;

    input.iter().copied().chain(repeat(0)).map(move |value| {
        (_, post_sum) = post_window.tick_u8(current, post_sum);
        (current, pre_sum) = pre_window.tick_u8(value, pre_sum);
        let current_blurred = current as i32 + pre_sum + post_sum;
        current_blurred / factor
    })
}

pub fn blur_n_buffer(input: &[u8], output: &mut [u8], blur_size: usize) {
    blur_n(input, blur_size)
        .skip(blur_size)
        .take(input.len())
        .enumerate()
        .for_each(|(i, v)| output[i] = v as u8);
}

#[test]
fn test_blur_buffer() {
    let input: [u8; 9] = [9, 9, 9, 9, 9, 9, 9, 9, 9];
    let mut output = [0u8; 9];
    let expected: [u8; 9] = [5, 6, 7, 8, 9, 8, 7, 6, 5];
    blur_n_buffer(&input, &mut output, 4);
    assert_eq!(output.to_vec(), expected.to_vec());
}

fn scale_to_small_i8(v: i32) -> i8 {
    (((v >> 1) & 0xf0) | (v & 0x0f)) as i8
}

pub fn derivative_1_n<'a>(input: &'a [u8], blur_size: usize) -> impl Iterator<Item = i8> + 'a {
    let mut previous_value = input[0] as i32;
    let mut pre_window = BlurHalfWindow::<8>::new_i8(0);
    let mut pre_sum: i32 = 0;
    let mut post_window = BlurHalfWindow::<8>::new_i8(0);
    let mut post_sum: i32 = 0;
    let mut current: i8 = 0;
    let factor: i32 = (blur_size as i32 * 2) + 1;

    input.iter().copied().chain(repeat(0)).map(move |value| {
        let value = value as i32;
        let current_d1 = value - previous_value;
        previous_value = value;
        (_, post_sum) = post_window.tick_i8_n(current, post_sum, blur_size);
        (current, pre_sum) =
            pre_window.tick_i8_n(scale_to_small_i8(current_d1), pre_sum, blur_size);
        let current_blurred = current as i32 + pre_sum + post_sum;
        (current_blurred / factor) as i8
    })
}

pub fn derivative_1_n_buffer(input: &[u8], output: &mut [i8], blur_size: usize) {
    derivative_1_n(input, blur_size)
        .skip(blur_size)
        .take(input.len())
        .enumerate()
        .for_each(|(i, v)| output[i] = v);
}

pub fn derivative_2_n<'a>(input: &'a [u8], blur_size: usize) -> impl Iterator<Item = i8> + 'a {
    let mut previous_value = input[0] as i32;
    let mut previous_d1 = 0i32;
    let mut pre_window = BlurHalfWindow::<8>::new_i8(0);
    let mut pre_sum: i32 = 0;
    let mut post_window = BlurHalfWindow::<8>::new_i8(0);
    let mut post_sum: i32 = 0;
    let mut current: i8 = 0;
    let factor: i32 = (blur_size as i32 * 2) + 1;

    input.iter().copied().chain(repeat(0)).map(move |value| {
        let value = value as i32;
        let current_d1 = value - previous_value;
        let current_d2 = previous_d1 - current_d1;
        previous_d1 = current_d1;
        previous_value = value;
        (_, post_sum) = post_window.tick_i8_n(current, post_sum, blur_size);
        (current, pre_sum) =
            pre_window.tick_i8_n(scale_to_small_i8(current_d2), pre_sum, blur_size);
        let current_blurred = current as i32 + pre_sum + post_sum;
        (current_blurred / factor) as i8
    })
}

pub fn derivative_2_n_buffer(input: &[u8], output: &mut [i8], blur_size: usize) {
    derivative_2_n(input, blur_size)
        .skip(blur_size)
        .take(input.len())
        .enumerate()
        .for_each(|(i, v)| output[i] = v);
}

pub fn derivative<'a>(
    input: impl Iterator<Item = (u8, i32)> + 'a,
) -> impl Iterator<Item = (u8, i32)> + 'a {
    let mut previous: i32 = 0;
    input.map(move |(original, value)| {
        let delta = value - previous;
        previous = value;
        (original, delta)
    })
}

#[test]
fn test_derivative() {
    let input: Vec<i32> = vec![5, 9, 7, 6, 3, 0, -2, -5, 3];
    let expected: Vec<i32> = vec![5, 4, -2, -1, -3, -3, -2, -3, 8];
    let output: Vec<i32> = derivative(input.iter().copied().map(|v| (v as u8, v)))
        .map(|(_, v)| v)
        .collect();
    assert_eq!(expected, output);
}

pub fn derivative_buffer_u8(input: &[u8], output: &mut [i32]) {
    derivative(input.iter().copied().map(|v| (v, v as i32)))
        .map(|(_, v)| v)
        .enumerate()
        .for_each(|(i, v)| output[i] = v)
}

pub fn derivative_buffer_i32(input: &[i32], output: &mut [i32]) {
    derivative(input.iter().copied().map(|v| (v as u8, v)))
        .map(|(_, v)| v)
        .enumerate()
        .for_each(|(i, v)| output[i] = v)
}

pub fn min_max<'a>(
    input: impl Iterator<Item = (isize, u8, i32)> + 'a,
) -> impl Iterator<Item = (isize, u8, i32)> + 'a {
    let mut increasing: bool = false;
    let mut previous: i32 = 0;
    input.filter_map(move |(index, original, next)| {
        let current = previous;
        previous = next;
        if increasing {
            if next < current {
                increasing = false;
                Some((index - 1, original, current))
            } else {
                None
            }
        } else {
            if next > current {
                increasing = true;
                Some((index - 1, original, current))
            } else {
                None
            }
        }
    })
}

#[test]
fn test_min_max() {
    let input: Vec<i32> = vec![5, 9, 7, 6, 3, 0, -2, -5, 3];
    let expected: Vec<(isize, u8, i32)> = vec![(-1, 0, 0), (1, 2, 9), (7, 8, -5)];
    let output: Vec<(isize, u8, i32)> = min_max(
        input
            .iter()
            .copied()
            .enumerate()
            .map(|(i, v)| (i as isize, i as u8, v)),
    )
    .collect();
    assert_eq!(expected, output);
}

#[derive(Clone, Copy)]
pub struct CurvePoint {
    pub index: usize,
    pub value: u8,
    pub d2: i32,
}

impl CurvePoint {
    pub fn zero(index: usize, value: u8) -> Self {
        Self {
            index,
            value,
            d2: 0,
        }
    }
}

pub fn get_curve_points_blur_n<'a>(
    input: &'a [u8],
    blur_size: usize,
) -> impl Iterator<Item = CurvePoint> + 'a {
    let length = input.len();
    let signed_skip: isize = blur_size as isize + 4;
    let blurred = blur_n(input, blur_size).map(|b| (b as u8, b));
    let d1 = derivative(blurred);
    let d2 = derivative(d1);
    let indexed_d2 = d2
        .zip(-signed_skip..)
        .map(|((original, d2), index)| (index, original, d2));
    let curve_points = min_max(indexed_d2.filter(|(i, _, _)| *i >= 0).take(length));
    curve_points.map(|(index, value, d2)| CurvePoint {
        index: index as usize,
        value,
        d2,
    })
}

pub fn get_curve_points_blur<'a, const SIZE: usize>(
    input: &'a [u8],
) -> impl Iterator<Item = CurvePoint> + 'a {
    let length = input.len();
    let signed_skip: isize = SIZE as isize + 4;
    let blurred = blur::<SIZE>(input).map(|b| (b as u8, b));
    let d1 = derivative(blurred);
    let d2 = derivative(d1);
    let indexed_d2 = d2
        .zip(-signed_skip..)
        .map(|((original, d2), index)| (index, original, d2));
    let curve_points = min_max(indexed_d2.filter(|(i, _, _)| *i >= 0).take(length));
    curve_points.map(|(index, value, d2)| CurvePoint {
        index: index as usize,
        value,
        d2,
    })
}

pub fn segment_buffer(input: &[u8], output: &mut [u8], blur_size: usize) {
    let length = input.len();
    let mut curve_points = Vec::with_capacity(length + 2);
    curve_points.push(CurvePoint::zero(0, input[0]));
    curve_points.extend(get_curve_points_blur_n(input, blur_size));
    curve_points.push(CurvePoint::zero(length - 1, input[length - 1]));
    let mut points_iter = curve_points.into_iter();
    let mut current = points_iter.next().unwrap();
    for next in points_iter {
        let current_index = current.index as i32;
        let next_index = next.index as i32;
        let delta_index = next_index - current_index;
        let current_value = current.value as i32;
        let next_value = next.value as i32;
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

#[test]
fn test_segment_buffer() {
    let input: Vec<u8> = vec![1, 2, 7, 6, 5, 0, 0, 3, 7];
    let expected: Vec<u8> = vec![1, 2, 7, 6, 5, 0, 0, 3, 7];
    let mut output = [0u8; 9];
    segment_buffer(&input, &mut output, 1);
    assert_eq!(expected, output);
}
