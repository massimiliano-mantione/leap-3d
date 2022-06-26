use std::{
    iter::repeat,
    ops::{Add, Div, Mul, Sub},
};

pub const MIN_BLUR_SIZE: usize = 1;
pub const MAX_BLUR_SIZE_VALUE: usize = 32;
pub const MAX_BLUR_SIZE_DERIVATIVE: usize = 32;
pub const MIN_VALUE_SCALING: i16 = 1;
pub const MAX_VALUE_SCALING: i16 = 8;
pub const MIN_DERIVATIVE_SCALING: i16 = 1;
pub const MAX_DERIVATIVE_SCALING: i16 = 64;

#[derive(Debug, Clone)]
pub struct LineProcessingParameters {
    pub blur_size: usize,
    pub blur_size_d1: usize,
    pub blur_size_d2: usize,
    pub value_scaling: i16,
    pub derivative_scaling_d1: i16,
    pub derivative_scaling_d2: i16,
}

impl Default for LineProcessingParameters {
    fn default() -> Self {
        Self {
            blur_size: 8,
            blur_size_d1: 8,
            blur_size_d2: 8,
            value_scaling: MIN_VALUE_SCALING,
            derivative_scaling_d1: 2,
            derivative_scaling_d2: 2,
        }
    }
}

fn fix<T>(value: &mut T, min: T, max: T)
where
    T: Copy + Ord,
{
    if *value < min {
        *value = min;
    }
    if *value > max {
        *value = max;
    }
}

impl LineProcessingParameters {
    pub fn fix(&mut self) {
        fix(&mut self.blur_size, MIN_BLUR_SIZE, MAX_BLUR_SIZE_VALUE);
        fix(
            &mut self.blur_size_d1,
            MIN_BLUR_SIZE,
            MAX_BLUR_SIZE_DERIVATIVE,
        );
        fix(
            &mut self.blur_size_d2,
            MIN_BLUR_SIZE,
            MAX_BLUR_SIZE_DERIVATIVE,
        );

        fix(
            &mut self.value_scaling,
            MIN_VALUE_SCALING,
            MAX_VALUE_SCALING,
        );
        fix(
            &mut self.derivative_scaling_d1,
            MIN_DERIVATIVE_SCALING,
            MAX_DERIVATIVE_SCALING,
        );
        fix(
            &mut self.derivative_scaling_d2,
            MIN_DERIVATIVE_SCALING,
            MAX_DERIVATIVE_SCALING,
        );
    }
}

pub trait FromImageValue {
    fn from_value_u8(value: u8) -> Self;
    fn from_value_i16(value: i16) -> Self;
    fn from_value_usize(value: usize) -> Self;
    fn into_value_u8(&self) -> u8;
    fn into_value_i16(&self) -> i16;
    fn mul_saturating(&self, v: &Self) -> Self;
}

macro_rules! impl_from_image_value_i {
    ($t: ident) => {
        impl FromImageValue for $t {
            fn from_value_u8(value: u8) -> Self {
                value as $t
            }

            fn from_value_i16(value: i16) -> Self {
                value as $t
            }

            fn from_value_usize(value: usize) -> Self {
                value as $t
            }

            fn into_value_u8(&self) -> u8 {
                *self as u8
            }

            fn into_value_i16(&self) -> i16 {
                *self as i16
            }

            fn mul_saturating(&self, v: &Self) -> Self {
                self.saturating_mul(*v)
            }
        }
    };
}

macro_rules! impl_from_image_value_f {
    ($t: ident) => {
        impl FromImageValue for $t {
            fn from_value_u8(value: u8) -> Self {
                value as $t
            }

            fn from_value_i16(value: i16) -> Self {
                value as $t
            }

            fn from_value_usize(value: usize) -> Self {
                value as $t
            }

            fn into_value_u8(&self) -> u8 {
                *self as u8
            }

            fn into_value_i16(&self) -> i16 {
                *self as i16
            }

            fn mul_saturating(&self, v: &Self) -> Self {
                self * v
            }
        }
    };
}

pub type PNum = f32;

impl_from_image_value_i!(i32);
impl_from_image_value_f!(f32);

pub trait SumWindow<T>
where
    T: Add<Output = T> + Sub<Output = T> + Copy + FromImageValue + Mul<Output = T>,
{
    fn tick(&mut self, input: T) -> T;
    fn size(&self) -> usize;
    fn sum(&self) -> T;
}

pub struct SumWindowDynamic<T, const CAP: usize>
where
    T: Add<Output = T> + Sub<Output = T> + Copy + FromImageValue + Mul<Output = T>,
{
    data: [T; CAP],
    next: usize,
    size: usize,
    sum: T,
}

impl<T, const CAP: usize> SumWindowDynamic<T, CAP>
where
    T: Add<Output = T> + Sub<Output = T> + Copy + FromImageValue + Mul<Output = T>,
{
    pub fn new(fill: T, size: usize) -> Self {
        Self {
            data: [fill; CAP],
            next: 0,
            size,
            sum: fill * T::from_value_usize(size),
        }
    }
}

impl<T, const CAP: usize> SumWindow<T> for SumWindowDynamic<T, CAP>
where
    T: Add<Output = T> + Sub<Output = T> + Copy + FromImageValue + Mul<Output = T>,
{
    fn tick(&mut self, input: T) -> T {
        let output = self.data[self.next];
        self.data[self.next] = input;
        self.sum = self.sum - output;
        self.sum = self.sum + input;
        self.next = (self.next + 1) % self.size;
        output
    }

    fn size(&self) -> usize {
        self.size
    }

    fn sum(&self) -> T {
        self.sum
    }
}

pub type SumWindowI16 = SumWindowDynamic<i16, 32>;

pub struct SumWindowConst<T, const SIZE: usize>
where
    T: Add<Output = T> + Sub<Output = T> + Copy + FromImageValue + Mul<Output = T>,
{
    data: [T; SIZE],
    next: usize,
    sum: T,
}

impl<T, const SIZE: usize> SumWindowConst<T, SIZE>
where
    T: Add<Output = T> + Sub<Output = T> + Copy + FromImageValue + Mul<Output = T>,
{
    pub fn new(fill: T) -> Self {
        Self {
            data: [fill; SIZE],
            next: 0,
            sum: fill * T::from_value_usize(SIZE),
        }
    }
}

impl<T, const SIZE: usize> SumWindow<T> for SumWindowConst<T, SIZE>
where
    T: Add<Output = T> + Sub<Output = T> + Copy + FromImageValue + Mul<Output = T>,
{
    fn tick(&mut self, input: T) -> T {
        let output = self.data[self.next];
        self.data[self.next] = input;
        self.sum = self.sum - output;
        self.sum = self.sum + input;
        self.next = (self.next + 1) % SIZE;
        output
    }

    fn size(&self) -> usize {
        SIZE
    }

    fn sum(&self) -> T {
        self.sum
    }
}

pub fn blur_n<'a, T>(
    input: &'a [u8],
    params: &'a LineProcessingParameters,
) -> impl Iterator<Item = T> + 'a
where
    T: 'a
        + Add<Output = T>
        + Sub<Output = T>
        + Copy
        + FromImageValue
        + Mul<Output = T>
        + Div<Output = T>,
{
    let fill = T::from_value_u8(input[0]);
    let tail = T::from_value_u8(input[input.len() - 1]);
    let mut window = SumWindowDynamic::<T, 32>::new(fill, params.blur_size);

    input
        .iter()
        .copied()
        .map(|v| T::from_value_u8(v))
        .chain(repeat(tail))
        .map(move |value| {
            let value = value.mul_saturating(&T::from_value_i16(params.value_scaling));
            window.tick(value);
            window.sum() / T::from_value_usize(params.blur_size)
        })
}

pub fn blur_n_buffer<T>(input: &[u8], output: &mut [u8], params: &LineProcessingParameters)
where
    T: Add<Output = T>
        + Sub<Output = T>
        + Copy
        + FromImageValue
        + Mul<Output = T>
        + Div<Output = T>,
{
    blur_n::<T>(input, params)
        .skip(params.blur_size)
        .take(input.len())
        .map(|v| v / T::from_value_i16(params.value_scaling))
        .map(|v| v.into_value_u8())
        .enumerate()
        .for_each(|(i, v)| output[i] = v);
}

#[test]
fn test_blur_buffer_flat() {
    let params = LineProcessingParameters {
        blur_size: 4,
        value_scaling: 0,
        ..Default::default()
    };
    let input: [u8; 9] = [9, 9, 9, 9, 9, 9, 9, 9, 9];
    let mut output = [0u8; 9];
    let expected: [u8; 9] = [9, 9, 9, 9, 9, 9, 9, 9, 9];
    blur_n_buffer::<PNum>(&input, &mut output, &params);
    assert_eq!(output.to_vec(), expected.to_vec());
}

#[test]
fn test_blur_buffer_spike() {
    let params = LineProcessingParameters {
        blur_size: 4,
        value_scaling: 0,
        ..Default::default()
    };
    let input: [u8; 9] = [9, 9, 9, 9, 18, 9, 9, 9, 9];
    let mut output = [0u8; 9];
    let expected: [u8; 9] = [10, 10, 10, 10, 10, 10, 10, 10, 10];
    blur_n_buffer::<PNum>(&input, &mut output, &params);
    assert_eq!(output.to_vec(), expected.to_vec());
}

pub fn derivative_n<T>(
    input: impl Iterator<Item = T>,
    blur_size: usize,
    derivative_scaling: i16,
) -> impl Iterator<Item = T>
where
    T: Add<Output = T>
        + Sub<Output = T>
        + Copy
        + FromImageValue
        + Mul<Output = T>
        + Div<Output = T>,
{
    let zero = T::from_value_u8(0);
    let mut previous_value = zero;
    let mut window = SumWindowDynamic::<T, 32>::new(zero, blur_size);
    let blur_size = T::from_value_usize(blur_size);
    let derivative_scaling = T::from_value_i16(derivative_scaling);

    input.chain(repeat(zero)).map(move |value| {
        let raw_d1 = (value - previous_value) * derivative_scaling;
        previous_value = value;
        window.tick(raw_d1);
        window.sum() / blur_size
    })
}

pub fn derivative_1_buffer<T>(input: &[u8], output: &mut [i16], params: &LineProcessingParameters)
where
    T: Add<Output = T>
        + Sub<Output = T>
        + Copy
        + FromImageValue
        + Mul<Output = T>
        + Div<Output = T>,
{
    let blurred = blur_n::<T>(input, params);
    let derivative_1 = derivative_n(blurred, params.blur_size_d1, params.derivative_scaling_d1);
    derivative_1
        .skip(params.blur_size + params.blur_size_d1)
        .take(input.len())
        .enumerate()
        .for_each(|(i, v)| output[i] = v.into_value_i16());
}

pub fn derivative_2_buffer<T>(input: &[u8], output: &mut [i16], params: &LineProcessingParameters)
where
    T: Add<Output = T>
        + Sub<Output = T>
        + Copy
        + FromImageValue
        + Mul<Output = T>
        + Div<Output = T>,
{
    let blurred = blur_n::<T>(input, params);
    let derivative_1 = derivative_n(blurred, params.blur_size_d1, params.derivative_scaling_d1);
    let derivative_2 = derivative_n(
        derivative_1,
        params.blur_size_d2,
        params.derivative_scaling_d2,
    );
    derivative_2
        .skip(params.blur_size + params.blur_size_d1 + params.blur_size_d2)
        .take(input.len())
        .enumerate()
        .for_each(|(i, v)| output[i] = v.into_value_i16());
}

pub fn min_max<'a, T>(
    input: impl Iterator<Item = (usize, T)> + 'a,
) -> impl Iterator<Item = (usize, T)> + 'a
where
    T: 'a + Copy + PartialOrd + FromImageValue,
{
    let mut increasing = false;
    let mut decreasing = false;
    let mut previous = T::from_value_u8(0);
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
    let input: Vec<i32> = vec![5, 9, 7, 6, 3, 0, -2, -5, 3];
    let expected: Vec<(usize, i32)> = vec![(1, 9), (7, -5)];
    let output: Vec<(usize, i32)> = min_max(input.iter().copied().enumerate()).collect();
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

pub fn get_curve_points_blur_n<'a, T>(
    input: &'a [u8],
    params: &'a LineProcessingParameters,
) -> impl Iterator<Item = CurvePoint> + 'a
where
    T: 'a
        + Add<Output = T>
        + Sub<Output = T>
        + Copy
        + FromImageValue
        + Mul<Output = T>
        + Div<Output = T>
        + PartialOrd,
{
    let length = input.len();
    let skip = params.blur_size + params.blur_size_d1 + params.blur_size_d2 + 1;
    let blurred = blur_n::<T>(input, params);
    let d1 = derivative_n(blurred, params.blur_size_d1, params.derivative_scaling_d1);
    let d2 = derivative_n(d1, params.blur_size_d2, params.derivative_scaling_d2);
    let indexed_d2 = d2.skip(skip).enumerate();
    let curve_points = min_max(indexed_d2.filter(move |(i, _)| *i < length).take(length));
    curve_points.map(|(index, d2)| CurvePoint {
        index,
        d2: d2.into_value_i16(),
    })
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

pub fn segment_buffer<T>(input: &[u8], output: &mut [u8], params: &LineProcessingParameters)
where
    T: Add<Output = T>
        + Sub<Output = T>
        + Copy
        + FromImageValue
        + Mul<Output = T>
        + Div<Output = T>
        + PartialOrd,
{
    let length = input.len();
    let mut curve_points = Vec::with_capacity(length + 2);
    curve_points.push(CurvePoint::zero(0));
    curve_points.extend(get_curve_points_blur_n::<T>(input, params));
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
