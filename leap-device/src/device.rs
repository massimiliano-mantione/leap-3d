use lib::image::{open as image_open, ImageError, Rgb, RgbImage};
pub use lib::v4l::buffer::Metadata;
use lib::v4l::buffer::Type;
use lib::v4l::io::traits::CaptureStream;
use lib::v4l::video::capture::Parameters;
use lib::v4l::video::Capture;
use lib::v4l::{prelude::*, Format, Fraction};

const MAX_DEVICE: usize = 42;

pub const LEAP_MAX_X_RESOLUTION: usize = 752;
pub const LEAP_MAX_Y_RESOLUTION: usize = 480;
pub const LEAP_MAX_FRAME_SIZE: usize = LEAP_MAX_X_RESOLUTION * LEAP_MAX_Y_RESOLUTION * 2;

#[derive(Debug, Clone, Copy)]
pub struct LeapMode {
    x_res: u32,
    y_res: u32,
    interval_num: u32,
    interval_den: u32,
}

impl LeapMode {
    pub fn x_res(&self) -> u32 {
        self.x_res
    }
    pub fn y_res(&self) -> u32 {
        self.y_res
    }
    pub fn interval_num(&self) -> u32 {
        self.interval_num
    }
    pub fn interval_den(&self) -> u32 {
        self.interval_den
    }

    pub const fn m640x480() -> Self {
        Self {
            x_res: 640,
            y_res: 480,
            interval_num: 2,
            interval_den: 115,
        }
    }

    pub const fn m640x240() -> Self {
        Self {
            x_res: 640,
            y_res: 240,
            interval_num: 1,
            interval_den: 115,
        }
    }

    pub const fn m640x120() -> Self {
        Self {
            x_res: 640,
            y_res: 120,
            interval_num: 5841,
            interval_den: 1250000,
        }
    }

    pub const fn m752x480() -> Self {
        Self {
            x_res: 752,
            y_res: 480,
            interval_num: 1,
            interval_den: 50,
        }
    }

    pub const fn m752x240() -> Self {
        Self {
            x_res: 752,
            y_res: 240,
            interval_num: 1,
            interval_den: 100,
        }
    }

    pub const fn m752x120() -> Self {
        Self {
            x_res: 752,
            y_res: 120,
            interval_num: 1,
            interval_den: 190,
        }
    }
}

#[derive(Debug)]
pub enum LeapDeviceError {
    DeviceNotFound(Option<String>),
    QueryCapsFailed(String, std::io::Error),
    SetFormatFailed(String, std::io::Error),
    SetParamsFailed(String, std::io::Error),
    CreateStreamFailed(String, std::io::Error),
    FrameCaptureFailed(String, std::io::Error),
    FileLoadFailed(String, ImageError),
    ImageColorMismatch(String),
    ImageResolutionMismatch(String, u32, u32, LeapMode),
    FileSaveFailed(String, ImageError),
}

impl std::fmt::Display for LeapDeviceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LeapDeviceError::DeviceNotFound(_) => f.write_str("Leap Motion device not found"),
            LeapDeviceError::QueryCapsFailed(path, err) => f.write_fmt(format_args!(
                "query caps failed (device {}, err {})",
                path, err
            )),
            LeapDeviceError::SetFormatFailed(path, err) => f.write_fmt(format_args!(
                "set format failed: (device {}, err {})",
                path, err
            )),
            LeapDeviceError::SetParamsFailed(path, err) => f.write_fmt(format_args!(
                "set params failed: (device {}, err {})",
                path, err
            )),
            LeapDeviceError::CreateStreamFailed(path, err) => f.write_fmt(format_args!(
                "stream creation failed: (device {}, err {})",
                path, err
            )),
            LeapDeviceError::FrameCaptureFailed(path, err) => f.write_fmt(format_args!(
                "frame capture failed: (device {}, err {})",
                path, err
            )),
            LeapDeviceError::FileLoadFailed(filename, err) => f.write_fmt(format_args!(
                "File load failed: (file: {}, err {})",
                filename, err
            )),
            LeapDeviceError::ImageColorMismatch(filename) => f.write_fmt(format_args!(
                "Image color mismatch, cannot convert ro rgb8: (file: {})",
                filename,
            )),
            LeapDeviceError::ImageResolutionMismatch(filename, x, y, mode) => {
                f.write_fmt(format_args!(
                    "Image resolution mismatch: (file: {}, res {}x{} (mode {}x{})",
                    filename,
                    x,
                    y,
                    mode.x_res(),
                    mode.y_res()
                ))
            }
            LeapDeviceError::FileSaveFailed(filename, err) => f.write_fmt(format_args!(
                "File save failed: (file: {}, err {})",
                filename, err
            )),
        }
    }
}

impl std::error::Error for LeapDeviceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

pub type LeapDeviceResult<T> = Result<T, LeapDeviceError>;

pub struct LeapDevice {
    path: String,
    dev: Device,
    mode: LeapMode,
}

pub struct LeapStream<'a> {
    path: String,
    mode: LeapMode,
    stream: MmapStream<'a>,
}

impl<'a> LeapStream<'a> {
    pub fn next_frame(&mut self) -> LeapDeviceResult<(&[u8], &Metadata)> {
        match self.stream.next() {
            Ok(capture) => Ok(capture),
            Err(err) => Err(LeapDeviceError::FrameCaptureFailed(self.path.clone(), err)),
        }
    }

    pub fn mode(&self) -> LeapMode {
        self.mode
    }

    pub fn load_frame(&self, filename: &str) -> LeapDeviceResult<Vec<u8>> {
        let (img_w, img_h) = (self.mode().x_res() as usize, self.mode().y_res() as usize);
        let img = image_open(filename)
            .map_err(|err| LeapDeviceError::FileLoadFailed(filename.to_string(), err))?;
        let img = img
            .as_rgb8()
            .ok_or_else(|| LeapDeviceError::ImageColorMismatch(filename.to_string()))?;
        if (img_w as u32, img_h as u32) != img.dimensions() {
            return Err(LeapDeviceError::ImageResolutionMismatch(
                filename.to_string(),
                img.dimensions().0,
                img.dimensions().1,
                self.mode(),
            ));
        }

        let mut frame = vec![0u8; img_w * img_h * 2];
        for (y, line) in frame.chunks_exact_mut(img_w * 2).enumerate().take(img_h) {
            for (x, values) in line.chunks_exact_mut(2).enumerate().take(img_w) {
                let pixel = img.get_pixel(x as u32, y as u32);
                values[0] = pixel[0];
                values[1] = pixel[1];
            }
        }
        Ok(frame)
    }

    pub fn save_frame(&self, frame: &[u8], filename: &str) -> LeapDeviceResult<()> {
        let (img_w, img_h) = (self.mode().x_res(), self.mode().y_res());
        let mut img = RgbImage::new(img_w, img_h);

        let (img_w, img_h) = (img_w as usize, img_h as usize);
        for (y, line) in frame.chunks_exact(img_w * 2).enumerate().take(img_h) {
            for (x, values) in line.chunks_exact(2).enumerate().take(img_w) {
                let left = values[0];
                let right = values[1];
                let p = Rgb([left, right, 0]);
                img.put_pixel(x as u32, y as u32, p);
            }
        }
        img.save(filename)
            .map_err(|err| LeapDeviceError::FileSaveFailed(filename.to_string(), err))?;
        Ok(())
    }
}

impl LeapDevice {
    pub fn setup(mode: LeapMode) -> LeapDeviceResult<Self> {
        Self::setup_with_options(mode, false)
    }

    pub fn setup_with_options(mode: LeapMode, verbose: bool) -> LeapDeviceResult<Self> {
        (0..MAX_DEVICE)
            .into_iter()
            .map(|i| format!("/dev/video{}", i))
            .fold(
                None as Option<Result<Self, LeapDeviceError>>,
                |dev, path| {
                    dev.or_else(|| {
                        let attempt = Device::with_path(&path)
                            .map(|dev| (dev, &path))
                            .map_err(|err| LeapDeviceError::DeviceNotFound(Some(err.to_string())))
                            .and_then(|(dev, path)| {
                                dev.query_caps()
                                    .map(|caps| (caps, dev, path))
                                    .map_err(|err| {
                                        LeapDeviceError::QueryCapsFailed(path.clone(), err)
                                    })
                            })
                            .and_then(|(caps, dev, path)| {
                                dev.format().map(|f| (caps, dev, f, path)).map_err(|err| {
                                    LeapDeviceError::DeviceNotFound(Some(err.to_string()))
                                })
                            })
                            .and_then(|(caps, dev, f, path)| {
                                if caps.card == "Leap Motion Controller" {
                                    Ok((dev, f.fourcc, path))
                                } else {
                                    Err(LeapDeviceError::DeviceNotFound(Some(format!(
                                        "device is a \"{}\"",
                                        caps.card
                                    ))))
                                }
                            })
                            .and_then(|(dev, fourcc, path)| {
                                dev.set_format(&Format::new(mode.x_res(), mode.y_res(), fourcc))
                                    .map(|_| (dev, path))
                                    .map_err(|err| {
                                        LeapDeviceError::SetFormatFailed(path.clone(), err)
                                    })
                            })
                            .and_then(|(dev, path)| {
                                dev.set_params(&Parameters::new(Fraction::new(
                                    mode.interval_num(),
                                    mode.interval_den(),
                                )))
                                .map(|_| LeapDevice {
                                    path: path.clone(),
                                    dev,
                                    mode,
                                })
                                .map_err(|err| LeapDeviceError::SetParamsFailed(path.clone(), err))
                            });
                        match attempt {
                            Ok(leap) => {
                                if verbose {
                                    println!("using device {}", &path);
                                }
                                Some(Ok(leap))},
                            Err(err) => match err {
                                LeapDeviceError::DeviceNotFound(reason) => {
                                    if verbose {
                                        println!("skipping device {} ({})", &path, reason.unwrap_or_else(||"unknown reason".to_string()));
                                    }
                                    None
                                }
                                _ => {
                                    if verbose {
                                        println!("error on device {}: {}", &path, &err);
                                    }
                                    Some(Err(err))
                                }
                            },
                        }
                    })
                },
            )
            .unwrap_or_else(|| Err(LeapDeviceError::DeviceNotFound(None)))
    }

    pub fn mode(&self) -> &LeapMode {
        &self.mode
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn stream<'a>(&self, buf_count: u32) -> LeapDeviceResult<LeapStream<'a>> {
        MmapStream::with_buffers(&self.dev, Type::VideoCapture, buf_count)
            .map(|stream| LeapStream {
                path: self.path.clone(),
                mode: self.mode,
                stream,
            })
            .map_err(|err| LeapDeviceError::CreateStreamFailed(self.path.clone(), err))
    }
}
