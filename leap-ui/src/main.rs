use leap_device::device::{
    LeapDevice, LeapMode, LeapStream, LEAP_MAX_FRAME_SIZE, LEAP_MAX_X_RESOLUTION,
};
use leap_device::processing::{
    blur_n_buffer, derivative_1_buffer, derivative_2_buffer, get_curve_points_blur_n,
    segment_buffer, LineProcessingParameters, MAX_BLUR_SIZE_DERIVATIVE, MAX_BLUR_SIZE_VALUE,
    MAX_DERIVATIVE_SCALING, MIN_BLUR_SIZE, MIN_DERIVATIVE_SCALING,
};
use leap_device::vision::{
    LensModelParams, LensModelParamsData, LineReader, LEAP_BARREL_DISTORTION_MAX,
    LEAP_BARREL_DISTORTION_MIN, LEAP_Y_OFFSET_MAX, LEAP_Y_OFFSET_MIN,
};
use lib::eframe::egui::plot::{Line, Value, Values};
use lib::eframe::egui::{Color32, Visuals};
use lib::eframe::epaint::{ColorImage, TextureHandle};
use lib::eframe::{egui, CreationContext};

const LEAP_MODE: LeapMode = LeapMode::m640x480();
const BUFFER_COUNT: u32 = 4;

const DISPLAY_DERIVATIVE_SCALING: i32 = 16;
fn derivative_scaling(params_scaling: i16) -> i32 {
    let s = params_scaling as i32;
    if s < DISPLAY_DERIVATIVE_SCALING {
        DISPLAY_DERIVATIVE_SCALING / s
    } else if s > DISPLAY_DERIVATIVE_SCALING {
        s / DISPLAY_DERIVATIVE_SCALING
    } else {
        1
    }
}
fn derivative_scaling_d1(params: &LineProcessingParameters) -> i32 {
    derivative_scaling(params.derivative_scaling_d1)
}
fn derivative_scaling_d2(params: &LineProcessingParameters) -> i32 {
    derivative_scaling(params.derivative_scaling_d2)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ProcessingDisplay {
    Raw,
    Blur,
    Segmented,
}

impl ProcessingDisplay {
    pub fn text(&self) -> &'static str {
        match self {
            ProcessingDisplay::Raw => "RAW",
            ProcessingDisplay::Blur => "BLUR",
            ProcessingDisplay::Segmented => "SEGMENTED",
        }
    }
}

struct UiState<'a> {
    pub lens_data: LensModelParamsData,
    pub corrected: bool,
    pub current_line: usize,
    pub tick: usize,
    pub dev: LeapDevice,
    pub stream: LeapStream<'a>,
    pub reader: LineReader,
    pub texture: TextureHandle,
    pub current_left_line: [u8; LEAP_MAX_X_RESOLUTION],
    pub current_right_line: [u8; LEAP_MAX_X_RESOLUTION],
    pub capture_frame: bool,
    pub frame_is_captured: bool,
    pub captured_frame: [u8; LEAP_MAX_FRAME_SIZE],
    pub line_processing_params: LineProcessingParameters,
    pub show_left: bool,
    pub show_right: bool,
    pub display_image: ProcessingDisplay,
    pub display_line: ProcessingDisplay,
    pub display_line_d1: bool,
    pub display_line_d2: bool,
    pub display_curve_points: bool,
}

impl<'s> UiState<'s> {
    pub fn new(cc: &CreationContext, dev: LeapDevice) -> Self {
        cc.egui_ctx.set_visuals(Visuals::dark());
        let img_w = dev.mode().x_res() as usize;
        let img_h = dev.mode().y_res() as usize;
        let stream = dev.stream(BUFFER_COUNT).unwrap();
        let reader = LineReader::new(LensModelParams::new(
            *dev.mode(),
            LensModelParamsData::default(),
        ));
        let image = ColorImage::new([img_w, img_h], Color32::from_rgb(0, 0, 0).to_opaque());
        let texture = cc.egui_ctx.load_texture("leap-motion-image", image);

        Self {
            lens_data: LensModelParamsData::default()
                .with_y_left_offset(-6)
                .with_y_right_offset(7),
            corrected: true,
            current_line: img_h / 2,
            tick: 0,
            dev,
            stream,
            reader,
            texture,
            current_left_line: [0; LEAP_MAX_X_RESOLUTION],
            current_right_line: [0; LEAP_MAX_X_RESOLUTION],
            capture_frame: false,
            frame_is_captured: false,
            captured_frame: [0; LEAP_MAX_FRAME_SIZE],
            line_processing_params: LineProcessingParameters::default(),
            show_left: true,
            show_right: true,
            display_image: ProcessingDisplay::Raw,
            display_line: ProcessingDisplay::Raw,
            display_line_d1: false,
            display_line_d2: false,
            display_curve_points: false,
        }
    }

    pub fn process_left_line_values(&self, output: &mut [u8], display: ProcessingDisplay) {
        match display {
            ProcessingDisplay::Raw => {
                output.copy_from_slice(&self.current_left_line);
            }
            ProcessingDisplay::Blur => blur_n_buffer(
                &self.current_left_line,
                output,
                self.line_processing_params.blur_size,
            ),
            ProcessingDisplay::Segmented => segment_buffer(
                &self.current_left_line,
                output,
                &self.line_processing_params,
            ),
        }
    }
    pub fn process_right_line_values(&self, output: &mut [u8], display: ProcessingDisplay) {
        match display {
            ProcessingDisplay::Raw => {
                output.copy_from_slice(&self.current_right_line);
            }
            ProcessingDisplay::Blur => blur_n_buffer(
                &self.current_right_line,
                output,
                self.line_processing_params.blur_size,
            ),
            ProcessingDisplay::Segmented => segment_buffer(
                &self.current_right_line,
                output,
                &self.line_processing_params,
            ),
        }
    }

    pub fn save_next_frame(&mut self) -> bool {
        if let Some((frame, _meta)) = self.stream.next_frame().ok() {
            self.captured_frame[0..frame.len()].clone_from_slice(frame);
            true
        } else {
            false
        }
    }

    pub fn show_next_frame(&mut self) {
        if let Some((frame, _meta)) = self.stream.next_frame().ok() {
            let (x_res, y_res) = (
                self.dev.mode().x_res() as usize,
                self.dev.mode().y_res() as usize,
            );
            let mut left_line_raw = vec![0u8; x_res];
            let mut right_line_raw = vec![0u8; x_res];
            let mut left_line_processed = vec![0u8; x_res];
            let mut right_line_processed = vec![0u8; x_res];

            let mut current_left_line = vec![0u8; x_res];
            let mut current_right_line = vec![0u8; x_res];

            let frame = if self.frame_is_captured {
                &self.captured_frame
            } else {
                frame
            };

            let mut image = ColorImage::new([x_res, y_res], Color32::from_rgb(0, 0, 0).to_opaque());
            for y in (0..y_res).into_iter() {
                if self.corrected {
                    self.reader
                        .get_left_line_corrected(frame, &mut left_line_raw, y as u32);
                    self.reader
                        .get_right_line_corrected(frame, &mut right_line_raw, y as u32);
                } else {
                    self.reader
                        .get_left_line_direct(frame, &mut left_line_raw, y as u32);
                    self.reader
                        .get_right_line_direct(frame, &mut right_line_raw, y as u32);
                }

                match self.display_image {
                    ProcessingDisplay::Raw => {
                        left_line_processed.clone_from_slice(&left_line_raw);
                        right_line_processed.clone_from_slice(&right_line_raw);
                    }
                    ProcessingDisplay::Blur => {
                        blur_n_buffer(
                            &left_line_raw[..],
                            &mut left_line_processed,
                            self.line_processing_params.blur_size,
                        );
                        blur_n_buffer(
                            &right_line_raw[..],
                            &mut right_line_processed,
                            self.line_processing_params.blur_size,
                        );
                    }
                    ProcessingDisplay::Segmented => {
                        segment_buffer(
                            &left_line_raw[..],
                            &mut left_line_processed,
                            &self.line_processing_params,
                        );
                        segment_buffer(
                            &right_line_raw[..],
                            &mut right_line_processed,
                            &self.line_processing_params,
                        );
                    }
                }

                if y == self.current_line {
                    current_left_line.clone_from_slice(&left_line_raw);
                    current_right_line.clone_from_slice(&right_line_raw);
                }

                let blue = if y == self.current_line { 255 } else { 0 };

                (0..x_res).into_iter().for_each(|x| {
                    let left = left_line_processed[x as usize];
                    let right = right_line_processed[x as usize];
                    let c = Color32::from_rgb(left, right, blue).to_opaque();
                    image[(x, y)] = c;
                });
            }
            self.texture.set(image);

            self.current_left_line[0..x_res].clone_from_slice(&current_left_line);
            self.current_right_line[0..x_res].clone_from_slice(&current_right_line);
        }
    }

    pub fn update_frame(&mut self) {
        self.tick += 1;

        if self.capture_frame {
            if self.save_next_frame() {
                self.capture_frame = false;
                self.frame_is_captured = true;
            }
        } else {
            self.show_next_frame();
        }
    }
}

impl<'s> lib::eframe::App for UiState<'s> {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut lib::eframe::Frame) {
        self.update_frame();

        egui::SidePanel::left("side_panel").show(ctx, |ui| {
            /*
             * Frame Control
             */
            ui.heading("Frame Control");

            ui.horizontal(|ui| {
                if self.frame_is_captured {
                    if ui.button("Resume").clicked() {
                        self.capture_frame = false;
                        self.frame_is_captured = false;
                    }
                    if ui.button("Load").clicked() {
                        match self.stream.load_frame("leapmotion-frame.png") {
                            Ok(buf) => {
                                self.captured_frame.fill(0);
                                self.captured_frame[0..buf.len()].clone_from_slice(&buf);
                            }
                            Err(err) => {
                                println!("ERROR: {}", err);
                            }
                        }
                        if let Err(err) = self.stream.load_frame("leapmotion-frame.png") {
                            println!("ERROR: {}", err);
                        };
                    }
                    if ui.button("Save").clicked() {
                        if let Err(err) = self
                            .stream
                            .save_frame(&self.captured_frame, "leapmotion-frame.png")
                        {
                            println!("ERROR: {}", err);
                        };
                    }
                } else {
                    if ui.button("Freeze").clicked() {
                        self.capture_frame = true;
                        self.frame_is_captured = false;
                        self.captured_frame.fill(0);
                    }
                }
            });

            /*
             * Distortion
             */
            ui.heading("Distortion Parameters");

            ui.horizontal(|ui| {
                ui.checkbox(&mut self.corrected, "Corrected");
            });

            ui.horizontal(|ui| {
                if ui.button("-").clicked() {
                    self.lens_data.barrel_distortion -= 0.01;
                }
                if ui.button("+").clicked() {
                    self.lens_data.barrel_distortion += 0.01;
                }
                ui.add(
                    egui::Slider::new(
                        &mut self.lens_data.barrel_distortion,
                        LEAP_BARREL_DISTORTION_MIN..=LEAP_BARREL_DISTORTION_MAX,
                    )
                    .text("Distortion"),
                );
            });

            ui.horizontal(|ui| {
                if ui.button("-").clicked() {
                    self.lens_data.y_left_offset -= 1;
                }
                if ui.button("+").clicked() {
                    self.lens_data.y_left_offset += 1;
                }
                ui.add(
                    egui::Slider::new(
                        &mut self.lens_data.y_left_offset,
                        LEAP_Y_OFFSET_MIN..=LEAP_Y_OFFSET_MAX,
                    )
                    .text("Left Y Offset"),
                );
            });

            ui.horizontal(|ui| {
                if ui.button("-").clicked() {
                    self.lens_data.y_right_offset -= 1;
                }
                if ui.button("+").clicked() {
                    self.lens_data.y_right_offset += 1;
                }
                ui.add(
                    egui::Slider::new(
                        &mut self.lens_data.y_right_offset,
                        LEAP_Y_OFFSET_MIN..=LEAP_Y_OFFSET_MAX,
                    )
                    .text("Right Y Offset"),
                );
            });

            /*
             * Feature thresholds
             */
            ui.heading("Line Processing Parameters");
            ui.horizontal(|ui| {
                if ui.button("-").clicked() {
                    self.line_processing_params.blur_size -= 1;
                }
                if ui.button("+").clicked() {
                    self.line_processing_params.blur_size += 1;
                }
                ui.add(
                    egui::Slider::new(
                        &mut self.line_processing_params.blur_size,
                        MIN_BLUR_SIZE..=MAX_BLUR_SIZE_VALUE,
                    )
                    .text("Blur Size"),
                );
            });
            ui.horizontal(|ui| {
                if ui.button("-").clicked() {
                    self.line_processing_params.blur_size_d1 -= 1;
                }
                if ui.button("+").clicked() {
                    self.line_processing_params.blur_size_d1 += 1;
                }
                ui.add(
                    egui::Slider::new(
                        &mut self.line_processing_params.blur_size_d1,
                        MIN_BLUR_SIZE..=MAX_BLUR_SIZE_DERIVATIVE,
                    )
                    .text("Blur Size d1"),
                );
            });
            ui.horizontal(|ui| {
                if ui.button("-").clicked() {
                    self.line_processing_params.blur_size_d2 -= 1;
                }
                if ui.button("+").clicked() {
                    self.line_processing_params.blur_size_d2 += 1;
                }
                ui.add(
                    egui::Slider::new(
                        &mut self.line_processing_params.blur_size_d2,
                        MIN_BLUR_SIZE..=MAX_BLUR_SIZE_DERIVATIVE,
                    )
                    .text("Blur Size d2"),
                );
            });
            ui.horizontal(|ui| {
                if ui.button("-").clicked() {
                    self.line_processing_params.derivative_scaling_d1 -= 1;
                }
                if ui.button("+").clicked() {
                    self.line_processing_params.derivative_scaling_d1 += 1;
                }
                ui.add(
                    egui::Slider::new(
                        &mut self.line_processing_params.derivative_scaling_d1,
                        MIN_DERIVATIVE_SCALING..=MAX_DERIVATIVE_SCALING,
                    )
                    .text("Derivative Scaling d2"),
                );
            });
            ui.horizontal(|ui| {
                if ui.button("-").clicked() {
                    self.line_processing_params.derivative_scaling_d2 -= 1;
                }
                if ui.button("+").clicked() {
                    self.line_processing_params.derivative_scaling_d2 += 1;
                }
                ui.add(
                    egui::Slider::new(
                        &mut self.line_processing_params.derivative_scaling_d2,
                        MIN_DERIVATIVE_SCALING..=MAX_DERIVATIVE_SCALING,
                    )
                    .text("Derivative Scaling d2"),
                );
            });
            self.line_processing_params.fix();

            /*
             * Diagrams selection
             */
            ui.heading("Diagrams");
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.show_left, "Left");
                if ui.button("Both").clicked() {
                    self.show_left = true;
                    self.show_right = true;
                }
                ui.checkbox(&mut self.show_right, "Right");
            });
            ui.horizontal(|ui| {
                ui.label("Display:");
                ui.selectable_value(
                    &mut self.display_image,
                    ProcessingDisplay::Raw,
                    ProcessingDisplay::Raw.text(),
                );
                ui.selectable_value(
                    &mut self.display_image,
                    ProcessingDisplay::Blur,
                    ProcessingDisplay::Blur.text(),
                );
                ui.selectable_value(
                    &mut self.display_image,
                    ProcessingDisplay::Segmented,
                    ProcessingDisplay::Segmented.text(),
                );
            });

            /*
             * Scan Line
             */
            ui.heading("Scan Line");

            let (current_line_min, current_line_max) =
                (self.reader.corrected_y_min(), self.reader.corrected_y_max());
            ui.horizontal(|ui| {
                if ui.button("-").clicked() {
                    self.current_line -= 1;
                }
                if ui.button("+").clicked() {
                    self.current_line += 1;
                }
                ui.add(
                    egui::Slider::new(&mut self.current_line, current_line_min..=current_line_max)
                        .text("Current Line"),
                );
            });
            self.current_line = self
                .current_line
                .min(current_line_max)
                .max(current_line_min);
            ui.label("Current Line Display:");
            ui.horizontal(|ui| {
                ui.selectable_value(
                    &mut self.display_line,
                    ProcessingDisplay::Raw,
                    ProcessingDisplay::Raw.text(),
                );
                ui.selectable_value(
                    &mut self.display_line,
                    ProcessingDisplay::Blur,
                    ProcessingDisplay::Blur.text(),
                );
                ui.selectable_value(
                    &mut self.display_line,
                    ProcessingDisplay::Segmented,
                    ProcessingDisplay::Segmented.text(),
                );
            });
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.display_line_d1, "Line d1/dp");
                ui.checkbox(&mut self.display_line_d2, "Line d2/dp");
                ui.checkbox(&mut self.display_curve_points, "Curve Points");
            });
            ui.with_layout(egui::Layout::bottom_up(egui::Align::Center), |ui| {
                if ui.button("Quit").clicked() {
                    std::process::exit(0);
                }
            });

            self.lens_data = self.lens_data.fix();
            if self.reader.params().data() != self.lens_data {
                self.reader =
                    LineReader::new(LensModelParams::new(self.stream.mode(), self.lens_data));
            }
        });

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.vertical(|ui| {
                let x_res = self.stream.mode().x_res() as usize;
                let y_res = self.stream.mode().y_res() as usize;

                ui.add(egui::widgets::Image::new(
                    self.texture.id(),
                    [x_res as f32, y_res as f32],
                ));

                let plot = egui::widgets::plot::Plot::new("plot")
                    .include_x(0.0)
                    .include_x(x_res as f32)
                    .include_y(255.0)
                    .include_y(-255.0)
                    .width(x_res as f32)
                    .view_aspect(x_res as f32 / (2.0 * 255.0));

                plot.show(ui, |plot_ui| {
                    // Left line
                    if self.show_left {
                        let mut line_values = vec![0u8; LEAP_MAX_X_RESOLUTION];
                        self.process_left_line_values(&mut line_values, self.display_line);

                        plot_ui.line(
                            Line::new(values_to_line_points(
                                line_values.iter().copied().enumerate(),
                            ))
                            .color(Color32::RED),
                        );

                        if self.display_line_d1 {
                            let mut d = vec![0i16; LEAP_MAX_X_RESOLUTION];
                            derivative_1_buffer(
                                &self.current_left_line,
                                &mut d,
                                &self.line_processing_params,
                            );

                            plot_ui.line(
                                Line::new(values_i32_to_line_points(
                                    d.iter()
                                        .copied()
                                        .map(|v| {
                                            v as i32
                                                * derivative_scaling_d1(
                                                    &self.line_processing_params,
                                                )
                                        })
                                        .enumerate(),
                                ))
                                .color(Color32::RED),
                            );
                        }

                        if self.display_line_d2 {
                            let mut d = vec![0i16; LEAP_MAX_X_RESOLUTION];
                            derivative_2_buffer(
                                &self.current_left_line,
                                &mut d,
                                &self.line_processing_params,
                            );

                            plot_ui.line(
                                Line::new(values_i32_to_line_points(
                                    d.iter()
                                        .copied()
                                        .map(|v| {
                                            v as i32
                                                * derivative_scaling_d2(
                                                    &self.line_processing_params,
                                                )
                                        })
                                        .enumerate(),
                                ))
                                .color(Color32::RED),
                            );
                        }

                        if self.display_curve_points {
                            let curve_points = get_curve_points_blur_n(
                                &self.current_left_line,
                                &self.line_processing_params,
                            );

                            values_to_vertical_lines(
                                curve_points.map(|p| (p.index, self.current_left_line[p.index])),
                            )
                            .for_each(|l| plot_ui.line(l.color(Color32::RED)));
                        }
                    }

                    // Right line
                    if self.show_right {
                        let mut line_values = vec![0u8; LEAP_MAX_X_RESOLUTION];
                        self.process_right_line_values(&mut line_values, self.display_line);

                        plot_ui.line(
                            Line::new(values_to_line_points(
                                line_values.iter().copied().enumerate(),
                            ))
                            .color(Color32::GREEN),
                        );

                        if self.display_line_d1 {
                            let mut d = vec![0i16; LEAP_MAX_X_RESOLUTION];
                            derivative_1_buffer(
                                &self.current_right_line,
                                &mut d,
                                &self.line_processing_params,
                            );

                            plot_ui.line(
                                Line::new(values_i32_to_line_points(
                                    d.iter()
                                        .copied()
                                        .map(|v| {
                                            v as i32
                                                * derivative_scaling_d1(
                                                    &self.line_processing_params,
                                                )
                                        })
                                        .enumerate(),
                                ))
                                .color(Color32::GREEN),
                            );
                        }

                        if self.display_line_d2 {
                            let mut d = vec![0i16; LEAP_MAX_X_RESOLUTION];
                            derivative_2_buffer(
                                &self.current_right_line,
                                &mut d,
                                &self.line_processing_params,
                            );

                            plot_ui.line(
                                Line::new(values_i32_to_line_points(
                                    d.iter()
                                        .copied()
                                        .map(|v| {
                                            v as i32
                                                * derivative_scaling_d2(
                                                    &self.line_processing_params,
                                                )
                                        })
                                        .enumerate(),
                                ))
                                .color(Color32::GREEN),
                            );
                        }

                        if self.display_curve_points {
                            let curve_points = get_curve_points_blur_n(
                                &self.current_right_line,
                                &self.line_processing_params,
                            );

                            values_to_vertical_lines(
                                curve_points.map(|p| (p.index, self.current_right_line[p.index])),
                            )
                            .for_each(|l| plot_ui.line(l.color(Color32::GREEN)));
                        }
                    }
                });
            });
        });

        ctx.request_repaint();
    }
}

fn values_to_line_points(line: impl Iterator<Item = (usize, u8)>) -> Values {
    Values::from_values_iter(line.map(|(pos, val)| Value::new(pos as f64, val as f64)))
}

fn values_to_vertical_lines(line: impl Iterator<Item = (usize, u8)>) -> impl Iterator<Item = Line> {
    line.map(|(pos, val)| {
        Line::new(Values::from_values(vec![
            Value::new(pos as f64, -1.0),
            Value::new(pos as f64, val as f64),
        ]))
    })
}

fn values_i32_to_line_points(line: impl Iterator<Item = (usize, i32)>) -> Values {
    Values::from_values_iter(line.map(|(pos, val)| Value::new(pos as f64, val as f64)))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = LeapDevice::setup_with_options(LEAP_MODE, true)?;
    let options = lib::eframe::NativeOptions {
        initial_window_size: Some(egui::vec2(
            dev.mode().x_res() as f32,
            dev.mode().y_res() as f32,
        )),
        ..Default::default()
    };
    lib::eframe::run_native(
        "Leap Motion Output",
        options,
        Box::new(|cc| {
            let app_state = UiState::new(cc, dev);
            Box::new(app_state)
        }),
    );
}
