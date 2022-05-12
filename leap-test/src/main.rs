use leap_device::device::{LeapDevice, LeapMode, LeapStream};
use lib::{
    eframe,
    eframe::egui::{self, Visuals},
    eframe::{
        epaint::{Color32, ColorImage, TextureHandle},
        CreationContext,
    },
};

const BUFFER_COUNT: u32 = 4;

struct AppState<'a> {
    pub dev: LeapDevice,
    pub stream: LeapStream<'a>,
    pub texture: TextureHandle,
}

impl<'a> AppState<'a> {
    pub fn new(cc: &CreationContext, dev: LeapDevice) -> Self {
        cc.egui_ctx.set_visuals(Visuals::dark());
        let img_w = dev.mode().x_res() as usize;
        let img_h = dev.mode().y_res() as usize;
        let stream = dev.stream(BUFFER_COUNT).unwrap();
        let image = ColorImage::new([img_w, img_h], Color32::from_rgb(0, 0, 0).to_opaque());
        let texture = cc.egui_ctx.load_texture("leap-motion-image", image);
        Self {
            dev,
            stream,
            texture,
        }
    }

    pub fn img_w(&self) -> usize {
        self.dev.mode().x_res() as usize
    }

    pub fn img_h(&self) -> usize {
        self.dev.mode().y_res() as usize
    }
}

impl<'a> eframe::App for AppState<'a> {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        let img_w = self.img_w();
        let img_h = self.img_h();
        let (buf, _) = self.stream.next_frame().unwrap();

        let mut image = ColorImage::new([img_w, img_h], Color32::from_rgb(0, 0, 0).to_opaque());
        for (y, line) in buf.chunks_exact(img_w * 2).enumerate().take(img_h) {
            for (x, values) in line.chunks_exact(2).enumerate().take(img_w) {
                let left = values[0];
                let right = values[1];
                let c = Color32::from_rgb(left, right, 0).to_opaque();
                image[(x, y)] = c;
            }
        }
        self.texture.set(image);

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.image(self.texture.id(), [img_w as f32, img_h as f32]);
        });

        ctx.request_repaint();
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = LeapDevice::setup_with_options(LeapMode::m640x480(), true)?;
    let options = eframe::NativeOptions {
        initial_window_size: Some(egui::vec2(
            dev.mode().x_res() as f32,
            dev.mode().y_res() as f32,
        )),
        ..Default::default()
    };
    eframe::run_native(
        "Leap Motion Output",
        options,
        Box::new(|cc| {
            let app_state = AppState::new(cc, dev);
            Box::new(app_state)
        }),
    );
}
