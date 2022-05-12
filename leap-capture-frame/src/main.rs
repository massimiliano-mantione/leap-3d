use leap_device::device::{LeapDevice, LeapMode};
use lib::image::{Rgb, RgbImage};

const BUFFER_COUNT: u32 = 4;

fn save_frame(dev: &LeapDevice) -> Result<(), Box<dyn std::error::Error>> {
    let img_w = dev.mode().x_res();
    let img_h = dev.mode().y_res();

    println!("IMG {} {}", img_w, img_h);

    let mut stream = dev.stream(BUFFER_COUNT)?;
    let mut img = RgbImage::new(img_w, img_h);
    let (img_w, img_h) = (img_w as usize, img_h as usize);
    let (buf, meta) = stream.next_frame()?;
    println!(
        "bytes: {} frame: {} time: {}",
        buf.len(),
        meta.sequence,
        meta.timestamp
    );
    for (y, line) in buf.chunks_exact(img_w * 2).enumerate().take(img_h) {
        for (x, values) in line.chunks_exact(2).enumerate().take(img_w) {
            let left = values[0];
            let right = values[1];
            let p = Rgb([left, right, 0]);
            img.put_pixel(x as u32, y as u32, p);
        }
    }
    img.save("frame.png")?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut dev = LeapDevice::setup(LeapMode::m640x480())?;
    save_frame(&mut dev)?;
    Ok(())
}
