use eframe::egui;

mod gui;
mod optimizer;
mod parser;

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1000.0, 700.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Градиентный спуск",
        options,
        Box::new(|_cc| Ok(Box::new(gui::GradientDescentApp::default()))),
    )
}
