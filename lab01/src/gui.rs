use crate::optimizer::{self, OptimizerResult};
use crate::parser::ParsedFunction;
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use nalgebra::DVector;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;

#[derive(PartialEq)]
enum OptimizerState {
    Idle,
    Running,
    Finished,
    Stopping,
}

pub struct GradientDescentApp {
    // Входные данные
    func_str: String,
    num_vars: usize,
    initial_point_str: String,
    initial_step: f64,
    step_decay: f64,
    step_increase: f64,
    tolerance: f64,
    max_iterations: usize,

    // Состояние
    state: OptimizerState,
    result: Option<OptimizerResult>,
    error_message: Option<String>,
    stop_flag: Arc<AtomicBool>,

    // Канал для получения результата из потока
    result_receiver: Option<Receiver<OptimizerResult>>,
    result_sender: Option<Sender<OptimizerResult>>,

    // Парсер
    parsed_func: Option<ParsedFunction>,
}

impl Default for GradientDescentApp {
    fn default() -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            func_str: "x1^2 + x2^2".to_string(),
            num_vars: 2,
            initial_point_str: "2, 2".to_string(),
            initial_step: 1.0,
            step_decay: 0.5,
            step_increase: 1.2,
            tolerance: 1e-6,
            max_iterations: 1000,
            state: OptimizerState::Idle,
            result: None,
            error_message: None,
            stop_flag: Arc::new(AtomicBool::new(false)),
            result_receiver: Some(rx),
            result_sender: Some(tx),
            parsed_func: None,
        }
    }
}

impl GradientDescentApp {
    fn parse_initial_point(&self) -> Option<DVector<f64>> {
        let parts: Vec<&str> = self.initial_point_str.split(',').collect();
        if parts.len() != self.num_vars {
            return None;
        }
        let mut vec = Vec::with_capacity(self.num_vars);
        for part in parts {
            match part.trim().parse::<f64>() {
                Ok(val) => vec.push(val),
                Err(_) => return None,
            }
        }
        Some(DVector::from_vec(vec))
    }

    fn start_optimization(&mut self) {
        self.error_message = None;
        self.stop_flag.store(false, Ordering::SeqCst);
        self.result = None;

        // Парсим функцию
        let parsed = match ParsedFunction::new(&self.func_str, self.num_vars) {
            Ok(func) => func,
            Err(e) => {
                self.error_message = Some(format!("Ошибка в функции: {}", e));
                self.state = OptimizerState::Idle;
                return;
            }
        };

        self.parsed_func = Some(parsed.clone());

        let start_point = match self.parse_initial_point() {
            Some(p) => p,
            None => {
                self.error_message =
                    Some("Ошибка в начальной точке. Используйте формат 'x1, x2'".to_string());
                self.state = OptimizerState::Idle;
                return;
            }
        };

        let sender = self.result_sender.take().expect("Sender already taken");
        let stop_flag_clone = self.stop_flag.clone();

        let initial_step = self.initial_step;
        let step_decay = self.step_decay;
        let step_increase = self.step_increase;
        let tolerance = self.tolerance;
        let max_iterations = self.max_iterations;

        self.state = OptimizerState::Running;

        let parsed_for_f = parsed.clone();
        let parsed_for_grad = parsed.clone();

        std::thread::spawn(move || {
            let f = move |x: &DVector<f64>| parsed_for_f.eval(x).unwrap();

            let grad = move |x: &DVector<f64>| parsed_for_grad.gradient(x, 1e-6).unwrap();

            let result = optimizer::gradient_descent(
                start_point,
                &f,
                &grad,
                initial_step,
                step_decay,
                step_increase,
                tolerance,
                max_iterations,
                stop_flag_clone,
            );

            let _ = sender.send(result);
        });
    }

    fn stop_optimization(&mut self) {
        self.stop_flag.store(true, Ordering::SeqCst);
        self.state = OptimizerState::Stopping;
    }

    fn check_for_result(&mut self) {
        if let Some(rx) = &self.result_receiver {
            if let Ok(res) = rx.try_recv() {
                self.result = Some(res);
                self.state = OptimizerState::Finished;
                let (tx, new_rx) = mpsc::channel();
                self.result_sender = Some(tx);
                self.result_receiver = Some(new_rx);
            }
        }
    }
}

impl eframe::App for GradientDescentApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.check_for_result();

        egui::SidePanel::left("control_panel")
            .resizable(true)
            .show(ctx, |ui| {
                ui.heading("Параметры");

                ui.horizontal(|ui| {
                    ui.label("Размерность (n):");
                    ui.add(egui::DragValue::new(&mut self.num_vars).range(1..=10));
                });

                ui.horizontal(|ui| {
                    ui.label("Функция f(x):");
                    ui.text_edit_singleline(&mut self.func_str);
                });

                ui.horizontal(|ui| {
                    ui.label("Начальная точка:");
                    ui.text_edit_singleline(&mut self.initial_point_str);
                });

                ui.separator();

                ui.horizontal(|ui| {
                    ui.label("Начальный шаг:");
                    ui.add(
                        egui::DragValue::new(&mut self.initial_step)
                            .speed(0.1)
                            .range(0.0..=10.0),
                    );
                });

                ui.horizontal(|ui| {
                    ui.label("Коэф. дробления:");
                    ui.add(
                        egui::DragValue::new(&mut self.step_decay)
                            .speed(0.05)
                            .range(0.1..=0.9),
                    );
                });

                ui.horizontal(|ui| {
                    ui.label("Коэф. увеличения:");
                    ui.add(
                        egui::DragValue::new(&mut self.step_increase)
                            .speed(0.1)
                            .range(1.0..=2.0),
                    );
                });

                ui.separator();

                ui.horizontal(|ui| {
                    ui.label("Точность:");
                    ui.add(
                        egui::DragValue::new(&mut self.tolerance)
                            .speed(1e-7)
                            .range(1e-12..=1.0),
                    );
                });

                ui.horizontal(|ui| {
                    ui.label("Макс. итераций:");
                    ui.add(
                        egui::DragValue::new(&mut self.max_iterations)
                            .speed(1)
                            .range(1..=10000),
                    );
                });

                ui.separator();

                match self.state {
                    OptimizerState::Idle => {
                        if ui.button("▶ Запуск").clicked() {
                            self.start_optimization();
                        }
                    }
                    OptimizerState::Running | OptimizerState::Stopping => {
                        let button_text = if self.state == OptimizerState::Running {
                            "⏸ Стоп"
                        } else {
                            "⏹ Остановка..."
                        };
                        if ui.button(button_text).clicked() {
                            self.stop_optimization();
                        }
                    }
                    OptimizerState::Finished => {
                        if ui.button("Сброс").clicked() {
                            self.result = None;
                            self.state = OptimizerState::Idle;
                        }
                    }
                }

                if let Some(err) = &self.error_message {
                    ui.colored_label(egui::Color32::RED, err);
                }
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Визуализация");

            if let Some(res) = &self.result {
                ui.label(format!(
                    "Результат: x* = [{}], f(x*) = {:.6}, итераций: {}",
                    res.x
                        .iter()
                        .map(|v| format!("{:.6}", v))
                        .collect::<Vec<_>>()
                        .join(", "),
                    res.f_x,
                    res.iterations
                ));
                if res.terminated_early {
                    ui.colored_label(egui::Color32::YELLOW, "Досрочно остановлено пользователем");
                }

                if self.num_vars == 2 && !res.history.is_empty() {
                    let points: PlotPoints = res.history.iter().map(|(x, y, _)| [*x, *y]).collect();
                    let line = Line::new(points).name("Путь спуска");
                    Plot::new("path_plot").view_aspect(1.0).show(ui, |plot_ui| {
                        plot_ui.line(line);
                    });
                } else if self.num_vars != 2 {
                    ui.label("График доступен только для 2D задач.");
                }
            } else {
                ui.label("Запустите оптимизацию для отображения результатов.");
            }
        });
    }
}
