//! cohomology-guard: real-time sensor anomaly detection with cellular sheaves.
//!
//! Interactive ratatui terminal dashboard. Six temperature sensors (mixed
//! Celsius/Fahrenheit/Kelvin) form a cellular sheaf over a ring+chords graph.
//! Each tick, synthetic Gaussian-noised telemetry is pushed through the
//! sheaf, the coboundary/Laplacian machinery computes the Dirichlet energy
//! and per-node suspicion scores, and the dashboard renders live.

mod sheaf;
mod ui;

use std::io;
use std::panic;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use petgraph::graph::NodeIndex;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

use sheaf::CellularSheaf;
use ui::DashboardState;

/// Zone ambient temperature that healthy sensors fluctuate around.
const BASE_CELSIUS: f64 = 22.5;
/// Peak-to-peak Gaussian-ish measurement noise bound, in °C.
const NOISE_CELSIUS: f64 = 0.06;
/// Offset injected on a spoofed/faulty sensor, in °C.
const FAULT_OFFSET_CELSIUS: f64 = 45.0;
/// Dashboard tick interval.
const TICK_RATE: Duration = Duration::from_millis(250);

fn main() -> io::Result<()> {
    install_panic_hook();

    let mut terminal = init_terminal()?;
    let result = run(&mut terminal);
    restore_terminal()?;
    result
}

/// Install a panic hook that restores the terminal before the default panic
/// handler prints its message, so a panic never leaves the terminal in raw
/// mode / alternate screen.
fn install_panic_hook() {
    let default_hook = panic::take_hook();
    panic::set_hook(Box::new(move |info| {
        let _ = restore_terminal();
        default_hook(info);
    }));
}

fn init_terminal() -> io::Result<Terminal<CrosstermBackend<io::Stdout>>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    Terminal::new(CrosstermBackend::new(stdout))
}

fn restore_terminal() -> io::Result<()> {
    disable_raw_mode()?;
    execute!(io::stdout(), LeaveAlternateScreen)?;
    Ok(())
}

/// Per-sensor fault state and the base (unfaulted) celsius reading each
/// sensor would report, so toggling faults on/off is reversible.
struct SimState {
    rng: StdRng,
    fault_active: [bool; 6],
    tick: u64,
}

fn run(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> io::Result<()> {
    let (mut sheaf, nodes) = CellularSheaf::demo_network(BASE_CELSIUS);
    // Sanity-check the Laplacian is well-formed before entering the loop.
    let laplacian = sheaf.compute_sheaf_laplacian();
    debug_assert_eq!(laplacian.nrows(), laplacian.ncols());
    let mut dashboard = DashboardState::new();
    let mut sim = SimState {
        rng: StdRng::seed_from_u64(0xC0C0_4011),
        fault_active: [false; 6],
        tick: 0,
    };

    let mut last_tick = Instant::now();

    loop {
        let timeout = TICK_RATE
            .checked_sub(last_tick.elapsed())
            .unwrap_or(Duration::ZERO);

        if event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => break,
                        KeyCode::Char('r') => {
                            for f in sim.fault_active.iter_mut() {
                                *f = false;
                            }
                            for n in &nodes {
                                sheaf.set_faulty(*n, false).ok();
                            }
                        }
                        KeyCode::Char(c @ '1'..='6') => {
                            let idx = (c as u8 - b'1') as usize;
                            sim.fault_active[idx] = !sim.fault_active[idx];
                            sheaf.set_faulty(nodes[idx], sim.fault_active[idx]).ok();
                        }
                        _ => {}
                    }
                }
            }
        }

        if last_tick.elapsed() >= TICK_RATE {
            tick_telemetry(&mut sheaf, &nodes, &mut sim);
            last_tick = Instant::now();
        }

        let report = sheaf.evaluate_consistency();
        dashboard.push_energy(report.dirichlet_energy);

        terminal.draw(|frame| ui::draw(frame, &sheaf, &report, &dashboard))?;
    }

    Ok(())
}

/// Advance synthetic telemetry by one tick: slow shared thermal drift plus
/// small per-sensor Gaussian-ish noise, with a fixed offset added to any
/// sensor whose fault flag is set.
fn tick_telemetry(sheaf: &mut CellularSheaf, nodes: &[NodeIndex], sim: &mut SimState) {
    sim.tick += 1;
    let drift = 0.4 * (sim.tick as f64 * 0.1).sin();
    let true_celsius = BASE_CELSIUS + drift;
    let half = NOISE_CELSIUS / 2.0;

    for (i, node) in nodes.iter().enumerate() {
        let noise = gaussian_noise(&mut sim.rng, half);
        let mut celsius = true_celsius + noise;
        if sim.fault_active[i] {
            celsius += FAULT_OFFSET_CELSIUS;
        }
        let sensor_type = sheaf.stalk(*node).sensor_type;
        let raw = sensor_type.from_celsius(celsius);
        sheaf.set_raw_reading(*node, raw).expect("node from same graph");
    }
}

/// Approximate standard-normal noise (sum of uniforms, central-limit-ish),
/// scaled so the peak-to-peak spread matches `half * 2`.
fn gaussian_noise(rng: &mut StdRng, half: f64) -> f64 {
    let sum: f64 = (0..4).map(|_| rng.gen_range(-half..=half)).sum();
    sum / 2.0
}
