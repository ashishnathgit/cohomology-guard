//! cohomology-guard: real-time sensor anomaly detection with cellular sheaves.

mod sheaf;
mod simulator;

use colored::Colorize;
use sheaf::{CellularSheaf, ConsistencyReport};
use simulator::{SimConfig, TelemetryFrame};

/// Dirichlet energy (°C²) above which the network is declared obstructed.
const ENERGY_THRESHOLD: f64 = 0.05;
/// Edge residual (°C) above which an edge is highlighted.
const EDGE_THRESHOLD: f64 = 0.15;

#[tokio::main]
async fn main() {
    let config = SimConfig::default();
    let mut sheaf = simulator::build_ring_network(config.base_celsius);
    print_banner(&sheaf);

    let nodes: Vec<_> = sheaf.graph.node_indices().collect();
    let mut rx = simulator::spawn_telemetry(config);
    let mut alarms = 0u32;

    while let Some(frame) = rx.recv().await {
        for (node, raw) in nodes.iter().zip(&frame.raw_readings) {
            sheaf
                .set_raw_reading(*node, *raw)
                .expect("node indices come from the same graph");
        }
        let report = sheaf.evaluate_consistency();
        if report.is_obstructed(ENERGY_THRESHOLD) {
            alarms += 1;
        }
        print_tick(&sheaf, &frame, &report);
    }

    println!("{}", "═".repeat(72).bright_black());
    let summary = format!("Simulation complete: {alarms} obstructed tick(s) detected.");
    if alarms > 0 {
        println!("{}", summary.red().bold());
    } else {
        println!("{}", summary.green().bold());
    }
}

fn print_banner(sheaf: &CellularSheaf) {
    let delta = sheaf.compute_coboundary_matrix();
    let laplacian = sheaf.compute_sheaf_laplacian();
    println!("{}", "═".repeat(72).bright_black());
    println!("{}", "  COHOMOLOGY GUARD — Sheaf-Laplacian Sensor Anomaly Detector".cyan().bold());
    println!("{}", "═".repeat(72).bright_black());
    println!(
        "  Sensors: {}   Links: {}   δ₀: {}×{}   L_F: {}×{}",
        sheaf.graph.node_count(),
        sheaf.graph.edge_count(),
        delta.nrows(),
        delta.ncols(),
        laplacian.nrows(),
        laplacian.ncols()
    );
    for n in sheaf.graph.node_indices() {
        let s = sheaf.stalk(n);
        println!("    [{}] {:<8} {:<10} stalk dim {}", n.index(), s.sensor_id, s.sensor_type, s.dim);
    }
    println!("  Energy threshold: {ENERGY_THRESHOLD} °C²   Edge threshold: {EDGE_THRESHOLD} °C");
}

fn print_tick(sheaf: &CellularSheaf, frame: &TelemetryFrame, report: &ConsistencyReport) {
    println!("{}", "─".repeat(72).bright_black());
    let status = if report.is_obstructed(ENERGY_THRESHOLD) {
        "OBSTRUCTION (H¹ ≠ 0)".red().bold()
    } else {
        "CONSISTENT (global section)".green().bold()
    };
    let injected = if frame.fault_active {
        "  [fault injection active]".yellow().to_string()
    } else {
        String::new()
    };
    println!(
        "{}  true zone temp {:.2} °C  {}{}",
        format!("Tick {:>2}", frame.tick).bold(),
        frame.true_celsius,
        status,
        injected
    );

    // Readings
    let readings: Vec<String> = sheaf
        .graph
        .node_indices()
        .map(|n| {
            let s = sheaf.stalk(n);
            format!("S{}={:.2}{}", n.index(), s.raw(), s.sensor_type.unit())
        })
        .collect();
    println!("  Readings: {}", readings.join("  "));

    // Edge residual table
    println!("  {:<10} {:>14} {:>14} {:>12}", "Edge", "F_u·x_u (°C)", "F_v·x_v (°C)", "Residual");
    for r in &report.edge_residuals {
        let (u, v) = (sheaf.stalk(r.source), sheaf.stalk(r.target));
        let label = format!("e{}: {}→{}", r.edge.index(), r.source.index(), r.target.index());
        let res = format!("{:>12.4}", r.residual);
        let res = if r.residual > EDGE_THRESHOLD { res.red().bold() } else { res.green() };
        println!("  {:<10} {:>14.4} {:>14.4} {}", label, u.as_celsius(), v.as_celsius(), res);
    }

    let energy = format!("{:.6}", report.dirichlet_energy);
    let energy = if report.is_obstructed(ENERGY_THRESHOLD) { energy.red().bold() } else { energy.green() };
    println!(
        "  Total Dirichlet Energy ‖δ₀x‖² = {}   (xᵀLx = {:.6})",
        energy, report.laplacian_quadratic_form
    );

    match report.suspected_fault(ENERGY_THRESHOLD) {
        Some(top) => {
            let s = sheaf.stalk(top.node);
            let ranking: Vec<String> = report
                .node_ranking
                .iter()
                .take(3)
                .map(|ns| format!("{}:{:.3}", ns.node.index(), ns.score))
                .collect();
            println!(
                "  Suspected Faulty Node: {}  (incident residual sum {:.4}; top-3 {})",
                format!("#{} {} [{}]", top.node.index(), s.sensor_id, s.sensor_type).red().bold(),
                top.score,
                ranking.join(", ")
            );
        }
        None => println!("  Suspected Faulty Node: {}", "none".green()),
    }
}
