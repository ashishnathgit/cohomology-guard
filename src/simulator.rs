//! Async synthetic telemetry for a 6-sensor ring monitoring one thermal zone.

use crate::sheaf::{CellularSheaf, NodeStalk, SensorType};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Duration;
use tokio::sync::mpsc;

/// Sensor layout of the ring: mixed units observing the same zone.
pub const SENSOR_TYPES: [SensorType; 6] = [
    SensorType::Celsius,
    SensorType::Fahrenheit,
    SensorType::Kelvin,
    SensorType::Celsius,
    SensorType::Fahrenheit,
    SensorType::Kelvin,
];

/// Index of the sensor that gets compromised.
pub const FAULTY_SENSOR: usize = 4;
/// Tick at which the fault is injected.
pub const FAULT_TICK: u32 = 6;
pub const TOTAL_TICKS: u32 = 10;

#[derive(Debug, Clone)]
pub struct SimConfig {
    pub ticks: u32,
    pub tick_interval: Duration,
    pub base_celsius: f64,
    /// Peak-to-peak bound on per-sensor measurement noise (°C).
    pub noise_celsius: f64,
    /// Spoof offset added to the faulty sensor, in its native unit.
    pub fault_offset_native: f64,
    pub seed: u64,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            ticks: TOTAL_TICKS,
            tick_interval: Duration::from_millis(150),
            base_celsius: 22.5,
            noise_celsius: 0.02,
            fault_offset_native: 27.0,
            seed: 0xC0C0_4011,
        }
    }
}

/// One tick worth of readings, in each sensor's native unit.
#[derive(Debug, Clone)]
pub struct TelemetryFrame {
    pub tick: u32,
    pub true_celsius: f64,
    pub raw_readings: Vec<f64>,
    pub fault_active: bool,
}

/// Build the 6-node cyclic sheaf (0 → 1 → … → 5 → 0), plus one chord
/// (0 → 3) so the network has multiple independent loops.
pub fn build_ring_network(initial_celsius: f64) -> CellularSheaf {
    let mut sheaf = CellularSheaf::new();
    let nodes: Vec<_> = SENSOR_TYPES
        .iter()
        .enumerate()
        .map(|(i, &t)| {
            sheaf.add_sensor(NodeStalk::new(
                format!("S{i}-{}", t.unit()),
                t,
                t.from_celsius(initial_celsius),
            ))
        })
        .collect();
    let n = nodes.len();
    for i in 0..n {
        sheaf
            .connect(nodes[i], nodes[(i + 1) % n])
            .expect("ring edge dimensions are valid");
    }
    sheaf
        .connect(nodes[0], nodes[3])
        .expect("chord edge dimensions are valid");
    sheaf
}

/// Spawn the telemetry producer; frames arrive on the returned channel.
pub fn spawn_telemetry(config: SimConfig) -> mpsc::Receiver<TelemetryFrame> {
    let (tx, rx) = mpsc::channel(16);
    tokio::spawn(async move {
        let mut rng = StdRng::seed_from_u64(config.seed);
        let mut interval = tokio::time::interval(config.tick_interval);
        for tick in 1..=config.ticks {
            interval.tick().await;
            // Slow, smooth thermal drift shared by the whole zone.
            let true_celsius = config.base_celsius + 0.4 * (tick as f64 * 0.6).sin();
            let fault_active = tick >= FAULT_TICK;
            let half = config.noise_celsius / 2.0;
            let raw_readings = SENSOR_TYPES
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    let observed = true_celsius + rng.gen_range(-half..=half);
                    let mut raw = t.from_celsius(observed);
                    if fault_active && i == FAULTY_SENSOR {
                        // Spoofed value that grows slightly as the attack persists.
                        raw += config.fault_offset_native
                            + 1.5 * f64::from(tick - FAULT_TICK);
                    }
                    raw
                })
                .collect();
            let frame = TelemetryFrame { tick, true_celsius, raw_readings, fault_active };
            if tx.send(frame).await.is_err() {
                break; // consumer dropped
            }
        }
    });
    rx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn stream_is_healthy_then_faulty() {
        let cfg = SimConfig { tick_interval: Duration::from_millis(1), ..SimConfig::default() };
        let mut sheaf = build_ring_network(cfg.base_celsius);
        let nodes: Vec<_> = sheaf.graph.node_indices().collect();
        let mut rx = spawn_telemetry(cfg);
        let mut count = 0;
        while let Some(frame) = rx.recv().await {
            count += 1;
            for (n, raw) in nodes.iter().zip(&frame.raw_readings) {
                sheaf.set_raw_reading(*n, *raw).unwrap();
            }
            let report = sheaf.evaluate_consistency();
            if frame.tick < FAULT_TICK {
                assert!(report.dirichlet_energy < 0.01, "tick {}", frame.tick);
            } else {
                let top = report.suspected_fault(1.0).expect("obstruction expected");
                assert_eq!(top.node.index(), FAULTY_SENSOR);
            }
        }
        assert_eq!(count, TOTAL_TICKS);
    }
}
