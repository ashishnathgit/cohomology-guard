//! Cellular sheaves on sensor-network graphs.
//!
//! Each sensor (graph vertex) carries a *stalk* holding its reading in
//! homogeneous coordinates `x_v = [t, 1]^T`, where `t` is the raw value in the
//! sensor's native unit. The homogeneous `1` lets affine unit conversions
//! (e.g. °F → °C, which has an offset) act as **linear** restriction maps.
//!
//! Each link (graph edge `e = u → v`) carries a 1-dimensional edge stalk: the
//! shared comparison space, temperature in °C. The restriction maps
//! `F_{u ⊴ e}` and `F_{v ⊴ e}` are `1 × 2` matrices that convert each endpoint's
//! reading into Celsius.
//!
//! The coboundary `δ₀ : C⁰ → C¹` acts as `(δ₀ x)_e = F_{v⊴e} x_v − F_{u⊴e} x_u`,
//! the sheaf Laplacian is `L = δ₀ᵀ δ₀`, and the Dirichlet energy
//! `E(x) = xᵀ L x = ‖δ₀ x‖²` vanishes exactly when `x` is a global section,
//! i.e. when every pair of linked sensors agrees after unit conversion.

use nalgebra::{DMatrix, DVector};
use petgraph::graph::{EdgeIndex, Graph, NodeIndex};
use petgraph::visit::EdgeRef;
use std::fmt;

/// Physical unit a sensor reports in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SensorType {
    Celsius,
    Fahrenheit,
    Kelvin,
}

impl SensorType {
    /// Short unit label.
    pub fn unit(self) -> &'static str {
        match self {
            SensorType::Celsius => "°C",
            SensorType::Fahrenheit => "°F",
            SensorType::Kelvin => "K",
        }
    }

    /// Restriction map (1 × 2) sending a homogeneous native reading `[t, 1]`
    /// to Celsius on the edge stalk.
    pub fn to_celsius_map(self) -> DMatrix<f64> {
        match self {
            SensorType::Celsius => DMatrix::from_row_slice(1, 2, &[1.0, 0.0]),
            SensorType::Fahrenheit => {
                DMatrix::from_row_slice(1, 2, &[5.0 / 9.0, -32.0 * 5.0 / 9.0])
            }
            SensorType::Kelvin => DMatrix::from_row_slice(1, 2, &[1.0, -273.15]),
        }
    }

    /// Convert a Celsius temperature to this sensor's native unit.
    pub fn from_celsius(self, celsius: f64) -> f64 {
        match self {
            SensorType::Celsius => celsius,
            SensorType::Fahrenheit => celsius * 9.0 / 5.0 + 32.0,
            SensorType::Kelvin => celsius + 273.15,
        }
    }
}

impl fmt::Display for SensorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            SensorType::Celsius => "Celsius",
            SensorType::Fahrenheit => "Fahrenheit",
            SensorType::Kelvin => "Kelvin",
        };
        f.write_str(name)
    }
}

/// Vertex stalk: one sensor and its current reading.
#[derive(Debug, Clone)]
pub struct NodeStalk {
    pub sensor_id: String,
    pub sensor_type: SensorType,
    /// Homogeneous reading `[t_native, 1]`.
    pub reading: DVector<f64>,
    /// Stalk dimension (2 for homogeneous scalar readings).
    pub dim: usize,
}

impl NodeStalk {
    pub fn new(sensor_id: impl Into<String>, sensor_type: SensorType, raw: f64) -> Self {
        Self {
            sensor_id: sensor_id.into(),
            sensor_type,
            reading: DVector::from_vec(vec![raw, 1.0]),
            dim: 2,
        }
    }

    /// Raw reading in the sensor's native unit.
    pub fn raw(&self) -> f64 {
        self.reading[0]
    }

    pub fn set_raw(&mut self, raw: f64) {
        self.reading[0] = raw;
    }

    /// Reading pushed into the Celsius comparison space.
    pub fn as_celsius(&self) -> f64 {
        (self.sensor_type.to_celsius_map() * &self.reading)[0]
    }
}

/// Edge stalk data: restriction maps from both endpoints into the edge space.
#[derive(Debug, Clone)]
pub struct EdgeRestriction {
    /// `F_{u ⊴ e}` for the edge's source vertex `u`.
    pub source_map: DMatrix<f64>,
    /// `F_{v ⊴ e}` for the edge's target vertex `v`.
    pub target_map: DMatrix<f64>,
    /// Dimension of the edge stalk.
    pub edge_dim: usize,
}

impl EdgeRestriction {
    pub fn new(source_map: DMatrix<f64>, target_map: DMatrix<f64>) -> Result<Self, SheafError> {
        if source_map.nrows() != target_map.nrows() {
            return Err(SheafError::EdgeDimMismatch {
                source_rows: source_map.nrows(),
                target_rows: target_map.nrows(),
            });
        }
        Ok(Self {
            edge_dim: source_map.nrows(),
            source_map,
            target_map,
        })
    }

    /// Unit-conversion restriction maps for a pair of temperature sensors.
    pub fn unit_conversion(source: &NodeStalk, target: &NodeStalk) -> Self {
        Self::new(
            source.sensor_type.to_celsius_map(),
            target.sensor_type.to_celsius_map(),
        )
        .expect("Celsius maps are all 1 x 2")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SheafError {
    EdgeDimMismatch { source_rows: usize, target_rows: usize },
    StalkDimMismatch { node: usize, expected: usize, found: usize },
    UnknownNode(usize),
}

impl fmt::Display for SheafError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SheafError::EdgeDimMismatch { source_rows, target_rows } => write!(
                f,
                "restriction maps disagree on edge stalk dimension ({source_rows} vs {target_rows})"
            ),
            SheafError::StalkDimMismatch { node, expected, found } => write!(
                f,
                "restriction map for node {node} expects {found} columns but stalk has dim {expected}"
            ),
            SheafError::UnknownNode(n) => write!(f, "node {n} does not exist"),
        }
    }
}

impl std::error::Error for SheafError {}

/// Residual of one edge: `‖F_{v⊴e} x_v − F_{u⊴e} x_u‖`.
#[derive(Debug, Clone)]
pub struct EdgeResidual {
    pub edge: EdgeIndex,
    pub source: NodeIndex,
    pub target: NodeIndex,
    pub residual: f64,
}

/// Fault-isolation score of one node: sum of incident edge residuals.
#[derive(Debug, Clone)]
pub struct NodeScore {
    pub node: NodeIndex,
    pub score: f64,
}

/// Result of a consistency evaluation.
#[derive(Debug, Clone)]
pub struct ConsistencyReport {
    pub edge_residuals: Vec<EdgeResidual>,
    /// `‖δ₀ x‖²`.
    pub dirichlet_energy: f64,
    /// `xᵀ L x` (should equal `dirichlet_energy` up to round-off).
    pub laplacian_quadratic_form: f64,
    /// Nodes ranked by descending score.
    pub node_ranking: Vec<NodeScore>,
}

impl ConsistencyReport {
    /// Whether the energy exceeds `threshold` (no global section near `x`).
    pub fn is_obstructed(&self, threshold: f64) -> bool {
        self.dirichlet_energy > threshold
    }

    /// Most suspicious node, only reported if the network is obstructed.
    pub fn suspected_fault(&self, threshold: f64) -> Option<&NodeScore> {
        if self.is_obstructed(threshold) {
            self.node_ranking.first()
        } else {
            None
        }
    }
}

/// A cellular sheaf over a directed sensor graph (edge orientation fixes the
/// sign convention of the coboundary).
#[derive(Debug, Clone, Default)]
pub struct CellularSheaf {
    pub graph: Graph<NodeStalk, EdgeRestriction>,
}

impl CellularSheaf {
    pub fn new() -> Self {
        Self { graph: Graph::new() }
    }

    pub fn add_sensor(&mut self, stalk: NodeStalk) -> NodeIndex {
        self.graph.add_node(stalk)
    }

    /// Link two sensors with explicit restriction maps.
    pub fn add_restriction(
        &mut self,
        u: NodeIndex,
        v: NodeIndex,
        restriction: EdgeRestriction,
    ) -> Result<EdgeIndex, SheafError> {
        let su = self.graph.node_weight(u).ok_or(SheafError::UnknownNode(u.index()))?;
        let sv = self.graph.node_weight(v).ok_or(SheafError::UnknownNode(v.index()))?;
        for (node, stalk, map) in [
            (u, su, &restriction.source_map),
            (v, sv, &restriction.target_map),
        ] {
            if map.ncols() != stalk.dim {
                return Err(SheafError::StalkDimMismatch {
                    node: node.index(),
                    expected: stalk.dim,
                    found: map.ncols(),
                });
            }
        }
        Ok(self.graph.add_edge(u, v, restriction))
    }

    /// Link two sensors using unit-conversion restriction maps into °C.
    pub fn connect(&mut self, u: NodeIndex, v: NodeIndex) -> Result<EdgeIndex, SheafError> {
        let su = self.graph.node_weight(u).ok_or(SheafError::UnknownNode(u.index()))?;
        let sv = self.graph.node_weight(v).ok_or(SheafError::UnknownNode(v.index()))?;
        let r = EdgeRestriction::unit_conversion(su, sv);
        self.add_restriction(u, v, r)
    }

    pub fn stalk(&self, n: NodeIndex) -> &NodeStalk {
        &self.graph[n]
    }

    pub fn set_raw_reading(&mut self, n: NodeIndex, raw: f64) -> Result<(), SheafError> {
        self.graph
            .node_weight_mut(n)
            .ok_or(SheafError::UnknownNode(n.index()))?
            .set_raw(raw);
        Ok(())
    }

    /// Starting row/column offset of each node block in `C⁰`.
    fn node_offsets(&self) -> (Vec<usize>, usize) {
        let mut offsets = Vec::with_capacity(self.graph.node_count());
        let mut acc = 0;
        for n in self.graph.node_indices() {
            offsets.push(acc);
            acc += self.graph[n].dim;
        }
        (offsets, acc)
    }

    /// Starting row offset of each edge block in `C¹`.
    fn edge_offsets(&self) -> (Vec<usize>, usize) {
        let mut offsets = Vec::with_capacity(self.graph.edge_count());
        let mut acc = 0;
        for e in self.graph.edge_indices() {
            offsets.push(acc);
            acc += self.graph[e].edge_dim;
        }
        (offsets, acc)
    }

    /// Concatenated 0-cochain `x ∈ C⁰ = ⊕_v F(v)`.
    pub fn cochain(&self) -> DVector<f64> {
        let (offsets, total) = self.node_offsets();
        let mut x = DVector::zeros(total);
        for n in self.graph.node_indices() {
            let s = &self.graph[n];
            x.rows_mut(offsets[n.index()], s.dim).copy_from(&s.reading);
        }
        x
    }

    /// Coboundary operator `δ₀ : C⁰ → C¹`. Block `(e, v)` is `+F_{v⊴e}`,
    /// block `(e, u)` is `−F_{u⊴e}` for an oriented edge `e = u → v`.
    pub fn compute_coboundary_matrix(&self) -> DMatrix<f64> {
        let (node_off, n_total) = self.node_offsets();
        let (edge_off, e_total) = self.edge_offsets();
        let mut delta = DMatrix::zeros(e_total, n_total);
        for edge in self.graph.edge_references() {
            let r = edge.weight();
            let row = edge_off[edge.id().index()];
            let (u, v) = (edge.source().index(), edge.target().index());
            let mut src_block = delta.view_mut((row, node_off[u]), (r.edge_dim, r.source_map.ncols()));
            src_block -= &r.source_map;
            let mut tgt_block = delta.view_mut((row, node_off[v]), (r.edge_dim, r.target_map.ncols()));
            tgt_block += &r.target_map;
        }
        delta
    }

    /// Sheaf Laplacian `L = δ₀ᵀ δ₀`.
    pub fn compute_sheaf_laplacian(&self) -> DMatrix<f64> {
        let delta = self.compute_coboundary_matrix();
        delta.transpose() * delta
    }

    /// Compute edge residuals, Dirichlet energy and a fault ranking.
    pub fn evaluate_consistency(&self) -> ConsistencyReport {
        let x = self.cochain();
        let delta = self.compute_coboundary_matrix();
        let dx = &delta * &x;
        let dirichlet_energy = dx.norm_squared();
        let laplacian = delta.transpose() * &delta;
        let laplacian_quadratic_form = x.dot(&(&laplacian * &x));

        let (edge_off, _) = self.edge_offsets();
        let mut scores = vec![0.0_f64; self.graph.node_count()];
        let edge_residuals: Vec<EdgeResidual> = self
            .graph
            .edge_references()
            .map(|edge| {
                let r = edge.weight();
                let residual = dx.rows(edge_off[edge.id().index()], r.edge_dim).norm();
                scores[edge.source().index()] += residual;
                scores[edge.target().index()] += residual;
                EdgeResidual {
                    edge: edge.id(),
                    source: edge.source(),
                    target: edge.target(),
                    residual,
                }
            })
            .collect();

        let mut node_ranking: Vec<NodeScore> = self
            .graph
            .node_indices()
            .map(|node| NodeScore { node, score: scores[node.index()] })
            .collect();
        node_ranking.sort_by(|a, b| b.score.total_cmp(&a.score));

        ConsistencyReport {
            edge_residuals,
            dirichlet_energy,
            laplacian_quadratic_form,
            node_ranking,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TYPES: [SensorType; 6] = [
        SensorType::Celsius,
        SensorType::Fahrenheit,
        SensorType::Kelvin,
        SensorType::Celsius,
        SensorType::Fahrenheit,
        SensorType::Kelvin,
    ];

    /// 6-node directed ring, all sensors observing `celsius`.
    fn ring(celsius: f64) -> (CellularSheaf, Vec<NodeIndex>) {
        let mut sheaf = CellularSheaf::new();
        let nodes: Vec<_> = TYPES
            .iter()
            .enumerate()
            .map(|(i, &t)| sheaf.add_sensor(NodeStalk::new(format!("S{i}"), t, t.from_celsius(celsius))))
            .collect();
        for i in 0..nodes.len() {
            sheaf.connect(nodes[i], nodes[(i + 1) % nodes.len()]).unwrap();
        }
        (sheaf, nodes)
    }

    #[test]
    fn unit_conversions_round_trip() {
        for t in TYPES {
            let s = NodeStalk::new("x", t, t.from_celsius(37.5));
            assert!((s.as_celsius() - 37.5).abs() < 1e-12, "{t}");
        }
    }

    #[test]
    fn coboundary_and_laplacian_shapes_and_properties() {
        let (sheaf, _) = ring(20.0);
        let delta = sheaf.compute_coboundary_matrix();
        assert_eq!(delta.shape(), (6, 12));
        let l = sheaf.compute_sheaf_laplacian();
        assert_eq!(l.shape(), (12, 12));
        assert!((&l - l.transpose()).abs().max() < 1e-12, "L must be symmetric");
        let eig = l.symmetric_eigenvalues();
        assert!(eig.iter().all(|&ev| ev > -1e-9), "L must be PSD: {eig}");
    }

    #[test]
    fn coboundary_single_edge_blocks() {
        let mut sheaf = CellularSheaf::new();
        let a = sheaf.add_sensor(NodeStalk::new("a", SensorType::Celsius, 0.0));
        let b = sheaf.add_sensor(NodeStalk::new("b", SensorType::Kelvin, 273.15));
        sheaf.connect(a, b).unwrap();
        let d = sheaf.compute_coboundary_matrix();
        let expected = DMatrix::from_row_slice(1, 4, &[-1.0, 0.0, 1.0, -273.15]);
        assert!((d - expected).abs().max() < 1e-12);
    }

    #[test]
    fn rejects_mismatched_restriction_maps() {
        let mut sheaf = CellularSheaf::new();
        let a = sheaf.add_sensor(NodeStalk::new("a", SensorType::Celsius, 0.0));
        let b = sheaf.add_sensor(NodeStalk::new("b", SensorType::Celsius, 0.0));
        let bad = EdgeRestriction::new(DMatrix::zeros(1, 3), DMatrix::zeros(1, 2)).unwrap();
        assert!(matches!(
            sheaf.add_restriction(a, b, bad),
            Err(SheafError::StalkDimMismatch { .. })
        ));
        assert!(EdgeRestriction::new(DMatrix::zeros(1, 2), DMatrix::zeros(2, 2)).is_err());
    }

    #[test]
    fn consistent_global_section_has_zero_energy_across_units() {
        for c in [-40.0, 0.0, 21.7, 100.0] {
            let (sheaf, _) = ring(c);
            let report = sheaf.evaluate_consistency();
            assert!(
                report.dirichlet_energy.abs() < 1e-12,
                "energy {} at {c}°C",
                report.dirichlet_energy
            );
            assert!(report.laplacian_quadratic_form.abs() < 1e-8);
            assert!(report.edge_residuals.iter().all(|r| r.residual < 1e-9));
            assert!(!report.is_obstructed(1e-6));
            assert!(report.suspected_fault(1e-6).is_none());
        }
    }

    #[test]
    fn localized_fault_spikes_energy_and_is_isolated() {
        let (mut sheaf, nodes) = ring(22.0);
        let baseline = sheaf.evaluate_consistency().dirichlet_energy;

        // Spoof sensor 4 (Fahrenheit) by +30 °F.
        let faulty = nodes[4];
        let raw = sheaf.stalk(faulty).raw();
        sheaf.set_raw_reading(faulty, raw + 30.0).unwrap();

        let report = sheaf.evaluate_consistency();
        let expected_residual: f64 = 30.0 * 5.0 / 9.0;
        // Two incident edges each carry the full residual.
        assert!((report.dirichlet_energy - 2.0 * expected_residual.powi(2)).abs() < 1e-9);
        assert!(report.dirichlet_energy > baseline + 100.0);
        assert!((report.dirichlet_energy - report.laplacian_quadratic_form).abs() < 1e-6);
        let top = report.suspected_fault(1.0).expect("fault must be detected");
        assert_eq!(top.node, faulty);
        assert!(top.score > report.node_ranking[1].score + 1.0);
    }

    #[test]
    fn every_single_node_fault_is_localized() {
        for idx in 0..TYPES.len() {
            let (mut sheaf, nodes) = ring(18.0);
            let raw = sheaf.stalk(nodes[idx]).raw();
            sheaf.set_raw_reading(nodes[idx], raw - 12.0).unwrap();
            let report = sheaf.evaluate_consistency();
            assert_eq!(report.suspected_fault(1.0).map(|s| s.node), Some(nodes[idx]));
        }
    }
}
