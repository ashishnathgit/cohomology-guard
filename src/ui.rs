//! Interactive ratatui terminal dashboard for cohomology-guard.

use std::collections::VecDeque;

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, Cell, Gauge, List, ListItem, Paragraph, Row, Sparkline, Table,
};
use ratatui::Frame;

use crate::sheaf::{CellularSheaf, ConsistencyReport};

/// Dirichlet energy (°C²) above which the network is declared obstructed.
pub const ENERGY_THRESHOLD: f64 = 0.05;
/// Edge residual (°C) above which an edge is highlighted.
pub const EDGE_THRESHOLD: f64 = 0.15;
/// Number of ticks of Dirichlet energy history kept for the sparkline.
pub const HISTORY_LEN: usize = 60;

/// Mutable UI-side state that persists across ticks (history buffers etc).
pub struct DashboardState {
    pub energy_history: VecDeque<u64>,
    pub tick: u64,
}

impl DashboardState {
    pub fn new() -> Self {
        Self {
            energy_history: VecDeque::with_capacity(HISTORY_LEN),
            tick: 0,
        }
    }

    /// Record one tick's Dirichlet energy, scaling it into a sparkline-friendly
    /// integer range (energy is tiny normally, large under fault).
    pub fn push_energy(&mut self, energy: f64) {
        self.tick += 1;
        // Scale: 1 unit == 0.01 °C^2, clamp so a single spike doesn't flatten
        // the rest of the chart.
        let scaled = (energy * 100.0).clamp(0.0, 10_000.0) as u64;
        if self.energy_history.len() == HISTORY_LEN {
            self.energy_history.pop_front();
        }
        self.energy_history.push_back(scaled);
    }
}

/// Render the full dashboard for one frame.
pub fn draw(
    frame: &mut Frame,
    sheaf: &CellularSheaf,
    report: &ConsistencyReport,
    state: &DashboardState,
) {
    let area = frame.area();

    let root = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(10),
            Constraint::Length(3),
        ])
        .split(area);

    draw_header(frame, root[0], report);

    let body = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(root[1]);

    let top_row = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(body[0]);

    let bottom_row = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(body[1]);

    draw_sensor_table(frame, top_row[0], sheaf, report);
    draw_energy_sparkline(frame, top_row[1], state);
    draw_edge_residuals(frame, bottom_row[0], sheaf, report);
    draw_controls(frame, bottom_row[1], sheaf, report);

    draw_footer(frame, root[2]);
}

fn draw_header(frame: &mut Frame, area: Rect, report: &ConsistencyReport) {
    let obstructed = report.is_obstructed(ENERGY_THRESHOLD);
    let (badge, style) = if obstructed {
        (
            "[OBSTRUCTION DETECTED: H^1 != 0]",
            Style::default()
                .fg(Color::Red)
                .add_modifier(Modifier::BOLD | Modifier::REVERSED),
        )
    } else {
        (
            "[HEALTHY: H^0 GLOBAL SECTION]",
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD | Modifier::REVERSED),
        )
    };

    let energy_text = format!(
        "  Total Dirichlet Energy E(x) = {:.6} °C²  (x^T L x = {:.6})  ",
        report.dirichlet_energy, report.laplacian_quadratic_form
    );
    let energy_style = if obstructed {
        Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::Green)
    };

    let line = Line::from(vec![
        Span::styled(format!(" {badge} "), style),
        Span::styled(energy_text, energy_style),
    ]);

    let para = Paragraph::new(line)
        .block(Block::default().borders(Borders::ALL).title(" Cohomology Guard — Sheaf Laplacian Monitor "));
    frame.render_widget(para, area);
}

fn draw_sensor_table(frame: &mut Frame, area: Rect, sheaf: &CellularSheaf, report: &ConsistencyReport) {
    let obstructed = report.is_obstructed(ENERGY_THRESHOLD);
    let top_suspect = report.suspected_fault(ENERGY_THRESHOLD).map(|s| s.node);

    let header = Row::new(vec!["ID", "Unit", "Raw", "Normalized (°C)", "Suspicion", "Status"])
        .style(Style::default().add_modifier(Modifier::BOLD));

    let rows: Vec<Row> = sheaf
        .graph
        .node_indices()
        .map(|n| {
            let s = sheaf.stalk(n);
            let score = report
                .node_ranking
                .iter()
                .find(|ns| ns.node == n)
                .map(|ns| ns.score)
                .unwrap_or(0.0);
            let is_top_suspect = obstructed && top_suspect == Some(n);
            let status = if s.is_faulty { "FAULTY" } else { "ok" };

            let row_style = if is_top_suspect {
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
            } else if s.is_faulty {
                Style::default().fg(Color::Yellow)
            } else {
                Style::default()
            };

            Row::new(vec![
                Cell::from(format!("S{}", n.index())),
                Cell::from(s.sensor_type.unit()),
                Cell::from(format!("{:.3}", s.raw())),
                Cell::from(format!("{:.3}", s.as_celsius())),
                Cell::from(format!("{:.3}", score)),
                Cell::from(status),
            ])
            .style(row_style)
        })
        .collect();

    let widths = [
        Constraint::Length(4),
        Constraint::Length(6),
        Constraint::Length(10),
        Constraint::Length(16),
        Constraint::Length(11),
        Constraint::Length(8),
    ];

    let table = Table::new(rows, widths)
        .header(header)
        .block(Block::default().borders(Borders::ALL).title(" Sensor Stalks "));

    frame.render_widget(table, area);
}

fn draw_energy_sparkline(frame: &mut Frame, area: Rect, state: &DashboardState) {
    let data: Vec<u64> = state.energy_history.iter().copied().collect();
    let sparkline = Sparkline::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!(" Dirichlet Energy (last {HISTORY_LEN} ticks) ")),
        )
        .data(&data)
        .style(Style::default().fg(Color::Cyan))
        .max(2000);
    frame.render_widget(sparkline, area);
}

fn draw_edge_residuals(frame: &mut Frame, area: Rect, sheaf: &CellularSheaf, report: &ConsistencyReport) {
    let block = Block::default().borders(Borders::ALL).title(" Edge Coboundary Residuals ");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let n = report.edge_residuals.len().max(1);
    let constraints: Vec<Constraint> = (0..n).map(|_| Constraint::Length(1)).collect();
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(inner);

    for (row, r) in rows.iter().zip(report.edge_residuals.iter()) {
        let u = sheaf.stalk(r.source);
        let v = sheaf.stalk(r.target);
        let label = format!("e{}: S{} <-> S{}", r.edge.index(), u.sensor_id, v.sensor_id);
        let ratio = (r.residual / 5.0).clamp(0.0, 1.0);
        let color = if r.residual > EDGE_THRESHOLD { Color::Red } else { Color::Green };
        let gauge = Gauge::default()
            .label(format!("{label}  {:.3}°C", r.residual))
            .ratio(ratio)
            .gauge_style(Style::default().fg(color));
        frame.render_widget(gauge, *row);
    }
}

fn draw_controls(frame: &mut Frame, area: Rect, sheaf: &CellularSheaf, report: &ConsistencyReport) {
    let mut lines: Vec<ListItem> = Vec::new();

    match report.suspected_fault(ENERGY_THRESHOLD) {
        Some(top) => {
            let s = sheaf.stalk(top.node);
            lines.push(ListItem::new(Line::from(Span::styled(
                format!("ALERT: S{} ({}) suspected faulty — score {:.3}", top.node.index(), s.sensor_id, top.score),
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            ))));
        }
        None => {
            lines.push(ListItem::new(Line::from(Span::styled(
                "All sensors agree — no obstruction detected.",
                Style::default().fg(Color::Green),
            ))));
        }
    }

    lines.push(ListItem::new(""));
    lines.push(ListItem::new(Line::from(Span::styled(
        "Keybindings:",
        Style::default().add_modifier(Modifier::BOLD),
    ))));
    lines.push(ListItem::new("  1-6  toggle spoofing fault on sensor 0-5 (+45°C offset)"));
    lines.push(ListItem::new("  r    reset all sensors to healthy state"));
    lines.push(ListItem::new("  q / Esc  quit"));

    let list = List::new(lines).block(Block::default().borders(Borders::ALL).title(" Controls & Diagnostics "));
    frame.render_widget(list, area);
}

fn draw_footer(frame: &mut Frame, area: Rect) {
    let text = Line::from(vec![
        Span::styled(" [1-6] ", Style::default().fg(Color::Yellow)),
        Span::raw("inject fault on sensor  "),
        Span::styled(" [r] ", Style::default().fg(Color::Yellow)),
        Span::raw("reset  "),
        Span::styled(" [q/Esc] ", Style::default().fg(Color::Yellow)),
        Span::raw("quit"),
    ]);
    let para = Paragraph::new(text).block(Block::default().borders(Borders::ALL));
    frame.render_widget(para, area);
}
