package de.fhdw.knn.visualization;

import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.SymbolAxis;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.PaintScale;
import org.jfree.chart.renderer.xy.XYBlockRenderer;
import org.jfree.data.xy.DefaultXYZDataset;

import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;
import java.util.List;

public class HeatmapView extends JFrame {

    private final JTabbedPane tabs;
    private static final double THRESHOLD = 0.1;

    public HeatmapView() {
        super("Heatmap – Network View");

        this.tabs = new JTabbedPane();
        this.tabs.setTabLayoutPolicy(JTabbedPane.SCROLL_TAB_LAYOUT);

        this.setContentPane(tabs);
        this.setSize(1200, 900);
        this.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        this.setVisible(true);
    }

    public void showSingleMatrix(String title, HeatmapData matrix) {
        this.addEpoch(title, matrix);

        this.setTitle(title);
        this.setDefaultCloseOperation(javax.swing.JFrame.DISPOSE_ON_CLOSE);
        this.pack();
        this.setVisible(true);
    }

    // Methode nach einer Epoche auf, um einen neuen Tab hinzuzufügen
    public void addEpoch(String tabTitle, HeatmapData data) {
        SwingUtilities.invokeLater(() -> { // invokeLater, da Swing nicht Thread-safe ist
            JFreeChart chart = buildChart(data, tabTitle);
            ChartPanel panel = new ChartPanel(chart);

            panel.setMouseWheelEnabled(true);
            panel.setDomainZoomable(true);
            panel.setRangeZoomable(true);

            panel.getChart().getXYPlot().setDomainPannable(true);
            panel.getChart().getXYPlot().setRangePannable(true);

            JScrollPane scrollPane = new JScrollPane(panel);
            this.tabs.addTab(tabTitle, scrollPane);
            this.tabs.setSelectedIndex(this.tabs.getTabCount() - 1);
        });
    }

    private JFreeChart buildChart(HeatmapData data, String title) {
        double[][] m = data.buildFullWeightMatrix();
        String[] labels = data.getNeuronLabels();

        int N = m.length;

        DefaultXYZDataset dataset = new DefaultXYZDataset();
        List<Double> xList = new ArrayList<>();
        List<Double> yList = new ArrayList<>();
        List<Double> zList = new ArrayList<>();

        double min = -1.0;
        double max = 1.0;

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                double val = m[i][j];
                xList.add((double) i);
                yList.add((double) j);
                zList.add(val);
            }
        }

        double[][] dataArr = new double[3][xList.size()];
        for (int i = 0; i < xList.size(); i++) {
            dataArr[0][i] = xList.get(i);
            dataArr[1][i] = yList.get(i);
            dataArr[2][i] = zList.get(i);
        }
        dataset.addSeries("Weights", dataArr);

        // Renderer
        XYBlockRenderer renderer = getXyBlockRenderer(min, max);
        XYPlot plot = getXyPlot(labels, dataset, renderer);

        JFreeChart chart = new JFreeChart(title, JFreeChart.DEFAULT_TITLE_FONT, plot, false);
        chart.setBackgroundPaint(Color.WHITE);

        return chart;
    }

    private static XYPlot getXyPlot(String[] labels, DefaultXYZDataset dataset, XYBlockRenderer renderer) {
        SymbolAxis xAxis = new SymbolAxis("Source Neuron (Layer i)", labels);
        SymbolAxis yAxis = new SymbolAxis("Target Neuron (Layer i+1)", labels);

        xAxis.setGridBandsVisible(false);
        yAxis.setGridBandsVisible(false);

        XYPlot plot = new XYPlot(dataset, xAxis, yAxis, renderer);
        plot.setBackgroundPaint(Color.WHITE);
        plot.setDomainGridlinesVisible(false);
        plot.setRangeGridlinesVisible(false);
        return plot;
    }

    private static XYBlockRenderer getXyBlockRenderer(double min, double max) {
        XYBlockRenderer renderer = new XYBlockRenderer();
        renderer.setBlockWidth(1.0);
        renderer.setBlockHeight(1.0);
        renderer.setBlockAnchor(org.jfree.chart.ui.RectangleAnchor.CENTER);

        renderer.setPaintScale(new PaintScale() {
            @Override
            public double getLowerBound() {
                return min;
            }

            @Override
            public double getUpperBound() {
                return max;
            }

            @Override
            public Paint getPaint(double value) {
                if (Double.isNaN(value)) return new Color(240, 240, 240); // Helles Grau = "Keine Verbindung"

                if (Math.abs(value) < THRESHOLD) {
                    return Color.WHITE;
                }

                // Grün (negativ) -> Weiß (0) -> Rot (positiv)
                double ratio = Math.min(Math.abs(value), 1.0);
                if (value > 0) {
                    return new Color(1.0f, (float) (1 - ratio), (float) (1 - ratio)); // Rot
                } else {
                    return new Color((float) (1 - ratio), 1.0f, (float) (1 - ratio)); // Grün
                }
            }
        });
        return renderer;
    }
}