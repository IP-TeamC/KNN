package de.fhdw.knn.util;

import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYBlockRenderer;
import org.jfree.chart.ui.ApplicationFrame;
import org.jfree.chart.ui.UIUtils;
import org.jfree.data.xy.DefaultXYZDataset;

import java.awt.*;

public class Heatmap extends ApplicationFrame {

    private final double[][] matrix;

    public Heatmap(String title, double[][] matrix) {
        super(title);
        this.matrix = matrix;

        int rows = matrix.length;
        int cols = matrix[0].length;

        DefaultXYZDataset dataset = new DefaultXYZDataset();
        double[] xValues = new double[rows * cols];
        double[] yValues = new double[rows * cols];
        double[] zValues = new double[rows * cols];

        int index = 0;
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                xValues[index] = j;
                yValues[index] = i;
                zValues[index] = matrix[i][j];
                min = Math.min(min, matrix[i][j]);
                max = Math.max(max, matrix[i][j]);
                index++;
            }
        }

        double[][] data = new double[][]{xValues, yValues, zValues};
        dataset.addSeries("Weights", data);

        ChartPanel chartPanel = getChartPanel(min, max, dataset);
        setContentPane(chartPanel);
    }

    private static ChartPanel getChartPanel(double min, double max, DefaultXYZDataset dataset) {
        XYBlockRenderer renderer = new XYBlockRenderer();
        renderer.setBlockWidth(1.0);
        renderer.setBlockHeight(1.0);

        SmoothWeightPaintScale paintScale = new SmoothWeightPaintScale(min, max);
        renderer.setPaintScale(paintScale);

        NumberAxis xAxis = new NumberAxis("Neuron Out");
        xAxis.setLowerMargin(0);
        xAxis.setUpperMargin(0);
        NumberAxis yAxis = new NumberAxis("Neuron In");
        yAxis.setLowerMargin(0);
        yAxis.setUpperMargin(0);

        XYPlot plot = new XYPlot(dataset, xAxis, yAxis, renderer);

        JFreeChart chart = new JFreeChart("Neuronal Weights Heatmap", JFreeChart.DEFAULT_TITLE_FONT, plot, false);

        ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new Dimension(800, 600));
        return chartPanel;
    }

    public void drawHeatmap(){
        Heatmap demo = new Heatmap("Heatmap Demo", this.matrix);
        demo.pack();
        UIUtils.centerFrameOnScreen(demo); // Fenster zentrieren
        demo.setVisible(true);
    }
}
