package de.fhdw.knn.util;

import de.fhdw.knn.data.DataSet;
import de.fhdw.knn.network.Network;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.chart.ui.ApplicationFrame;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import java.awt.*;

public class Chart extends ApplicationFrame {

    public ChartPanel chartPanel;
    public JFreeChart chart;

    public static Chart draw(Network network, double[][] inputs, String inputLabel, int input, String outputLabel, int output) {
        XYSeriesCollection xys = new XYSeriesCollection();
        double[][] predictions = network.predict(inputs);
        XYSeries series = new XYSeries(inputLabel + " => " + outputLabel);
        for (int i = 0; i < predictions.length; i++) {
            series.add(inputs[i][input], predictions[i][output]);
        }
        xys.addSeries(series);
        return new Chart("Predictions", xys);
    }

    public Chart(String title, XYSeriesCollection xys) {
        super(title);
        JFreeChart chart = createChart(title, xys);
        chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new java.awt.Dimension(1000, 500));
        setContentPane(chartPanel);
        pack();
        setVisible(true);
    }

    private JFreeChart createChart(String title, XYDataset dataset) {
        chart = ChartFactory.createXYLineChart(
                title,
                "x",
                "y",
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );
        chart.setBackgroundPaint(Color.white);

        XYPlot plot = chart.getXYPlot();
        plot.setBackgroundPaint(Color.lightGray);
        plot.setDomainGridlinePaint(Color.white);
        plot.setRangeGridlinePaint(Color.white);

        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
        renderer.setSeriesLinesVisible(0, false);
        renderer.setSeriesShapesVisible(2, false);
        plot.setRenderer(renderer);

        NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        //rangeAxis.setRange(0.7, 1.3);
        //rangeAxis.setRange(-1000, 100);

        //plot.getDomainAxis().setRange(-1000, 1000);
        rangeAxis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());

        return chart;
    }

}