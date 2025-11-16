package de.fhdw.knn.run;

import java.awt.Color;
import java.util.Arrays;

import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.ui.ApplicationFrame;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

public class LineChart extends ApplicationFrame {

    public ChartPanel chartPanel;
    public JFreeChart chart;

    public LineChart(String title, String xLable, String yLable, double minX, double maxX, XYSeries... xySeries) {
        super(title);
        XYSeriesCollection dataset = new XYSeriesCollection();
        Arrays.stream(xySeries).forEach(dataset::addSeries);
        JFreeChart chart = createChart(title, xLable, yLable, dataset, minX, maxX);
        chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new java.awt.Dimension(1000, 500));
        setContentPane(chartPanel);
        pack();
        setVisible(true);
    }

    private JFreeChart createChart(String title, String xLable, String yLable, XYDataset dataset, double minX, double maxX) {
        chart = ChartFactory.createXYLineChart(
                title,
                xLable,
                yLable,
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
        rangeAxis.setRange(0, 40);
        plot.getDomainAxis().setRange(minX, maxX);
        rangeAxis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());

        return chart;
    }

}