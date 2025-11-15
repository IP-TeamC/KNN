package de.fhdw.knn.run;
import java.awt.Color;
import java.awt.event.MouseEvent;
import java.awt.geom.Rectangle2D;
import java.util.Arrays;

import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartMouseEvent;
import org.jfree.chart.ChartMouseListener;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.ui.ApplicationFrame;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

public class ChartDemo extends ApplicationFrame implements ChartMouseListener {

    /**
     * Creates a new demo.
     *
     * @param title  the frame title.
     */

    public static ChartPanel chartPanel = null;
    public static JFreeChart chart = null;
    public static XYSeries series1,series2;
    public ChartDemo(final String title, XYSeries... xySeries) {

        super(title);

        final XYDataset dataset = createDataset(xySeries);
        final JFreeChart chart = createChart(dataset);
        chartPanel = new ChartPanel(chart);
        chartPanel.addChartMouseListener(this);
        chartPanel.setPreferredSize(new java.awt.Dimension(500, 270));
        setContentPane(chartPanel);

    }

    /**
     * Creates a sample dataset.
     *
     * @return a sample dataset.
     */
    private XYDataset createDataset(XYSeries... xySeries) {

        final XYSeriesCollection dataset = new XYSeriesCollection();
        Arrays.stream(xySeries).forEach(dataset::addSeries);

        return dataset;

    }

    /**
     * Creates a chart.
     *
     * @param dataset  the data for the chart.
     *
     * @return a chart.
     */
    private JFreeChart createChart(final XYDataset dataset) {

        // create the chart...
        chart = ChartFactory.createXYLineChart(
                "Chart Demo",      // chart title
                "X",                      // x axis label
                "Y",                      // y axis label
                dataset,                  // data
                PlotOrientation.VERTICAL,
                true,                     // include legend
                true,                     // tooltips
                false                     // urls
        );


        chart.setBackgroundPaint(Color.white);

        final XYPlot plot = chart.getXYPlot();
        plot.setBackgroundPaint(Color.lightGray);
//    plot.setAxisOffset(new Spacer(Spacer.ABSOLUTE, 5.0, 5.0, 5.0, 5.0));
        plot.setDomainGridlinePaint(Color.white);
        plot.setRangeGridlinePaint(Color.white);

        final XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
        renderer.setSeriesLinesVisible(0, false);//to disable line on the graph
        renderer.setSeriesShapesVisible(2, false);//to disable shape on the graph

        plot.setRenderer(renderer);

        final NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        rangeAxis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());

        return chart;
    }


    public static void main(final String[] args)
    {

    }

    @Override
    public void chartMouseClicked(ChartMouseEvent cme)
    {
        int mouseX = cme.getTrigger().getX();
        int mouseY = cme.getTrigger().getY();
        System.out.println("x = " + mouseX + ", y = " + mouseY);
        java.awt.geom.Point2D p = chartPanel.translateScreenToJava2D(
                new java.awt.Point(mouseX, mouseY));
        XYPlot plot = (XYPlot) chart.getPlot();
        System.out.println("Mouse clicked!!!!!");
        Rectangle2D plotArea = this.chartPanel.getChartRenderingInfo().getPlotInfo().getDataArea();
        ValueAxis domainAxis = plot.getDomainAxis();
        org.jfree.chart.ui.RectangleEdge domainAxisEdge = plot.getDomainAxisEdge();
        ValueAxis rangeAxis = plot.getRangeAxis();
        org.jfree.chart.ui.RectangleEdge rangeAxisEdge = plot.getRangeAxisEdge();
        double chartX = domainAxis.java2DToValue(p.getX(), plotArea,
                domainAxisEdge);
        double chartY = rangeAxis.java2DToValue(p.getY(), plotArea,
                rangeAxisEdge);
        System.out.println("Chart: x = " + chartX + ", y = " + chartY);
        series2.add(chartX, chartY);
    }


    @Override
    public void chartMouseMoved(ChartMouseEvent cme)
    {
        if(cme.getTrigger().getButton() ==  MouseEvent.BUTTON1)
        {

        }
    }
}