package de.fhdw.knn.visualization;

import lombok.Setter;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.SymbolAxis;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.CrosshairState;
import org.jfree.chart.plot.PlotRenderingInfo;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.PaintScale;
import org.jfree.chart.renderer.xy.XYBlockRenderer;
import org.jfree.chart.renderer.xy.XYItemRendererState;
import org.jfree.chart.ui.RectangleAnchor;
import org.jfree.data.xy.DefaultXYZDataset;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYZDataset;

import javax.swing.*;
import java.awt.*;
import java.awt.geom.Rectangle2D;
import java.util.ArrayList;
import java.util.List;

/**
 * Die HeatmapView-Klasse erweitert JFrame und bietet Funktionen zur Visualisierung
 * von Heatmaps von neuronalen Netzwerk-Gewichtsmatrizen in einer Tab-Fenster-Oberfläche.
 * Jeder Tab zeigt eine Heatmap für eine bestimmte Epoche oder Konfiguration des Netzwerks.
 *
 * <p>Die Visualisierung verwendet JFreeChart zum Rendern von Diagrammen mit blockbasierter Darstellung
 * für die Heatmaps, wobei Gewichte auf Farben abgebildet und als Raster angezeigt werden.
 *
 * <p>Wir bedanken uns an dieser Stelle beim "TeamAB" für die Bereitstellung des Codes als Grundlage und zur Inspiration
 */
public class HeatmapView extends JFrame {

    /**
     * Eine JTabbedPane-Instanz, die als zentrale Registerkartenkomponente für die Anzeige
     * mehrerer Heatmap-Visualisierungen innerhalb der HeatmapView fungiert.
     *
     * <p>Die Registerkarten werden dynamisch verwaltet und aktualisiert,
     * wenn die Methode {@code addEpoch} aufgerufen wird.
     *
     * @see #addHeatmap(String, HeatmapData)
     */
    private final JTabbedPane tabs;

    /**
     * Stellt den Schwellenwert dar, der zur Bestimmung der Signifikanz bestimmter Verbindungen
     * im Zusammenhang mit Heatmap-Visualisierungen verwendet wird.
     */
    @Setter
    private double threshold = 0.1;

    /**
     * Gibt an, ob die Werte der Gewichte in der Heatmap angezeigt werden sollen.
     *
     * <p>Bei {@code false} werden nur die Farben dargestellt.
     */
    @Setter
    private boolean showWeights = true;

    /**
     * Gibt an, ob die Farbskala auf den tatsächlichen Wertebereich der Gewichtsmatrix normalisiert wird.
     *
     * <p>Bei {@code false} wird ein fester Bereich von [-1, 1] verwendet.
     *
     * @see ColorScheme
     */
    @Setter
    private boolean normalizeColors = true;

    /**
     * Das aktive Farbschema für die Heatmap-Visualisierung.
     * Bestimmt die Farben für negative und positive Gewichtswerte.
     *
     * @see ColorScheme
     */
    @Setter
    private ColorScheme colorSchema = ColorScheme.RED_GREEN;

    /**
     * Erstellt eine neue HeatmapView-Instanz.
     *
     * <p>Es erstellt einen Registerkartenbereich zur Anzeige verschiedener Heatmap-Visualisierungen,
     * konfiguriert die Fenstereigenschaften wie Größe und Standard-Schließvorgang und macht das Fenster sichtbar.
     */
    public HeatmapView() {
        super("Heatmap – Network View");

        this.tabs = new JTabbedPane();
        this.tabs.setTabLayoutPolicy(JTabbedPane.SCROLL_TAB_LAYOUT);

        this.setContentPane(tabs);
        this.setSize(1200, 900);
        this.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        this.setVisible(true);
    }
    /**
     * Fügt eine weitere Registerkarte zur {@code HeatmapView} hinzu.
     * Diese Methode wird im Event Dispatch Thread ausgeführt, da Swing-Komponenten nicht threadsicher sind.
     *
     * @param tabTitle Der Titel der hinzuzufügenden Registerkarte. Er steht in der Regel für die aktuelle Epoche.
     * @param data     Das {@code HeatmapData}-Objekt enthält Informationen über das Netzwerk,
     *                 einschließlich der Gewichtungsmatrix und Neuronenbezeichnungen.
     *
     * @see HeatmapView
     * @see HeatmapData
     */
    public void addHeatmap(String tabTitle, HeatmapData data) {
        SwingUtilities.invokeLater(() -> { // invokeLater, da Swing nicht Thread-safe ist

            JFreeChart chart = buildChart(data, tabTitle);
            ChartPanel panel = new ChartPanel(chart);
            XYPlot xyPlot = panel.getChart().getXYPlot();

            panel.setMouseWheelEnabled(true);

            xyPlot.setDomainPannable(true);
            xyPlot.setRangePannable(true);

            int N = data.buildFullWeightMatrix().length; // für Zoom-Grenzen

            xyPlot.getDomainAxis().setLowerBound(-0.5);
            xyPlot.getDomainAxis().setUpperBound(N - 0.5);
            xyPlot.getRangeAxis().setLowerBound(-0.5);
            xyPlot.getRangeAxis().setUpperBound(N - 0.5);

            // Beim Zoomen Viewgrenzen einhalten
            panel.addMouseWheelListener(e -> {
                ValueAxis domain = xyPlot.getDomainAxis();
                ValueAxis range  = xyPlot.getRangeAxis();

                double minBound = -0.5;
                double maxBound = N - 0.5;

                double domainLower = domain.getLowerBound();
                double domainUpper = domain.getUpperBound();
                if (domainLower < minBound) {
                    domain.setLowerBound(minBound);
                }
                if (domainUpper > maxBound) {
                    domain.setUpperBound(maxBound);
                }

                double rangeLower = range.getLowerBound();
                double rangeUpper = range.getUpperBound();
                if (rangeLower < minBound) {
                    range.setLowerBound(minBound);
                }
                if (rangeUpper > maxBound) {
                    range.setUpperBound(maxBound);
                }
            });

            JScrollPane scrollPane = new JScrollPane(panel);
            this.tabs.addTab(tabTitle, scrollPane);
            this.tabs.setSelectedIndex(this.tabs.getTabCount() - 1);
        });
    }

    /**
     * Erstellt eine JFreeChart-Instanz, um die Gewichtungsmatrix eines neuronalen Netzwerks
     * anhand der bereitgestellten Daten und des Titels als Heatmap zu visualisieren.
     *
     * @param data  Das {@code HeatmapData}-Objekt enthält die Gewichtungsmatrix und die Neuronenbezeichnungen für das Netzwerk.
     * @param title Der Titel, der auf dem Diagramm angezeigt werden soll.
     *
     * @return Ein {@code JFreeChart}-Objekt, das die Heatmap der Gewichtungsmatrix darstellt und
     *         mit den entsprechenden Darstellungs- und Achsenbeschriftungen konfiguriert ist.
     *
     * @see HeatmapData
     * @see JFreeChart
     */
    private JFreeChart buildChart(HeatmapData data, String title) {
        double[][] m = data.buildFullWeightMatrix();
        String[] labels = data.getNeuronLabels();

        int N = m.length;

        DefaultXYZDataset dataset = new DefaultXYZDataset();
        List<Double> xList = new ArrayList<>();
        List<Double> yList = new ArrayList<>();
        List<Double> zList = new ArrayList<>();

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

        double min, max;

        if (normalizeColors) {
            double dataMin = Double.MAX_VALUE;
            double dataMax = -Double.MAX_VALUE;
            for (double[] row : m) {
                for (double val : row) {
                    if (!Double.isNaN(val)) {
                        dataMin = Math.min(dataMin, val);
                        dataMax = Math.max(dataMax, val);
                    }
                }
            }
            // Symmetrisch um 0 halten
            double absMax = Math.max(Math.abs(dataMin), Math.abs(dataMax));
            min = -absMax;
            max = absMax;
        } else {
            min = -1.0;
            max = 1.0;
        }

        // Renderer
        XYBlockRenderer renderer = getXyBlockRenderer(min, max, threshold, showWeights, colorSchema);
        XYPlot plot = getXyPlot(labels, dataset, renderer);

        JFreeChart chart = new JFreeChart(title, JFreeChart.DEFAULT_TITLE_FONT, plot, false);
        chart.setBackgroundPaint(Color.WHITE);

        return chart;
    }

    /**
     * Erstellt und konfiguriert eine {@code XYPlot}-Instanz zur Visualisierung von Daten in einem 2D-Rasterformat.
     *
     * @param labels   Ein Array von {@code String}-Bezeichnungen, die für die symbolischen Achsen verwendet werden sollen.
     *                 Diese Beschriftungen geben die Namen der Neuronen auf beiden Achsen wieder.
     * @param dataset  Das {@code DefaultXYZDataset} enthält die zu visualisierenden Daten.
     * @param renderer Ein {@code XYBlockRenderer}, der für die Darstellung der Gitterzellen im Plot verantwortlich ist.
     *
     * @return Eine {@code XYPlot}-Instanz, konfiguriert mit symbolischen Achsen, dem bereitgestellten Datensatz und dem definierten Renderer.
     *
     * @see XYPlot
     * @see DefaultXYZDataset
     * @see XYBlockRenderer
     */
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

    /**
     * Erstellt und konfiguriert eine {@code XYBlockRenderer}-Instanz, um Daten in einem 2D-Rasterformat zu visualisieren.
     * Der Renderer ist dafür verantwortlich, Gitterzellen mit einer Farbskala basierend auf den angegebenen
     * Mindest- sowie Höchstwerten und unter Beachtung des {@code THRESHOLD} zu rendern.
     *
     * @param min         Der Mindestwert im Datenbereich. Dieser Wert stellt die Untergrenze der Farbskala dar.
     * @param max         Der Maximalwert im Datenbereich. Dieser Wert stellt die Obergrenze der Farbskala dar.
     * @param threshold   Der Schwellenwert, unter dem Gewichte als unbedeutend betrachtet und weiß dargestellt werden.
     * @param showWeights Gibt an, ob die numerischen Gewichtswerte in den Gitterzellen angezeigt werden sollen.
     * @param colorScheme Das Farbschema, das zum Zuordnen von Datenwerten zu Farben verwendet wird.
     *
     * @return Eine {@code XYBlockRenderer}-Instanz, konfiguriert mit einer Blockgröße von 1.0 und
     * einer Farbskala, um Datenwerte Farben von Grün (negativ) bis Rot (positiv) zuzuordnen.
     *
     * @see XYBlockRenderer
     * @see HeatmapView#threshold
     * @see ColorScheme
     */
    private static XYBlockRenderer getXyBlockRenderer(double min, double max, double threshold, boolean showWeights, ColorScheme colorScheme) {
        XYBlockRenderer renderer = new XYBlockRenderer() {
            @Override
            public void drawItem(Graphics2D g2,
                                 XYItemRendererState state,
                                 Rectangle2D dataArea,
                                 PlotRenderingInfo info,
                                 XYPlot plot,
                                 ValueAxis domainAxis,
                                 ValueAxis rangeAxis,
                                 XYDataset dataset,
                                 int series, int item,
                                 CrosshairState crosshairState,
                                 int pass) {

                super.drawItem(g2, state, dataArea, info, plot, domainAxis, rangeAxis,
                        dataset, series, item, crosshairState, pass);

                if (!showWeights) return;

                XYZDataset xyzDataset = (XYZDataset) dataset;
                double z = xyzDataset.getZValue(series, item);
                if (Double.isNaN(z)) return;

                double x = dataset.getXValue(series, item);
                double y = dataset.getYValue(series, item);

                double x0 = domainAxis.valueToJava2D(x - 0.5, dataArea, plot.getDomainAxisEdge());
                double x1 = domainAxis.valueToJava2D(x + 0.5, dataArea, plot.getDomainAxisEdge());
                double blockWidthPx = Math.abs(x1 - x0);

                // Mindestgröße für Schrift
                if (blockWidthPx < 14) return;

                double fontSize = Math.min(blockWidthPx * 0.3, 14);
                g2.setFont(new Font("SansSerif", Font.PLAIN, (int) fontSize));
                g2.setColor(((max > 0) ? Math.abs(z) / max : 0.0) > 0.5 ? Color.WHITE : Color.DARK_GRAY);

                String label = String.format("%.2f", z);

                double screenX = domainAxis.valueToJava2D(x, dataArea, plot.getDomainAxisEdge());
                double screenY = rangeAxis.valueToJava2D(y, dataArea, plot.getRangeAxisEdge());

                FontMetrics fm = g2.getFontMetrics();
                float textX = (float) (screenX - fm.stringWidth(label) / 2.0);
                float textY = (float) (screenY + fm.getAscent() / 2.0 - fm.getDescent() / 2.0);

                g2.drawString(label, textX, textY);
            }
        };

        renderer.setBlockWidth(1.0);
        renderer.setBlockHeight(1.0);
        renderer.setBlockAnchor(RectangleAnchor.CENTER);

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
                if (Math.abs(value) < threshold) return Color.WHITE; // Weiß = "Unter Threshold"

                double ratio = (max > 0) ? Math.abs(value) / max : 0.0;

                Color target = (value > 0) ? colorScheme.positive : colorScheme.negative;
                return interpolateToWhite(target, ratio);
            }
        });
        return renderer;
    }

    /**
     * Interpoliert eine bestimmte Farbe anhand eines festgelegten Verhältnisses in Richtung Weiß.
     *
     * @param target Die Zielfarbe {@code Color}, die in Weiß übergehen soll.
     * @param ratio  Ein {@code double}-Wert zwischen 0,0 und 1,0, der den Interpolationsfaktor angibt,
     *               wobei 0,0 die Originalfarbe und 1,0 reines Weiß ergibt.
     * @return Ein {@code Paint}-Objekt, das die interpolierte Farbe darstellt.
     */
    private static Paint interpolateToWhite(Color target, double ratio) {
        float r = Math.clamp(1.0f - (float) ratio * (1.0f - target.getRed()   / 255.0f), 0.0f, 1.0f);
        float g = Math.clamp(1.0f - (float) ratio * (1.0f - target.getGreen() / 255.0f), 0.0f, 1.0f);
        float b = Math.clamp(1.0f - (float) ratio * (1.0f - target.getBlue()  / 255.0f), 0.0f, 1.0f);
        return new Color(r, g, b);
    }
}
