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
     * @see #addEpoch(String, HeatmapData)
     */
    private final JTabbedPane tabs;

    /**
     * Stellt den Schwellenwert dar, der zur Bestimmung der Signifikanz bestimmter Verbindungen
     * im Zusammenhang mit Heatmap-Visualisierungen verwendet wird.
     *
     * <p>Der Schwellwert ist derzeit auf fest auf 0.1 codiert.
     */
    private static final double THRESHOLD = 0.1;

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
     * Zeigt eine einzelne Heatmap in einem neuen Tab innerhalb des HeatmapView-Fensters an.
     *
     * @param title  Der Titel, der für die Heatmap und die entsprechende Registerkarte angezeigt werden soll.
     * @param matrix das {@code HeatmapData}-Objekt, das Informationen über das neuronale Netzwerk und
     *               die zugehörige Gewichtungsmatrix enthält, die visualisiert werden sollen
     *
     * @see HeatmapData
     */
    public void showSingleMatrix(String title, HeatmapData matrix) {
        this.addEpoch(title, matrix);

        this.setTitle(title);
        this.setDefaultCloseOperation(javax.swing.JFrame.DISPOSE_ON_CLOSE);
        this.pack();
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
     * @param min Der Mindestwert im Datenbereich. Dieser Wert stellt die Untergrenze der Farbskala dar.
     * @param max Der Maximalwert im Datenbereich. Dieser Wert stellt die Obergrenze der Farbskala dar.
     *
     * @return Eine {@code XYBlockRenderer}-Instanz, konfiguriert mit einer Blockgröße von 1.0 und
     *         einer Farbskala, um Datenwerte Farben von Grün (negativ) bis Rot (positiv) zuzuordnen.
     *
     * @see XYBlockRenderer
     * @see HeatmapView#THRESHOLD
     */
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
