package de.fhdw.knn.visualization;

import javafx.application.Platform;
import javafx.scene.Scene;
import javafx.stage.Stage;
import eu.hansolo.fx.charts.SankeyPlot;
import eu.hansolo.fx.charts.data.PlotItem;
import lombok.Setter;

import java.util.List;

/**
 * Klasse, die eine Sankey-Ansicht zur Visualisierung von Netzwerkdaten mit JavaFX darstellt.
 * <p>Diese Klasse ist für die Initialisierung und Anzeige eines Fensters mit einem Sankey-Diagramm
 * sowie für dessen dynamische Aktualisierung mit neuen Daten verantwortlich.
 */
public class SankeyView {

    /**
     * Stellt die primäre JavaFX Stage für die Anzeige des Sankey-Diagramms dar.
     * Die Stage wird initialisiert und angezeigt, wenn die Methode {@code update} aufgerufen wird,
     * und dient als Container für das Sankey-Diagramm und die zugehörigen UI-Elemente.
     *
     * @see #update(SankeyData, String)
     */
    private Stage stage;
    /**
     * Stellt eine JavaFX-Komponente zur Visualisierung eines Sankey-Diagramms dar.
     * Die Instanz {@code SankeyPlot} ist für die Darstellung der Beziehungen verantwortlich.
     *
     * <p>Diese Variable wird im Kontext der Klasse {@code SankeyView} initialisiert und konfiguriert und
     * während der Laufzeit dynamisch aktualisiert, um den aktuellen Status der visualisierten Daten widerzuspiegeln.
     *
     * @see SankeyPlot
     * @see SankeyView
     */
    private SankeyPlot sankey;

    /**
     * Stellt den Schwellenwert dar, der zur Bestimmung der Signifikanz bestimmter Verbindungen
     * im Zusammenhang mit Heatmap-Visualisierungen verwendet wird.
     *
     * <p>Verbindungen mit {@code |weight| ≤ threshold} werden nicht dargestellt.
     */
    @Setter
    private double threshold;

    /**
     * Gibt an, ob positive, negative oder alle Gewichte dargestellt werden.
     *
     * @see WeightFilter
     */
    @Setter
    private WeightFilter weightFilter;

    /**
     * Erstellt eine neue {@code SankeyView} und visualisiert die initialen Netzwerkdaten
     * anhand der angegebenen Konfiguration.
     *
     * @param initialData  Die initialen {@code SankeyData}, die beim Öffnen des Fensters dargestellt werden.
     * @param title        Der Titel, der in der Titelleiste des Fensters angezeigt wird.
     * @param threshold    Der Schwellenwert, der zum Filtern von Verbindungen verwendet wird.
     * @param weightFilter Der Filter, der angibt, welche Gewichte dargestellt werden sollen.
     * @see SankeyConfig
     */
    public SankeyView(SankeyData initialData, String title, Double threshold, WeightFilter weightFilter) {
        this.threshold = threshold;
        this.weightFilter = weightFilter;

        List<PlotItem> initialItems = initialData.convertToPlotItems(threshold, weightFilter);

        Platform.runLater(() -> {
            this.stage = new Stage();
            this.sankey = new SankeyPlot();

            sankey.setItems(initialItems);
            sankey.setStreamFillMode(SankeyPlot.StreamFillMode.GRADIENT);
            sankey.setShowFlowDirection(false);

            this.stage.setTitle("Sankey - Network View (" + title + ")");
            this.stage.setScene(new Scene(sankey, 1200, 900));
            this.stage.show();
            this.stage.toFront();
            this.stage.requestFocus();

            System.out.println("Sankey-Fenster geöffnet.");
        });
    }

    /**
     * Erstellt eine neue {@code SankeyView} mit Standardkonfiguration
     * und visualisiert die initialen Netzwerkdaten.
     *
     * <p>- Threshold: {@code 0.1}
     * <br>- WeightFilter: {@code WeightFilter.BOTH}
     *
     * @param initialData Die initialen {@code SankeyData}, die beim Öffnen des Fensters dargestellt werden.
     * @param title       Der Titel, der in der Titelleiste des Fensters angezeigt wird.
     */
    public SankeyView(SankeyData initialData, String title) {
        this(initialData, title, 0.1, WeightFilter.BOTH);
    }

    /**
     * Aktualisiert das Sankey-Diagramm mit neuen Netzwerkdaten.
     * Die Konvertierung der Daten erfolgt mit dem aktuell konfigurierten
     * {@code threshold} und {@code weightFilter} dieser Instanz.
     *
     * @param data  Die {@code SankeyData}-Instanz, deren Netzwerk visualisiert werden soll.
     * @param title Der Titel, der in der Titelleiste des Fensters angezeigt wird.
     * @see SankeyData#convertToPlotItems(Double, WeightFilter)
     */
    public void update(SankeyData data, String title) {
        List<PlotItem> items = data.convertToPlotItems(this.threshold, this.weightFilter);
        Platform.runLater(() -> {
            stage.setTitle("Sankey - Network View (" + title + ")");
            sankey.setItems(items);
        });
    }
}
