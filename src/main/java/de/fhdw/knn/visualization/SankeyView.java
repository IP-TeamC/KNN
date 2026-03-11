package de.fhdw.knn.visualization;

import de.fhdw.knn.network.Network;
import javafx.application.Platform;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Label;
import javafx.scene.layout.StackPane;
import javafx.stage.Stage;
import eu.hansolo.fx.charts.SankeyPlot;
import eu.hansolo.fx.charts.data.PlotItem;

import java.util.List;

/**
 * Klasse, die eine Sankey-Ansicht zur Visualisierung von Netzwerkdaten mit JavaFX darstellt.
 * <p>Diese Klasse ist für die Initialisierung und Anzeige eines Fensters mit einem Sankey-Diagramm
 * sowie für dessen dynamische Aktualisierung mit neuen Daten verantwortlich.
 */
public class SankeyView {

    /**
     * Erzeugt eine Instanz von SankeyView
     *
     * @see SankeyView
     */
    public SankeyView() {
    }

    /**
     * Stellt die primäre JavaFX Stage für die Anzeige des Sankey-Diagramms dar.
     * Diese Stufe wird initialisiert und angezeigt, wenn die Methode {@code show} aufgerufen wird,
     * und dient als Container für das Sankey-Diagramm und die zugehörigen UI-Elemente.
     *
     * @see #show(Network)
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
     * Stellt eine JavaFX-Komponente zur Anzeige der aktuellen Epochennummer dar.
     */
    private Label epochLabel;

    /**
     * Öffnet ein JavaFX-Fenster mit einem Sankey-Diagramm, das mit den Daten aus dem bereitgestellten Netzwerk initialisiert wurde.
     *
     * @param initialNetwork Die ursprünglichen Netzwerkdaten, die im Sankey-Diagramm visualisiert werden sollen.
     */
    public void show(Network initialNetwork) {
        //Platform.setImplicitExit(false);
        List<PlotItem> initialItems = SankeyData.convertNetworkToItems(initialNetwork);

        // JavaFX Fenster im FX-Thread initialisieren
        Platform.runLater(() -> {
            this.stage = new Stage();
            this.sankey = new SankeyPlot();

            // Epochen-Label erstellen
            epochLabel = new javafx.scene.control.Label("");
            epochLabel.setStyle("-fx-font-size: 24px; -fx-font-weight: bold; -fx-text-fill: #333; -fx-padding: 10px;");
            StackPane root = new StackPane(sankey, epochLabel);
            StackPane.setAlignment(epochLabel, Pos.TOP_LEFT);

            // Initialdaten setzen
            if (!initialItems.isEmpty()) {
                sankey.setItems(initialItems);
            }

            sankey.setStreamFillMode(SankeyPlot.StreamFillMode.GRADIENT);
            sankey.setShowFlowDirection(false);

            this.stage.setTitle("Sankey - Network View");
            this.stage.setScene(new Scene(root, 1200, 800));

            this.stage.show();
            System.out.println("Sankey-Fenster geöffnet.");
        });
    }

    /**
     * Aktualisiert das Sankey-Diagramm mit den neuesten Daten aus dem Netzwerk.
     *
     * @param network Das Netzwerk, dessen Daten im Sankey-Diagramm visualisiert werden.
     * @param epoch   Die aktuelle Epochennummer, die im Label angezeigt werden soll.
     */
    public void update(Network network, int epoch) {
        List<PlotItem> items = SankeyData.convertNetworkToItems(network);
        Platform.runLater(() -> {
            if (sankey != null) {
                sankey.setItems(items);
                if (epochLabel != null) {
                    epochLabel.setText("Epoche: " + epoch);
                }
            }
        });
    }
}
