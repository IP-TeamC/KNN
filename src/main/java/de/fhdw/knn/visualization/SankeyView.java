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

public class SankeyView {

    private Stage stage;
    private SankeyPlot sankey;
    private Label epochLabel;

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

    public void update(Network network, int epoch) { // Methode wird vom Trainer aufgerufen
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