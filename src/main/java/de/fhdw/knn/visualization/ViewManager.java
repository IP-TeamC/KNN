package de.fhdw.knn.visualization;

import de.fhdw.knn.network.Network;
import javafx.application.Platform;

public class ViewManager {

    private final int heatmapInterval;
    private final int sankeyInterval;

    private HeatmapView heatmapView;
    private SankeyView sankeyView;

    public ViewManager(final int heatmapInterval, final int sankeyInterval, Network network) {
        this.heatmapInterval = heatmapInterval;
        this.sankeyInterval = sankeyInterval;

        if (this.heatmapInterval > 0) {
            this.heatmapView = new HeatmapView();
        }

        if (this.sankeyInterval > 0) {
            try {
                Platform.startup(() -> {
                }); // JavaFx initialisieren
            } catch (IllegalStateException ignored) {
            } // Ignorieren, falls es schon läuft

            this.sankeyView = new SankeyView();
            this.sankeyView.show(network);
        }
    }

    public void nextEpoch(final int epoch, Network network, int maxEpochs) {
        // Heatmap Update
        if (heatmapView != null && shouldUpdate(epoch, heatmapInterval, maxEpochs)) {
            heatmapView.addEpoch("Epoche " + epoch, new HeatmapData(network));
        }

        // Sankey Update
        if (sankeyView != null && shouldUpdate(epoch, sankeyInterval, maxEpochs)) {
            sankeyView.update(network, epoch);
        }
    }

    private boolean shouldUpdate(int epoch, int interval, int maxEpochs) {
        return epoch == 1 || epoch % interval == 0 || epoch == maxEpochs;
    }
}
