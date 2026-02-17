package de.fhdw.knn.visualization;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.trainer.Trainer;
import javafx.application.Platform;

public class LiveViewManager {

    private final int heatmapInterval;
    private final int sankeyInterval;

    private HeatmapWindow heatmapWindow;
    private SankeyLiveView sankeyView;

    public LiveViewManager(final int heatmapInterval, final int sankeyInterval, Network network) {
        this.heatmapInterval = heatmapInterval;
        this.sankeyInterval = sankeyInterval;

        if (this.heatmapInterval > 0) {
            this.heatmapWindow = new HeatmapWindow();
        }

        if (this.sankeyInterval > 0) {
            try {
                Platform.startup(() -> {}); // JavaFx initialisieren
            } catch (IllegalStateException ignored) {} // Ignorieren, falls es schon läuft

            this.sankeyView = new SankeyLiveView();
            this.sankeyView.show(network);
        }
    }

    public void nextEpoch(final int epoch, Network network, int maxEpochs) {
        // Heatmap Update
        if (heatmapWindow != null && shouldUpdate(epoch, heatmapInterval, maxEpochs)) {
            heatmapWindow.addEpoch("Epoche " + epoch, new HeatmapData(network));
        }

        // Sankey Update
        if (sankeyView != null&& shouldUpdate(epoch, sankeyInterval, maxEpochs)) {
            sankeyView.update(network, epoch);
        }
    }

    private boolean shouldUpdate(int epoch, int interval, int maxEpochs) {
        return epoch == 1 || epoch % interval == 0 || epoch == maxEpochs;
    }
}
