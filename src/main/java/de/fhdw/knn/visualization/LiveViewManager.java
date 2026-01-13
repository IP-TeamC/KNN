package de.fhdw.knn.visualization;

import de.fhdw.knn.trainer.Trainer;
import javafx.application.Platform;

public class LiveViewManager {

    // Einstellung: Alle wie viele Epochen soll die Heatmap/Sankey aktualisiert werden?
    public static final int HEATMAP_INTERVAL = 5;
    public static final int SANKEY_INTERVAL = 10;

    private final Trainer trainer;
    private final HeatmapWindow heatmapWindow;
    private final SankeyLiveView sankeyView;

    public LiveViewManager(final Trainer trainer) {
        this.trainer = trainer;

        if (HEATMAP_INTERVAL > 0) {
            this.heatmapWindow = new HeatmapWindow();
        }

        if (SANKEY_INTERVAL > 0) {
            try {
                Platform.startup(() -> {});
            } catch (IllegalStateException ignored) {} // Falls es schon läuft

            this.sankeyView = new SankeyLiveView();
            this.sankeyView.show(this.trainer.network);
        }
    }

    public void nextEpoch(final int epoch) {
        // Heatmap Update
        if (HEATMAP_INTERVAL > 0 && heatmapWindow != null) {
            if (epoch == 1 || epoch % HEATMAP_INTERVAL == 0 || epoch == trainer.maxEpochs) {
                heatmapWindow.addEpoch("Epoche " + epoch, new HeatmapData(trainer.network));
                System.out.println("Heatmap-Update bei Epoche " + epoch);
            }
        }

        // Sankey Update
        if (SANKEY_INTERVAL > 0 && sankeyView != null) {
            if (epoch == 1 || epoch % SANKEY_INTERVAL == 0 || epoch == trainer.maxEpochs) {
                sankeyView.update(trainer.network, epoch);
                System.out.println("Sankey-Update bei Epoche " + epoch);
            }
        }
    }

}
