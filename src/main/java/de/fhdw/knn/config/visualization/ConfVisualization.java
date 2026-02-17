package de.fhdw.knn.config.visualization;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.trainer.Trainer;
import de.fhdw.knn.visualization.LiveViewManager;
import io.github.wasabithumb.jtoml.serial.TomlSerializable;
import lombok.Data;

@Data
public class ConfVisualization implements TomlSerializable {

    private int heatmapInterval = 0;
    private int sankeyInterval = 0;

    public LiveViewManager create(Network network) {
        if (heatmapInterval <= 0 && sankeyInterval <= 0) {
            return null;
        }
        return new LiveViewManager(heatmapInterval, sankeyInterval, network);
    }

    public boolean isEnabled() {
        return heatmapInterval > 0 || sankeyInterval > 0;
    }
}
