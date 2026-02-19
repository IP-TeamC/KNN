package de.fhdw.knn.config.visualization;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.visualization.ViewManager;
import io.github.wasabithumb.jtoml.serial.TomlSerializable;
import lombok.Data;

@Data
public class ConfVisualization implements TomlSerializable {

    private int heatmapInterval = 0;
    private int sankeyInterval = 0;

    public ViewManager create(Network network) {
        if (heatmapInterval <= 0 && sankeyInterval <= 0) {
            return null;
        }
        return new ViewManager(heatmapInterval, sankeyInterval, network);
    }

    public boolean isEnabled() {
        return heatmapInterval > 0 || sankeyInterval > 0;
    }
}
