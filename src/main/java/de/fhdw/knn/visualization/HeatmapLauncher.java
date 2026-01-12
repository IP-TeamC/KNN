package de.fhdw.knn.visualization;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.io.Importer;

import javax.swing.*;

public class HeatmapLauncher {

    public static void main(String[] args) {
        String path = "models/datapoints.knn";

        System.out.println("Lade Netzwerk von: " + path);
        Network network = Importer.importNetwork(path);

        if (network == null) {
            System.err.println("Fehler beim Laden des Netzwerks!");
            return;
        }

        // GUI Thread starten
        SwingUtilities.invokeLater(() -> {
            HeatmapWindow window = new HeatmapWindow();
            HeatmapData data = new HeatmapData(network);
            window.addEpoch(0, data);
            System.out.println("Heatmap geöffnet.");
        });
    }
}