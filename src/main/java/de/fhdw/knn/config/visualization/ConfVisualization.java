package de.fhdw.knn.config.visualization;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.visualization.ViewManager;
import io.github.wasabithumb.jtoml.serial.TomlSerializable;
import lombok.Data;

/**
 * Die Klasse {@code ConfVisualization} stellt eine Konfiguration für einen {@code ViewManager} dar.
 *
 * <p>Es bietet Optionen zum Aktivieren und Konfigurieren von Intervallen für Heatmap- und Sankey-Diagramm-Visualisierungen.
 *
 * <p>Implementiert die Schnittstelle {@code TomlSerializable}, um die TOML-basierte Serialisierung zu ermöglichen.
 *
 * @see ViewManager
 * @see TomlSerializable
 */
@Data
public class ConfVisualization implements TomlSerializable {

    /**
     * Objekte dieser Klasse sollen nicht manuell instanziiert werden (deshalb package private)
     */
    ConfVisualization() {
    }

    /**
     * Gibt an, nach wie vielen Epochen eine neue Heatmap generiert werden soll.
     *
     * <p>{@code 0} bedeutet, dass keine Heatmaps generiert werden sollen.
     *
     * @see ConfVisualization#create(Network)
     */
    private int heatmapInterval = 0;
    /**
     * Gibt an, nach wie vielen Epochen ein neues SankeyPlot generiert werden soll.
     *
     * <p>{@code 0} bedeutet, dass keine SankeyPlots generiert werden sollen.
     *
     * @see ConfVisualization#create(Network)
     */
    private int sankeyInterval = 0;

    /**
     * Erstellt eine neue Instanz von {@code ViewManager}, die mit den angegebenen Intervallen
     * für Heatmap- und Sankey-Diagramm-Visualisierungen konfiguriert ist.
     *
     * <p>Wenn sowohl {@code heatmapInterval} als auch {@code sankeyInterval} den Wert {@code 0} oder
     * einen negativen Wert haben, gibt diese Methode {@code null} zurück.
     *
     * @param network die {@code Network}-Instanz, die zum Initialisieren der Visualisierungen verwendet wird.
     * @return eine {@code ViewManager}-Instanz, die mit den Heatmap- und Sankey-Intervallen konfiguriert ist;
     * oder {@code null}, wenn die Visualisierung deaktiviert ist.
     * @see ViewManager
     */
    public ViewManager create(Network network) {
        if (heatmapInterval <= 0 && sankeyInterval <= 0) {
            return null;
        }
        return new ViewManager(heatmapInterval, sankeyInterval, network);
    }

    /**
     * Legt fest, ob die Visualisierung basierend auf der Konfiguration der Heatmap- und Sankey-Intervalle aktiviert ist.
     *
     * @return {@code true}, wenn entweder das Heatmap-Intervall oder das Sankey-Intervall größer als 0 ist,
     * was bedeutet, dass mindestens eine Art der Visualisierung aktiviert ist; andernfalls {@code false}.
     */
    public boolean isEnabled() {
        return heatmapInterval > 0 || sankeyInterval > 0;
    }
}
