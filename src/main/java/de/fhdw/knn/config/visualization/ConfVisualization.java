package de.fhdw.knn.config.visualization;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.visualization.ViewManager;
import io.github.wasabithumb.jtoml.serial.TomlSerializable;

/**
 * Die Klasse {@code ConfVisualization} stellt die übergeordnete Konfiguration für einen {@code ViewManager} dar.
 *
 * <p>Heatmap- und Sankey-spezifische Einstellungen sind in {@link ConfHeatmap} bzw. {@link ConfSankey} ausgelagert.
 *
 * <p>Implementiert die Schnittstelle {@code TomlSerializable}, um die TOML-basierte Serialisierung zu ermöglichen.
 *
 * @see ConfHeatmap
 * @see ConfSankey
 * @see ViewManager
 * @see TomlSerializable
 */
public class ConfVisualization implements TomlSerializable {

    /**
     * Objekte dieser Klasse sollen nicht manuell instanziiert werden (deshalb package private).
     */
    ConfVisualization() {
    }

    /**
     * Gibt an, ob die Visualisierung aktiviert ist.
     */
    public boolean enabled = true;

    /**
     * Konfiguration für die Heatmap-Visualisierung.
     *
     * @see ConfHeatmap
     */
    public ConfHeatmap heatmap = new ConfHeatmap();

    /**
     * Konfiguration für die Sankey-Visualisierung.
     *
     * @see ConfSankey
     */
    public ConfSankey sankey = new ConfSankey();

    /**
     * Erstellt eine neue Instanz von {@code ViewManager} basierend auf dieser Konfiguration.
     *
     * <p>Gibt {@code null} zurück, wenn sowohl {@code heatmap.interval} als auch
     * {@code sankey.interval} den Wert {@code 0} haben.
     *
     * @param network die {@code Network}-Instanz zur Initialisierung der Visualisierungen.
     * @return eine konfigurierte {@code ViewManager}-Instanz oder {@code null},
     *         wenn die Visualisierung deaktiviert ist.
     * @see ViewManager
     */
    public ViewManager create(Network network) {
        if (!isEnabled()) return null;
        return new ViewManager(heatmap.toConfig(), sankey.toConfig(), network);
    }

    /**
     * Gibt an, ob die Visualisierung aktiviert ist.
     *
     * @return {@code true}, wenn mindestens eine Visualisierungsart aktiviert ist
     *         ({@code heatmap.interval != 0} oder {@code sankey.interval != 0}).
     */
    public boolean isEnabled() {
        return (heatmap.interval != 0 || sankey.interval != 0) && enabled;
    }
}