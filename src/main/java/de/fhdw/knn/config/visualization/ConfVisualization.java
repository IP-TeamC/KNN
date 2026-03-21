package de.fhdw.knn.config.visualization;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.visualization.ColorScheme;
import de.fhdw.knn.visualization.ViewManager;
import io.github.wasabithumb.jtoml.serial.TomlSerializable;

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
    public int heatmapInterval = 0;
    /**
     * Gibt an, nach wie vielen Epochen ein neues SankeyPlot generiert werden soll.
     *
     * <p>{@code 0} bedeutet, dass keine SankeyPlots generiert werden sollen.
     *
     * @see ConfVisualization#create(Network)
     */
    public int sankeyInterval = 0;

    /**
     * Gibt an, ob die Gewichtswerte innerhalb der Heatmap-Felder angezeigt werden sollen.
     *
     * <p>Bei {@code false} werden nur die Farben dargestellt.
     *
     * @see ColorScheme
     */
    public boolean showWeights = true;

    /**
     * Gibt an, ob die Farbskala der Heatmap auf den tatsächlichen Wertebereich der
     * Gewichtsmatrix normalisiert werden soll.
     *
     * <p>Bei {@code true} erhält der höchste bzw. niedrigste Gewichtswert die kräftigste Farbe,
     * unabhängig davon, ob die Werte im Bereich [-1, 1] liegen oder nicht.
     *
     * <p>Bei {@code false} wird ein fester Bereich von [-1, 1] verwendet. Werte außerhalb
     * dieses Bereichs werden mit der maximalen Farbsättigung dargestellt.
     *
     * @see de.fhdw.knn.visualization.ColorScheme
     */
    public boolean normalizeColors = true;

    /**
     * Legt das Farbschema fest, das für die Heatmap-Visualisierung verwendet werden soll.
     *
     * <p>Das Schema bestimmt, welche Farben für negative und positive Gewichtswerte verwendet werden.
     *
     * @see ColorScheme
     */
    public String colorScheme = "GREEN_RED";

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
        ColorScheme scheme = ColorScheme.valueOf(colorScheme.toUpperCase());

        return new ViewManager(heatmapInterval, sankeyInterval, showWeights, normalizeColors, scheme, network);
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
