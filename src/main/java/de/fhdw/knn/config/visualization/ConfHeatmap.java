package de.fhdw.knn.config.visualization;

import de.fhdw.knn.visualization.ColorScheme;
import de.fhdw.knn.visualization.HeatmapConfig;
import de.fhdw.knn.visualization.WeightFilter;
import io.github.wasabithumb.jtoml.serial.TomlSerializable;

/**
 * Konfiguration für die Heatmap-Visualisierung.
 *
 * <p>Wird als verschachtelte Sektion {@code [visu.heatmap]} in der TOML-Konfiguration verwendet.
 *
 * @see ConfVisualization
 * @see ColorScheme
 * @see TomlSerializable
 */
public class ConfHeatmap implements TomlSerializable {

    /**
     * Objekte dieser Klasse sollen nicht manuell instanziiert werden (deshalb package private).
     */
    ConfHeatmap() {
    }

    /**
     * Stellt das Intervall (in Epochen) dar, in dem die Heatmap aktualisiert wird.
     *
     * <p>Ein Wert größer als 0 gibt die Periodizität der Aktualisierungen an (z. B. alle n Epochen).
     * Der Wert {@code -1} aktiviert die Heatmap ausschließlich am Ende des Trainings.
     * Der Wert {@code 0} deaktiviert die Heatmap vollständig.
     */
    public int interval = 0;

    /**
     * Gibt an, ob die Gewichtswerte innerhalb der Heatmap-Felder angezeigt werden sollen.
     *
     * <p>Bei {@code false} werden nur die Farben dargestellt.
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
     */
    public boolean normalizeColors = true;

    /**
     * Legt das Farbschema für die Heatmap fest.
     * Mögliche Werte entsprechen den Konstanten von {@link ColorScheme}.
     *
     * @see ColorScheme
     */
    public String colorScheme = "GREEN_RED";

    /**
     * Schwellenwert für die Visualisierung von Gewichtsverbindungen.
     * Verbindungen mit {@code |weight| ≤ threshold} werden nicht dargestellt.
     *
     * <p>Gilt für Heatmap und Sankey.
     */
    public double threshold = 0.1;

    /**
     * Gibt an, ob positive, negative oder alle Gewichte visualisiert werden sollen.
     * Mögliche Werte: {@code POSITIVE}, {@code NEGATIVE}, {@code BOTH}.
     *
     * <p>Gilt für Heatmap und Sankey.
     *
     * @see de.fhdw.knn.visualization.WeightFilter
     */
    public String weightFilter = "BOTH";

    /**
     * Konvertiert diese TOML-Konfiguration in ein {@link HeatmapConfig}-Record.
     *
     * @return eine unveränderliche {@link HeatmapConfig}-Instanz mit den aktuellen Konfigurationswerten.
     * @throws IllegalArgumentException wenn {@code colorScheme} oder {@code weightFilter}
     *                                  keinem gültigen Enum-Wert entsprechen.
     * @see HeatmapConfig
     */
    public HeatmapConfig toConfig() {
        return new HeatmapConfig(
                interval,
                threshold,
                WeightFilter.valueOf(weightFilter.toUpperCase()),
                showWeights,
                normalizeColors,
                ColorScheme.valueOf(colorScheme.toUpperCase())
        );
    }
}