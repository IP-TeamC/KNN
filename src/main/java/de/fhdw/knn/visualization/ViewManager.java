package de.fhdw.knn.visualization;

import de.fhdw.knn.network.Network;
import javafx.application.Platform;

/**
 * Die Klasse {@code ViewManager} ist für die Verwaltung und Aktualisierung grafischer Ansichten
 * wie Heatmaps und Sankeyplots während des Trainingsprozesses eines neuronalen Netzwerks zuständig.
 * Sie übernimmt die Initialisierung und regelmäßige Aktualisierung dieser Ansichten in festgelegten Intervallen.
 */
public class ViewManager {

    /**
     * Stellt das Intervall (in Epochen) dar, in dem die Heatmap aktualisiert wird.
     *
     * <p>Ein Wert größer als 0 gibt die Periodizität der Aktualisierungen an (z. B. alle n Epochen), 
     * während ein Wert kleiner gleich 0 die Heatmap vollständig deaktiviert. 
     */
    private final int heatmapInterval;
    /**
     * Stellt das Intervall (in Epochen) dar, in dem das Sankeyplot aktualisiert wird.
     *
     * <p>Ein Wert größer als 0 gibt die Periodizität der Aktualisierungen an (z. B. alle n Epochen), 
     * während ein Wert kleiner gleich 0 das Sankeyplot vollständig deaktiviert. 
     */
    private final int sankeyInterval;

    /**
     * Stellt eine grafische Komponente dar, die für die Anzeige von Heatmap-Visualisierungen
     * der Gewichtsmatrizen eines neuronalen Netzwerks verantwortlich ist. Die Heatmap-Ansicht bietet
     * interaktive, scrollbare und zoombare Diagramme und unterstützt das Hinzufügen mehrerer Tabs für Visualisierungen.
     *
     *<p>Dieses Feld ist ein privates Mitglied der {@code ViewManager}-Klasse und wird initialisiert,
     * wenn die Heatmap-Funktion basierend auf dem angegebenen Aktualisierungsintervall aktiviert ist.
     * Es wird verwendet, um Heatmaps während des Trainingsprozesses eines neuronalen Netzwerks zu rendern,
     * und ermöglicht es Benutzern, Gewichtsverteilungen über Epochen hinweg visuell zu analysieren.
     * 
     * @see ViewManager
     */
    private HeatmapView heatmapView;
    /**
     * Stellt die Sankey-Visualisierungskomponente dar, die zur Anzeige von Netzwerkdaten
     * in einem grafischen Sankey-Diagramm verwendet wird. Dieses Feld ist eine Instanz von {@code SankeyView},
     * die Methoden zur Initialisierung, zum Rendern und zur Aktualisierung des Sankey-Plots basierend
     * auf den Netzwerkdaten und dem Epochenfortschritt bereitstellt.
     *
     * <p>Dieses Feld ist ein privates Mitglied der {@code ViewManager}-Klasse und wird initialisiert,
     * wenn die Sankeyplot-Funktion basierend auf dem angegebenen Aktualisierungsintervall aktiviert ist.
     * Es ist verantwortlich für die grafische Darstellung und Echtzeit-Aktualisierungen des
     * Sankey-Diagramms während des Trainings oder der Verarbeitung des Netzwerks.
     *
     * @see ViewManager
     */
    private SankeyView sankeyView;

    /**
     * Initialisiert den {@code ViewManager} mit angegebenen Intervallen für die Aktualisierung der Heatmap und dem Sankeyplot.
     * Wenn die angegebenen Intervalle größer als 0 sind, werden entsprechende Visualisierungen initialisiert.
     *
     * @param heatmapInterval Das Intervall, in dem die Heatmap aktualisiert werden soll. Der Wert 0 deaktiviert die Heatmap.
     * @param sankeyInterval Das Intervall, in dem das Sankeyplot aktualisiert werden soll. Der Wert 0 deaktiviert das Sankeyplot.
     * @param showWeights {@code true}, um die Werte der Gewichte in der Heatmap anzuzeigen.
     * @param normalizeColors {@code true}, um die Farbskala auf den tatsächlichen Wertebereich zu normalisieren;
     *                        {@code false} für einen festen Bereich von [-1, 1].
     * @param network Die Netzwerkdaten, die zur Initialisierung der Diagramme verwendet werden.
     * @param colorScheme Das {@link ColorScheme}, das für die Heatmap-Visualisierung verwendet werden soll.
     */
    public ViewManager(
            final int heatmapInterval,
            final int sankeyInterval,
            final boolean showWeights,
            final boolean normalizeColors,
            final ColorScheme colorScheme,
            Network network
    ) {
        this.heatmapInterval = heatmapInterval;
        this.sankeyInterval = sankeyInterval;

        if (this.heatmapInterval > 0) {
            this.heatmapView = new HeatmapView();
            this.heatmapView.setShowWeights(showWeights);
            this.heatmapView.setNormalizeColors(normalizeColors);
            this.heatmapView.setColorSchema(colorScheme);
            this.heatmapView.addEpoch("Epoche 0", new HeatmapData(network));
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

    /**
     * Aktualisiert die Heatmap und Sankeyplot basierend auf der aktuellen Epoche, den angegebenen Intervallen und der Gesamtzahl der Epochen.
     *
     * @param epoch Die aktuelle Epoche.
     * @param network Die für Aktualisierungen zu verwendenden Daten des neuronalen Netzwerks.
     * @param maxEpochs Die maximale Anzahl von Epochen im Trainingsprozess.
     */
    public void nextEpoch(final int epoch, Network network, int maxEpochs) {
        // Heatmap Update
        if (heatmapView != null && shouldUpdate(epoch, heatmapInterval, maxEpochs)) {
            heatmapView.addEpoch("Epoche " + epoch, new HeatmapData(network));
        }

        // Sankey Update
        if (sankeyView != null && shouldUpdate(epoch, sankeyInterval, maxEpochs)) {
            sankeyView.update(network, "Epoche " + epoch);
        }
    }

    /**
     * Erzwingt eine Aktualisierung der Heatmap bzw. des Sankeyplots bei einem EarlyStop.
     *
     * <p>Diese Methode wird in der Regel aufgerufen, wenn der Trainingsprozess vorzeitig durch
     * {@link de.fhdw.knn.trainer.stop.EarlyStopping} abgebrochen wird,
     * um den Endzustand in den Visualisierungen darzustellen.
     *
     * @param epoch Die aktuelle Epoche.
     * @param network Die für Aktualisierungen zu verwendenden Daten des neuronalen Netzwerks.
     */
    public void earlyStop(int epoch, Network network) {
        if (heatmapView != null) heatmapView.addEpoch("Epoche " + epoch + " (Early Stopping)", new HeatmapData(network));
        if (sankeyView != null) sankeyView.update(network, "Epoche " + epoch + " (Early Stopping)");
    }

    /**
     * Evaluiert, ob für eine bestimmte Epoche eine Aktualisierung ausgelöst werden soll.
     *
     * <p>The method evaluates three conditions:
     *
     * <br>1. Ob es sich um die erste Epoche handelt.
     * <br>2. Ob die Epoche ein Vielfaches des aktuellen Aktuallisierungsintervalles ist.
     * <br>3. Ob es sich um die letzte Epoche handelt.
     *
     * @param epoch Die aktuelle Epochennummer.
     * @param interval Das Intervall, in welchem eine Aktualisierung auftreten soll.
     * @param maxEpochs Die maximale Anzahl der Epochen.
     * @return {@code true} wenn eine Aktuallisierung für diese Epoche ausgelöst werden soll,
     *         {@code false} andernfalls.
     */
    private boolean shouldUpdate(int epoch, int interval, int maxEpochs) {
        return epoch == 1 || epoch % interval == 0 || epoch == maxEpochs;
    }
}
