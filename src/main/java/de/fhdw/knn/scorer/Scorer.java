package de.fhdw.knn.scorer;

import de.fhdw.knn.data.DataSet;

/**
 * Interface für Scorer, die ein Netzwerk für einen Test-Datensatz anhand bestimmter Metriken evaluieren
 */
public interface Scorer {

    /**
     * Führt die Evaluierung eines Netzwerks für den Test-Datensatz anhand bestimmter Metriken durch
     * @param data Test-Datensatz
     */
    Score score(DataSet data);

}
