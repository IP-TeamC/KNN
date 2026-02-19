package de.fhdw.knn.trainer.stop;

/**
 * Die StopFunction kann das Training vorzeitig beenden
 */
@FunctionalInterface
public interface StopFunction {

    /**
     * Beendet das Training nie vorzeitig.
     * Der Trainer trainiert immer für die angegebene Anzahl Epochen.
     */
    StopFunction NEVER = loss -> false;

    /**
     * Diese Methode war nach dem Durchlauf einer Epoche mit dem aktuellen Total-Loss aufgerufen.
     *
     * @return true, wenn das Training jetzt beendet werden soll
     */
    boolean isFinished(double loss);

}
